"""Campaign Asset Finder: local Haystack pipelines, real native model inference."""
import os
os.environ.setdefault('HAYSTACK_TELEMETRY_ENABLED', 'false')
from pathlib import Path
import hashlib
import importlib
import json
import time
import threading
from functools import lru_cache
from datetime import date
from typing import Any
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from haystack import Pipeline, Document, component
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'
MODEL = 'hyper3labs/hyper3-clip-v1'
REVISION = '12a8d89022cec75a3fbac91047683231cbc0fa82'
BASELINE = 'openai/clip-vit-base-patch16'
BASE_REV = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
LOCK = threading.Lock()
torch.set_num_threads(4)
CORPUS = json.loads((SITE/'data/library.json').read_text())
if isinstance(CORPUS, dict): CORPUS = CORPUS['assets']
BY_ID = {r['id']: r for r in CORPUS}

class Models:
    def __init__(self):
        print('Loading Hyper3-CLIP and CLIP baseline', flush=True)
        self.hyper = AutoModel.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True).eval()
        self.expmap = importlib.import_module(type(self.hyper).__module__)._lorentz_exp_map0
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        self.clip = CLIPModel.from_pretrained(BASELINE, revision=BASE_REV, local_files_only=True).eval()
        self.processor = CLIPProcessor.from_pretrained(BASELINE, revision=BASE_REV, local_files_only=True)
        identity = {'model': REVISION, 'baseline': BASE_REV, 'images':[(r['id'], hashlib.sha256((SITE/r['path']).read_bytes()).hexdigest()) for r in CORPUS]}
        digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        cache = ROOT/'.cache/embeddings.pt'
        cache.parent.mkdir(exist_ok=True)
        data = torch.load(cache, weights_only=True) if cache.exists() else {}
        if data.get('identity') != digest:
            native, baseline = [], []
            with torch.inference_mode():
                for start in range(0,len(CORPUS),8):
                    ims = [Image.open(SITE/r['path']).convert('RGB') for r in CORPUS[start:start+8]]
                    px = torch.stack([self.hyper.preprocess_image(im) for im in ims])
                    native.append(self.hyper.encode_image_lorentz(px).cpu())
                    baseline.append(F.normalize(self.clip.get_image_features(**self.processor(images=ims,return_tensors='pt')),dim=-1).cpu())
                    print(f'Indexed {min(start+8,len(CORPUS))}/{len(CORPUS)} photos',flush=True)
            data = {'identity':digest, 'native':torch.cat(native),'baseline':torch.cat(baseline)}
            torch.save(data,cache)
        self.native,self.baseline=data['native'],data['baseline']
        self.positions={r['id']:i for i,r in enumerate(CORPUS)}

    @lru_cache(maxsize=128)
    @torch.inference_mode()
    def query_vectors(self,query):
        tokens=self.tokenizer([query],padding=True,truncation=True,max_length=self.hyper.config.max_text_length,return_tensors='pt')
        tangent=self.hyper.encode_text(**tokens,normalize=False)
        tokens2=self.processor(text=[query],padding=True,truncation=True,max_length=77,return_tensors='pt')
        baseline=F.normalize(self.clip.get_text_features(**tokens2),dim=-1)
        return tangent,baseline

    @torch.inference_mode()
    def text(self,query):
        tangent,baseline=self.query_vectors(query)
        native=self.expmap(tangent.float()*self.hyper.textual_alpha.float().exp(),self.hyper.curvature)
        return native,baseline

    @torch.inference_mode()
    def radial(self,query,radius_scale):
        tangent,baseline=self.query_vectors(query)
        scaled=tangent.double()*self.hyper.textual_alpha.double().exp()*radius_scale
        # The release helper forces float32. Keep the radial path in float64
        # so near-origin cosh/radius differences remain numerically resolvable.
        sqrt_c=self.hyper.curvature.double().sqrt()
        theta=sqrt_c*scaled.norm(dim=-1,keepdim=True)
        sinhc=torch.where(theta.abs()<1e-8,1+theta.square()/6,torch.sinh(theta)/theta.clamp_min(1e-12))
        native=torch.cat([torch.cosh(theta)/sqrt_c,sinhc*scaled],dim=-1)
        return native,baseline,float(scaled.norm())

models = Models()
store = InMemoryDocumentStore(shared=False)
indexing = Pipeline()
indexing.add_component('asset_writer',DocumentWriter(document_store=store))
indexing.run({'asset_writer':{'documents':[Document(id=r['id'],content=r['caption'],meta=r) for r in CORPUS]}})

@component
class CollectionLoader:
    @component.output_types(documents=list[Document], trace=dict)
    def run(self):
        start=time.perf_counter()
        docs=store.filter_documents()
        return {'documents':docs,'trace':{'component':'CollectionLoader','images':len(docs),'ms':round((time.perf_counter()-start)*1000,2)}}

@component
class BriefEncoder:
    @component.output_types(native=Any, baseline=Any, trace=dict)
    def run(self,query:str):
        start=time.perf_counter(); n,b=models.text(query)
        return {'native':n,'baseline':b,'trace':{'component':'BriefEncoder','ms':round((time.perf_counter()-start)*1000,2),'query':query,'native_dimensions':513,'baseline_dimensions':512}}

@component
class RadialEncoder:
    @component.output_types(native=Any, baseline=Any, trace=dict)
    def run(self,query:str,radius_scale:float):
        start=time.perf_counter()
        native,baseline,distance=models.radial(query,radius_scale)
        return {'native':native,'baseline':baseline,'trace':{'component':'RadialEncoder','ms':round((time.perf_counter()-start)*1000,2),'query':query,'radius_scale':radius_scale,'origin_distance':distance,'direction':'fixed','method':'expmap0(radius_scale * textual_alpha * raw_query)','scoring':'lorentz_inner_product','baseline':'unchanged cosine query'}}

@component
class ImageRanker:
    def __init__(self,mode): self.mode=mode
    @component.output_types(results=list[dict], trace=dict)
    def run(self,documents:list[Document],embedding:Any):
        start=time.perf_counter(); positions=[models.positions[d.id] for d in documents]
        with torch.inference_mode():
            if not positions: scores=[]
            elif self.mode=='cone': scores=models.hyper.cone_score(embedding,models.native[positions])[0]
            elif self.mode=='lorentz': scores=models.hyper.lorentz_inner_product(embedding,models.native[positions])[0]
            else: scores=(embedding@models.baseline[positions].T)[0]
        rows=[{'id':d.id,'score':float(s)} for d,s in zip(documents,scores)]
        rows.sort(key=lambda r:(-r['score'],r['id']))
        names={'cone':'Hyper3ConeRanker','lorentz':'Hyper3LorentzRanker','cosine':'CLIPCosineRanker'}
        return {'results':rows,'trace':{'component':names[self.mode],'candidates':len(rows),'ms':round((time.perf_counter()-start)*1000,2),'scoring':self.mode}}

pipeline=Pipeline()
pipeline.add_component('collection',CollectionLoader())
pipeline.add_component('brief',BriefEncoder())
pipeline.add_component('hyper3',ImageRanker('cone'))
pipeline.add_component('baseline',ImageRanker('cosine'))
pipeline.connect('collection.documents','hyper3.documents')
pipeline.connect('collection.documents','baseline.documents')
pipeline.connect('brief.native','hyper3.embedding')
pipeline.connect('brief.baseline','baseline.embedding')

radial_pipeline=Pipeline()
radial_pipeline.add_component('collection',CollectionLoader())
radial_pipeline.add_component('radial',RadialEncoder())
radial_pipeline.add_component('hyper3',ImageRanker('lorentz'))
radial_pipeline.add_component('baseline',ImageRanker('cosine'))
radial_pipeline.connect('collection.documents','hyper3.documents')
radial_pipeline.connect('collection.documents','baseline.documents')
radial_pipeline.connect('radial.native','hyper3.embedding')
radial_pipeline.connect('radial.baseline','baseline.embedding')

class SearchRequest(BaseModel):
    query:str=Field(min_length=1,max_length=600)

class TraversalRequest(SearchRequest):
    radius_scale:float=Field(default=0.15,ge=0.05,le=2.0,allow_inf_nan=False)

class CaptionRequest(BaseModel):
    asset_id:str
    first:str=Field(min_length=1,max_length=600)
    second:str=Field(min_length=1,max_length=600)

app=FastAPI(title='Campaign Asset Finder')
app.mount('/assets',StaticFiles(directory=SITE/'assets'),name='assets')
app.mount('/static',StaticFiles(directory=SITE/'static'),name='static')

@app.get('/')
def home(): return FileResponse(SITE/'index.html')

@app.get('/api/library')
def library():
    return {'assets':CORPUS,'today':date.today().isoformat(),'models':{'hyper3':MODEL,'revision':REVISION,'baseline':BASELINE,'baseline_revision':BASE_REV}}

@app.post('/api/search')
def search(req:SearchRequest):
    if not req.query.strip(): raise HTTPException(422,'Enter a visual brief.')
    start=time.perf_counter()
    with LOCK:
        out=pipeline.run({'brief':{'query':req.query.strip()}},include_outputs_from={'collection','brief','hyper3','baseline'})
    trace=[out[k]['trace'] for k in ['collection','brief','hyper3','baseline']]
    return {'hyper3':out['hyper3']['results'],'baseline':out['baseline']['results'],'eligible':len(out['hyper3']['results']),'total':len(CORPUS),'elapsed_ms':round((time.perf_counter()-start)*1000),'trace':trace}

@app.post('/api/traverse')
def traverse(req:TraversalRequest):
    query=req.query.strip()
    if not query: raise HTTPException(422,'Choose an example or enter a description.')
    start=time.perf_counter()
    with LOCK:
        out=radial_pipeline.run({'radial':{'query':query,'radius_scale':req.radius_scale}},include_outputs_from={'collection','radial','hyper3','baseline'})
    return {'query':query,'radius_scale':req.radius_scale,'hyper3':out['hyper3']['results'],'baseline':out['baseline']['results'],'total':len(CORPUS),'elapsed_ms':round((time.perf_counter()-start)*1000),'trace':[out[k]['trace'] for k in ['collection','radial','hyper3','baseline']]}

@app.post('/api/compare-captions')
def compare(req:CaptionRequest):
    if req.asset_id not in BY_ID: raise HTTPException(404,'Unknown asset')
    if not req.first.strip() or not req.second.strip(): raise HTTPException(422,'Enter two descriptions.')
    i=models.positions[req.asset_id]
    with LOCK,torch.inference_mode():
        vectors=[models.text(t.strip()) for t in [req.first,req.second]]
        n=torch.cat([v[0] for v in vectors]); b=torch.cat([v[1] for v in vectors])
        hs=models.hyper.cone_score(n,models.native[i:i+1])[:,0].tolist()
        bs=(b@models.baseline[i:i+1].T)[:,0].tolist()
    def result(v): return {'scores':v,'preferred':'A' if v[0]>v[1] else 'B' if v[1]>v[0] else 'Tie','margin':v[0]-v[1]}
    return {'hyper3':result(hs),'baseline':result(bs)}

@app.get('/api/health')
def health(): return {'status':'ready','assets':len(CORPUS),'haystack_pipeline':True}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=7864)
