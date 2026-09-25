"""Shared native retrieval components for the demo and Haystack provider example.

Importing this module does not load models. Call load_collection() explicitly.
"""
import os
os.environ.setdefault('HAYSTACK_TELEMETRY_ENABLED', 'false')
from pathlib import Path
import hashlib
import importlib
import json
import time
from functools import lru_cache
from typing import Any
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
from haystack import Pipeline, Document, component
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
import math

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / 'site'
MODEL = 'hyper3labs/hyper3-clip-v1'
REVISION = '12a8d89022cec75a3fbac91047683231cbc0fa82'
BASELINE = 'openai/clip-vit-base-patch16'
BASE_REV = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
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

@component
class CollectionLoader:
    def __init__(self, document_store):
        self.document_store = document_store

    @component.output_types(documents=list[Document], trace=dict)
    def run(self):
        start=time.perf_counter()
        docs=self.document_store.filter_documents()
        return {'documents':docs,'trace':{'component':'CollectionLoader','images':len(docs),'ms':round((time.perf_counter()-start)*1000,2)}}

@component
class BriefEncoder:
    def __init__(self, models):
        self.models = models

    @component.output_types(native=Any, baseline=Any, trace=dict)
    def run(self,query:str):
        start=time.perf_counter(); n,b=self.models.text(query)
        return {'native':n,'baseline':b,'trace':{'component':'BriefEncoder','ms':round((time.perf_counter()-start)*1000,2),'query':query,'native_dimensions':513,'baseline_dimensions':512}}

@component
class RadialEncoder:
    def __init__(self, models):
        self.models = models

    @component.output_types(native=Any, baseline=Any, trace=dict)
    def run(self,query:str,radius_scale:float=1.0):
        query = query.strip()
        if not query:
            raise ValueError('Enter a nonempty description.')
        if not math.isfinite(radius_scale) or not .05 <= radius_scale <= 2.0:
            raise ValueError('radius_scale must be finite and between 0.05 and 2.0.')
        start=time.perf_counter()
        native,baseline,distance=self.models.radial(query,radius_scale)
        return {'native':native,'baseline':baseline,'trace':{'component':'RadialEncoder','ms':round((time.perf_counter()-start)*1000,2),'query':query,'radius_scale':radius_scale,'origin_distance':distance,'direction':'fixed','method':'expmap0(radius_scale * textual_alpha * raw_query)','scoring':'lorentz_inner_product','baseline':'unchanged cosine query'}}

@component
class ImageRanker:
    def __init__(self, models, mode):
        if mode not in {'cone', 'lorentz', 'cosine'}:
            raise ValueError('Unknown scoring mode: ' + mode)
        self.models = models
        self.mode = mode
    @component.output_types(results=list[dict], trace=dict)
    def run(self,documents:list[Document],embedding:Any):
        start=time.perf_counter(); models=self.models
        positions=[models.positions[d.id] for d in documents]
        with torch.inference_mode():
            if not positions: scores=[]
            elif self.mode=='cone': scores=models.hyper.cone_score(embedding,models.native[positions])[0]
            elif self.mode=='lorentz': scores=models.hyper.lorentz_inner_product(embedding,models.native[positions])[0]
            else: scores=(embedding@models.baseline[positions].T)[0]
        rows=[{'id':d.id,'score':float(s)} for d,s in zip(documents,scores)]
        rows.sort(key=lambda r:(-r['score'],r['id']))
        names={'cone':'Hyper3ConeRanker','lorentz':'Hyper3LorentzRanker','cosine':'CLIPCosineRanker'}
        return {'results':rows,'trace':{'component':names[self.mode],'candidates':len(rows),'ms':round((time.perf_counter()-start)*1000,2),'scoring':self.mode}}

def load_collection():
    """Encode the shipped gallery and index its records in a Haystack store."""
    models = Models()
    store = InMemoryDocumentStore(shared=False)
    indexing = Pipeline()
    indexing.add_component('asset_writer', DocumentWriter(document_store=store))
    indexing.run({'asset_writer': {'documents': [
        Document(id=r['id'], content=r['caption'], meta=r) for r in CORPUS
    ]}})
    return models, store


def build_radial_pipeline(models, document_store):
    """Build the same full-gallery Lorentz retrieval used for every slider state."""
    pipeline = Pipeline()
    pipeline.add_component('collection', CollectionLoader(document_store))
    pipeline.add_component('radial', RadialEncoder(models))
    pipeline.add_component('hyper3', ImageRanker(models, 'lorentz'))
    pipeline.add_component('baseline', ImageRanker(models, 'cosine'))
    pipeline.connect('collection.documents', 'hyper3.documents')
    pipeline.connect('collection.documents', 'baseline.documents')
    pipeline.connect('radial.native', 'hyper3.embedding')
    pipeline.connect('radial.baseline', 'baseline.embedding')
    return pipeline


def build_caption_pipeline(models, document_store):
    """Legacy caption-search endpoint; separate from the radial website route."""
    pipeline = Pipeline()
    pipeline.add_component('collection', CollectionLoader(document_store))
    pipeline.add_component('brief', BriefEncoder(models))
    pipeline.add_component('hyper3', ImageRanker(models, 'cone'))
    pipeline.add_component('baseline', ImageRanker(models, 'cosine'))
    pipeline.connect('collection.documents', 'hyper3.documents')
    pipeline.connect('collection.documents', 'baseline.documents')
    pipeline.connect('brief.native', 'hyper3.embedding')
    pipeline.connect('brief.baseline', 'baseline.embedding')
    return pipeline


def run_traversal(pipeline, query, radius_scale):
    """Run shared components and assemble the website's result/trace format."""
    start = time.perf_counter()
    out = pipeline.run(
        {'radial': {'query': query, 'radius_scale': radius_scale}},
        include_outputs_from={'collection', 'radial', 'hyper3', 'baseline'},
    )
    return {
        'query': query.strip(), 'radius_scale': radius_scale,
        'hyper3': out['hyper3']['results'], 'baseline': out['baseline']['results'],
        'total': len(out['hyper3']['results']),
        'elapsed_ms': round((time.perf_counter()-start)*1000),
        'trace': [out[k]['trace'] for k in ['collection', 'radial', 'hyper3', 'baseline']],
    }
