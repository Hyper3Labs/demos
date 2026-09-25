"""Optional local HTTP wrapper around the shared Haystack retrieval pipeline."""
from datetime import date
import threading
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from haystack_pipeline import (
    SITE, MODEL, REVISION, BASELINE, BASE_REV, CORPUS, BY_ID,
    load_collection, build_radial_pipeline, build_caption_pipeline, run_traversal,
)

LOCK = threading.Lock()
models, store = load_collection()
pipeline = build_caption_pipeline(models, store)
radial_pipeline = build_radial_pipeline(models, store)

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
    with LOCK:
        return run_traversal(radial_pipeline, query, req.radius_scale)

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
