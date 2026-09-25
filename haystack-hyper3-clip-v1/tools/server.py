"""Optional local HTTP wrapper around the shared Haystack retrieval pipeline."""
from datetime import date
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from haystack_pipeline import (
    SITE, MODEL, REVISION, BASELINE, BASE_REV, CORPUS,
    load_collection, build_radial_pipeline, run_traversal,
)

LOCK = threading.Lock()
models, store = load_collection()
radial_pipeline = build_radial_pipeline(models, store)

class TraversalRequest(BaseModel):
    query:str=Field(min_length=1,max_length=600)

    radius_scale:float=Field(default=0.15,ge=0.05,le=2.0,allow_inf_nan=False)

app=FastAPI(title='Hyper3-CLIP radial explorer')
app.mount('/assets',StaticFiles(directory=SITE/'assets'),name='assets')


@app.get('/')
def home():
    html = (SITE/'index.html').read_text().replace('/haystack-hyper3-clip-v1/', '/')
    return HTMLResponse(html)

@app.get('/static/runtime.js')
def runtime():
    return Response('window.RADIAL_DEMO = {precomputed: false, basePath: "/"};', media_type='application/javascript', headers={'Cache-Control':'no-store'})

app.mount('/static',StaticFiles(directory=SITE/'static'),name='static')

@app.get('/credits')
def credits(): return HTMLResponse((SITE/'credits.html').read_text().replace('/haystack-hyper3-clip-v1/', '/'))

@app.get('/api/library')
def library():
    return {'assets':CORPUS,'today':date.today().isoformat(),'models':{'hyper3':MODEL,'revision':REVISION,'baseline':BASELINE,'baseline_revision':BASE_REV}}

@app.post('/api/traverse')
def traverse(req:TraversalRequest):
    query=req.query.strip()
    if not query: raise HTTPException(422,'Choose an example or enter a description.')
    with LOCK:
        return run_traversal(radial_pipeline, query, req.radius_scale)

@app.get('/api/health')
def health(): return {'status':'ready','assets':len(CORPUS),'haystack_pipeline':True}

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=7864)
