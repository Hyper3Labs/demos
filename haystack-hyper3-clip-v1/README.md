# Hyper3-CLIP v1 × Haystack

[Open the demo](https://demos.hyper3labs.com/haystack-hyper3-clip-v1/).

Choose Animals, People or Coast and move the slider. The description and spatial direction stay fixed; Hyper3's query point moves along one ray in hyperbolic space. Images are ranked by distance to that point. CLIP ranks the same gallery by cosine similarity using its unchanged query embedding.

The 101 slider positions use `radius_scale = 0.05 * 40 ** (position / 100)`. The origin exponential map scales the query's geodesic distance from the origin while keeping its direction fixed. Native Lorentz inner product gives the same ranking as increasing hyperbolic distance. “Broad” and “Specific” are exploration labels, not a validated semantic hierarchy or model-quality result.

## Website source

- `site/index.html`: interface and How it works explanation.
- `site/static/app.js`: example selection, slider playback and image previews.
- `site/static/style.css`: styling.
- `site/data/`: gallery metadata and 303 recorded result states, including pipeline traces.
- `site/assets/`: 42 unedited photographs.
- `site/credits.html`: titles, photographer attribution, source links and individual licenses.

The website displays results computed by a real Haystack pipeline in advance. It does not execute model inference on slider input. The explanation was authored with Claude Opus 5.5 and checked against the implementation. Edit the HTML directly for wording changes.

## Regenerate results (optional)

Building or hosting the website does not require these dependencies. Regeneration uses the published Hyper3-CLIP v1 and CLIP checkpoints:

- `hyper3labs/hyper3-clip-v1` at `12a8d89022cec75a3fbac91047683231cbc0fa82`
- `openai/clip-vit-base-patch16` at `57c216476eefef5ab752ec549e440a49ae4ae5f3`

Use Python 3.12 and install `requirements-inference.txt` (Haystack 3.2.0). Obtain model access on the Hyper3-CLIP model page, authenticate, and cache both pinned checkpoints. Run from this demo's directory:

```sh
pip install -r requirements-inference.txt
hf auth login
hf download hyper3labs/hyper3-clip-v1 --revision 12a8d89022cec75a3fbac91047683231cbc0fa82
hf download openai/clip-vit-base-patch16 --revision 57c216476eefef5ab752ec549e440a49ae4ae5f3
python tools/example.py
```

The provider-page example in `tools/example.py` and the website's result exporter both call `build_radial_pipeline` in [`tools/haystack_pipeline.py`](tools/haystack_pipeline.py). That shared module loads native 513-coordinate image embeddings, writes gallery records through Haystack's `DocumentWriter`, and builds these custom Haystack components:

```text
CollectionLoader ── documents ──┬── Hyper3LorentzRanker
                               └── CLIPCosineRanker
RadialEncoder ── native query ───── Hyper3LorentzRanker
              └─ CLIP query ─────── CLIPCosineRanker
```

`ImageRanker(models, "lorentz")` is the Hyper3 ranker; `ImageRanker(models, "cosine")` is the CLIP comparison. Hyper3 uses 512 spatial coordinates plus one hyperboloid coordinate, with radius preserved. Its score is the Lorentz inner product, `-q[0]*x[0] + dot(q[1:], x[1:])`; larger scores give smaller hyperbolic distances. It evaluates every image in this small gallery. It does not use a cosine candidate prefilter or the normalized Sentence Transformers adapter.

These are example components shipped in this repository, not built-in Haystack components or a separately published integration package. The optional HTTP server imports the same module. Models are loaded explicitly by `load_collection()` and cached locally; credentials and weights are not shipped with the website.

To verify or regenerate the recorded website results:

```sh
python tools/export_results.py --check-only  # Compare rankings without changing data
python tools/export_results.py               # Regenerate data
```

Image embeddings are cached in the ignored `.cache/` directory. Export verifies constant direction, independent hyperbolic-distance ranking and exact CLIP invariance before writing results. `--check-only` also compares all 303 states' image ordering and scores with the recorded website data. The website's recorded data is included, so anyone can build the demo without model access.

## Image attribution

The 42 images were retained only when both COCO license metadata and current Flickr attribution records supported commercial reuse under CC BY, BY-SA or BY-ND 2.0. Files are unmodified COCO dataset images displayed with `object-fit: contain`. Individual licenses and original photo links are in [credits](site/credits.html) and `site/data/library.json`. The photographers do not endorse the demo. COCO annotations are CC BY 4.0; image descriptions are drawn from COCO and SugarCrepe.
