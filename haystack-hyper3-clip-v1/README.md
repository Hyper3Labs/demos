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

Use Python 3.12 and install `requirements-inference.txt`. Obtain any required model access and cache both pinned checkpoints using Hugging Face tooling first; the inference source intentionally uses `local_files_only=True`. Then run:

```sh
python tools/export_results.py --check-only  # Compare rankings without changing data
python tools/export_results.py               # Regenerate data
```

The code in `tools/server.py` supplies the genuine Haystack document store, radial encoder and rankers. Image embeddings are cached in the ignored `.cache/` directory. Export verifies constant direction, independent hyperbolic-distance ranking and exact CLIP invariance before writing results. Model access credentials and model files are not part of this repository. The website's recorded data is included, so anyone can build the demo without model access.

## Image attribution

The 42 images were retained only when both COCO license metadata and current Flickr attribution records supported commercial reuse under CC BY, BY-SA or BY-ND 2.0. Files are unmodified COCO dataset images displayed with `object-fit: contain`. Individual licenses and original photo links are in [credits](site/credits.html) and `site/data/library.json`. The photographers do not endorse the demo. COCO annotations are CC BY 4.0; image descriptions are drawn from COCO and SugarCrepe.
