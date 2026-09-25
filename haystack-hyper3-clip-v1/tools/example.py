"""Runnable provider-page example; shares the website's Haystack components."""
from haystack_pipeline import build_radial_pipeline, load_collection

models, document_store = load_collection()
pipeline = build_radial_pipeline(models, document_store)
query = "A cat and dog napping together on a couch"

for position in (0, 50, 100):
    radius_scale = 0.05 * 40 ** (position / 100)
    result = pipeline.run({
        "radial": {"query": query, "radius_scale": radius_scale}
    })
    print(f"Slider {position}: radius_scale={radius_scale:.6f}")
    for name in ("hyper3", "baseline"):
        print(name, [row["id"] for row in result[name]["results"][:9]])
