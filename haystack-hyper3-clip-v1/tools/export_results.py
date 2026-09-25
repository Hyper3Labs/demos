"""Regenerate recorded results using cached, pinned models and real Haystack."""
import json
import math
import sys
from pathlib import Path
import numpy as np
from haystack_pipeline import CORPUS, load_collection, build_radial_pipeline, run_traversal

models, store = load_collection()
pipeline = build_radial_pipeline(models, store)

ROOT = Path(__file__).resolve().parent.parent
site = ROOT / 'site'
examples = {
    'animals': 'A cat and dog napping together on a couch',
    'people': 'A man sitting on a park bench using a laptop',
    'coast': 'A surfer riding a wave with a sailboat in the background',
}
gallery = models.native.double().numpy()
curvature = float(models.hyper.curvature.detach())
for name, query in examples.items():
    positions = []
    baseline = None
    direction = None
    for i in range(101):
        radius = .05 * math.pow(40, i / 100)
        output = run_traversal(pipeline, query, radius)
        point, _, _ = models.radial(query, radius)
        q = point.numpy()[0]
        ray = q[1:] / np.linalg.norm(q[1:])
        if direction is None:
            direction = ray
        assert np.max(np.abs(direction - ray)) < 1e-12
        inner = gallery[:, 1:] @ q[1:] - gallery[:, 0] * q[0]
        distance = np.arccosh(np.maximum(1, -curvature * inner)) / np.sqrt(curvature)
        order = sorted(range(len(CORPUS)), key=lambda j: (distance[j], CORPUS[j]['id']))
        assert [CORPUS[j]['id'] for j in order] == [r['id'] for r in output['hyper3']]
        if baseline is None:
            baseline = output['baseline']
        assert output['baseline'] == baseline
        output.update(slider_position=i, execution='precomputed Haystack pipeline; no model inference in the browser')
        output['hyper3'] = output['hyper3'][:9]
        output['baseline'] = output['baseline'][:9]
        positions.append(output)
    if '--check-only' in sys.argv:
        existing = json.loads((site / f'data/{name}.json').read_text())['positions']
        assert len(existing) == len(positions) == 101
        for old, new in zip(existing, positions):
            for model in ['hyper3', 'baseline']:
                assert [r['id'] for r in old[model]] == [r['id'] for r in new[model]]
                np.testing.assert_allclose([r['score'] for r in old[model]], [r['score'] for r in new[model]], rtol=1e-6, atol=1e-7)
        continue
    (site / f'data/{name}.json').write_text(json.dumps({'query':query, 'positions':positions},ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
print('PASS: all 303 states, shared Haystack pipeline, independent distance ranking, fixed direction and CLIP; ' + ('recorded rankings/scores match' if '--check-only' in sys.argv else 'results written'))
