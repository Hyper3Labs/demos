"""Regenerate recorded results using cached, pinned models and real Haystack."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import server

ROOT = Path(__file__).resolve().parent.parent
site = ROOT / 'site'
examples = {
    'animals': 'A cat and dog napping together on a couch',
    'people': 'A man sitting on a park bench using a laptop',
    'coast': 'A surfer riding a wave with a sailboat in the background',
}
gallery = server.models.native.double().numpy()
curvature = float(server.models.hyper.curvature.detach())
for name, query in examples.items():
    positions = []
    baseline = None
    direction = None
    for i in range(101):
        radius = .05 * math.pow(40, i / 100)
        output = server.traverse(server.TraversalRequest(query=query, radius_scale=radius))
        point, _, _ = server.models.radial(query, radius)
        q = point.numpy()[0]
        ray = q[1:] / np.linalg.norm(q[1:])
        if direction is None:
            direction = ray
        assert np.max(np.abs(direction - ray)) < 1e-12
        inner = gallery[:, 1:] @ q[1:] - gallery[:, 0] * q[0]
        distance = np.arccosh(np.maximum(1, -curvature * inner)) / np.sqrt(curvature)
        order = sorted(range(len(server.CORPUS)), key=lambda j: (distance[j], server.CORPUS[j]['id']))
        assert [server.CORPUS[j]['id'] for j in order] == [r['id'] for r in output['hyper3']]
        if baseline is None:
            baseline = output['baseline']
        assert output['baseline'] == baseline
        output.update(slider_position=i, execution='precomputed Haystack pipeline; no model inference in the browser')
        output['hyper3'] = output['hyper3'][:9]
        output['baseline'] = output['baseline'][:9]
        positions.append(output)
    if '--check-only' in sys.argv:
        existing = json.loads((site / f'data/{name}.json').read_text())['positions']
        for old, new in zip(existing, positions):
            for model in ['hyper3', 'baseline']:
                assert [r['id'] for r in old[model]] == [r['id'] for r in new[model]]
        continue
    (site / f'data/{name}.json').write_text(json.dumps({'query':query, 'positions':positions},ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
print('Exported and verified all 303 slider states')
