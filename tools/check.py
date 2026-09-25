"""Validate publishable paths, links, and the recorded retrieval demo."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import json
import math
from build import ROOT, discover

out = ROOT / 'dist'

class Links(HTMLParser):
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key not in ('href', 'src') or not value or not value.startswith('/') or value.startswith('//'):
                continue
            path = out / unquote(urlsplit(value).path).lstrip('/')
            assert path.resolve().is_relative_to(out.resolve()), value
            assert path.exists() or path.with_suffix('.html').is_file(), value

for page in out.rglob('*.html'):
    Links().feed(page.read_text())
for folder, metadata in discover():
    assert (out / folder.name / 'index.html').is_file()
    assert f'href="/{folder.name}/"' in (out / 'index.html').read_text()
for private in ['tools', '.git', '.github', '.env', 'requirements-inference.txt']:
    assert not (out / private).exists()

site = out / 'haystack-hyper3-clip-v1'
assert '"precomputed": true' in (site / 'static/runtime.js').read_text(), 'Deployed demo must stay precomputed'
library = json.loads((site / 'data/library.json').read_text())
ids = {a['id'] for a in library['assets']}
assert len(ids) == 42
for asset in library['assets']:
    assert (site / asset['path']).is_file()
    assert asset['author'] and asset['license_url'].startswith('https://creativecommons.org/licenses/')
    assert '-nc' not in asset['license_url']
for example in ['animals', 'people', 'coast']:
    data = json.loads((site / f'data/{example}.json').read_text())
    assert len(data['positions']) == 101
    first = data['positions'][0]['baseline']
    for i, row in enumerate(data['positions']):
        assert row['slider_position'] == i and row['total'] == len(ids)
        assert math.isclose(row['radius_scale'], 0.05 * 40 ** (i / 100), rel_tol=1e-12)
        assert row['trace'][2]['scoring'] == 'lorentz'
        assert row['baseline'] == first
        for model in ['hyper3', 'baseline']:
            assert len(row[model]) == 9 and all(r['id'] in ids for r in row[model])
print('PASS: site links, demo registration, photo credits, 303 recorded states and fixed CLIP')
