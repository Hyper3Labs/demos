"""Build the demos index and mount every sibling demo's site/ directory."""
from pathlib import Path
import html
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent.parent


def discover(root=ROOT):
    demos = []
    for descriptor in sorted(root.glob('*/demo.json')):
        folder = descriptor.parent
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', folder.name):
            raise ValueError(f'Invalid demo slug: {folder.name}')
        if folder.name == 'shared':
            raise ValueError('Reserved demo slug: shared')
        metadata = json.loads(descriptor.read_text())
        if not all(isinstance(metadata.get(k), str) and metadata[k].strip() for k in ['title', 'description']):
            raise ValueError(f'Missing title or description: {descriptor}')
        if not (folder / 'site/index.html').is_file():
            raise ValueError(f'Missing site/index.html: {folder.name}')
        if any(p.is_symlink() for p in (folder / 'site').rglob('*')):
            raise ValueError(f'Symlinks are not publishable: {folder.name}')
        demos.append((folder, metadata))
    if not demos:
        raise ValueError('No demo.json files found')
    return demos


def build():
    demos = discover()
    out = ROOT / 'dist'
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    shutil.copytree(ROOT / 'shared', out / 'shared', ignore=shutil.ignore_patterns('index.html', '_headers', '_redirects', '404.html'))
    for name in ['_headers', '_redirects', '404.html']:
        shutil.copy2(ROOT / 'shared' / name, out / name)
    cards = []
    for folder, metadata in demos:
        shutil.copytree(folder / 'site', out / folder.name, ignore=shutil.ignore_patterns('.DS_Store'))
        cards.append(f'<a class="demo-card" href="/{folder.name}/"><h2>{html.escape(metadata["title"])}</h2><p>{html.escape(metadata["description"])}</p><span>Open demo ↗</span></a>')
    template = (ROOT / 'shared/index.html').read_text()
    assert template.count('{{DEMO_CARDS}}') == 1
    (out / 'index.html').write_text(template.replace('{{DEMO_CARDS}}', '\n'.join(cards)))
    print('Built demos: ' + ', '.join(folder.name for folder, _ in demos))


if __name__ == '__main__':
    build()
