const $ = id => document.getElementById(id);
const examples = [
  {id: 'animals', label: 'Animals', query: 'A cat and dog napping together on a couch'},
  {id: 'people', label: 'People', query: 'A man sitting on a park bench using a laptop'},
  {id: 'coast', label: 'Coast', query: 'A surfer riding a wave with a sailboat in the background'},
];
const TOP_K = 9;
const staticDemo = window.RADIAL_DEMO?.precomputed === true;
const basePath = window.RADIAL_DEMO?.basePath || '/';
const recordedExamples = new Map();
let byId = {}, selected = examples[0], ready = false, version = 0, request = null, timer = null;
let baselineKey = '', lastImageButton = null;
const imageUrl = a => basePath + a.path.split('/').map(encodeURIComponent).join('/');
const scaleAt = value => 0.05 * Math.pow(40, Number(value) / 100);

function updateSlider() {
  const value = $('breadth').value;
  $('breadth').style.setProperty('--position', value + '%');
  $('breadth').setAttribute('aria-valuetext', value + ' percent toward specific');
}
function renderExamples() {
  $('examples').replaceChildren(...examples.map(example => {
    const button = document.createElement('button');
    button.textContent = example.label;
    button.disabled = !ready;
    button.dataset.example = example.id;
    button.setAttribute('aria-pressed', String(example.id === selected.id));
    button.onclick = () => {
      selected = example;
      $('breadth').value = 30;
      updateSlider(); renderExamples(); scheduleSearch(true);
    };
    return button;
  }));
}
function renderGrid(id, rows) {
  const grid = $(id);
  const signature = rows.slice(0, TOP_K).map(row => row.id).join('|');
  if (grid.dataset.signature === signature) return;
  grid.dataset.signature = signature;
  grid.replaceChildren(...rows.slice(0, TOP_K).map(row => {
    const asset = byId[row.id];
    const button = document.createElement('button');
    button.className = 'image-button'; button.dataset.asset = asset.id;
    button.setAttribute('aria-label', 'Open image: ' + asset.caption);
    const image = document.createElement('img');
    image.src = imageUrl(asset); image.alt = asset.caption; image.loading = 'eager';
    button.append(image);
    button.onclick = () => {
      lastImageButton = button;
      $('detailImage').src = imageUrl(asset); $('detailImage').alt = asset.caption;
      $('sourceLink').href = asset.source_url;
      $('imageCredit').hidden = !asset.author;
      if (asset.author) {
        $('imageAuthor').textContent = asset.photo_title + ' — ' + asset.author;
        $('imageLicense').textContent = asset.source_license;
        $('imageLicense').href = asset.license_url;
      }
      $('imageDialog').showModal();
    };
    return button;
  }));
}
function scheduleSearch(immediate = false) {
  clearTimeout(timer);
  const current = ++version;
  request?.abort();
  $('results').setAttribute('aria-busy', 'true');
  $('hyperStatus').textContent = 'Updating…';
  $('error').hidden = true;
  if (immediate) search(current);
  else timer = setTimeout(() => search(current), 90);
}
async function search(current) {
  const example = selected;
  const radiusScale = scaleAt($('breadth').value);
  const controller = new AbortController(); request = controller;
  try {
    let data;
    if (staticDemo) {
      if (!recordedExamples.has(example.id)) {
        const response = await fetch(basePath + 'data/' + example.id + '.json', {signal: controller.signal});
        if (!response.ok) throw Error('Unable to load results. Move the slider to retry.');
        recordedExamples.set(example.id, await response.json());
      }
      data = recordedExamples.get(example.id).positions[Number($('breadth').value)];
    } else {
      const response = await fetch('/api/traverse', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: example.query, radius_scale: radiusScale}),
        signal: controller.signal,
      });
      data = await response.json();
      if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Unable to load results. Move the slider to retry.');
    }
    if (current !== version) return;
    renderGrid('hyperGrid', data.hyper3);
    // Preserve the baseline DOM throughout a slider gesture. Only a new example updates it.
    if (baselineKey !== example.id) {renderGrid('baselineGrid', data.baseline); baselineKey = example.id;}
    $('fixedQuery').textContent = example.query;
    $('radiusValue').textContent = radiusScale.toFixed(3) + '× the encoded query radius';
    $('trace').textContent = JSON.stringify(data.trace, null, 2);
    $('hyperStatus').textContent = '';
    $('announcement').textContent = 'Updated Hyper3 images for ' + example.label + '. CLIP remains fixed.';
  } catch (error) {
    if (error.name === 'AbortError' || current !== version) return;
    $('error').textContent = error.message; $('error').hidden = false;
    $('hyperStatus').textContent = 'Try again';
  } finally {
    if (current === version) {$('results').setAttribute('aria-busy', 'false'); if (request === controller) request = null;}
  }
}
$('breadth').addEventListener('input', () => {updateSlider(); scheduleSearch();});
$('pipelineBtn').onclick = () => $('pipelineDialog').showModal();
$('closePipeline').onclick = () => $('pipelineDialog').close();
$('closeImage').onclick = () => $('imageDialog').close();
$('imageDialog').addEventListener('close', () => {if (lastImageButton?.isConnected) lastImageButton.focus();});
for (const id of ['pipelineDialog', 'imageDialog']) $(id).addEventListener('click', event => {
  if (event.target !== $(id)) return;
  const rect = $(id).getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $(id).close();
});
(async () => {
  renderExamples(); updateSlider();
  for (const id of ['hyperGrid', 'baselineGrid']) $(id).innerHTML = '<div class="placeholder"></div>'.repeat(TOP_K);
  try {
    const response = await fetch(staticDemo ? basePath + 'data/library.json' : '/api/library');
    if (!response.ok) throw Error('Unable to load the image collection. Reload to retry.');
    const data = await response.json(); byId = Object.fromEntries(data.assets.map(a => [a.id, a]));
    ready = true; renderExamples();
    $('breadth').disabled = false;
    scheduleSearch(true);
  } catch (error) {$('error').textContent = error.message; $('error').hidden = false; $('hyperStatus').textContent = 'Unavailable'; $('results').setAttribute('aria-busy', 'false');}
})();
