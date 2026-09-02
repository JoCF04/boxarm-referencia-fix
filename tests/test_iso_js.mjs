import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

function loadViewer(overlaps = [], provisionalBoxes = []) {
  const elements = new Map();
  const noop = () => {};
  const drawnTexts = [];
  const context2d = new Proxy({}, {
    get(target, key) {
      if (key === 'measureText') {
        return (value) => ({ width: String(value).length * 8 });
      }
      if (key === 'createLinearGradient' || key === 'createRadialGradient') {
        return () => ({ addColorStop() {} });
      }
      if (key === 'fillText') {
        return (value) => drawnTexts.push(String(value));
      }
      return target[key] || noop;
    },
    set(target, key, value) { target[key] = value; return true; },
  });
  const element = (id) => {
    if (!elements.has(id)) {
      const listeners = new Map();
      const classes = new Set();
      elements.set(id, {
        textContent: '',
        style: {},
        clientWidth: 960,
        clientHeight: 720,
        width: 0,
        height: 0,
        classList: {
          add(...names) { names.forEach((name) => classes.add(name)); },
          remove(...names) { names.forEach((name) => classes.delete(name)); },
          contains(name) { return classes.has(name); },
        },
        getContext() { return context2d; },
        setPointerCapture() {},
        getBoundingClientRect() { return { left: 0, top: 0, width: 960, height: 720 }; },
        addEventListener(type, handler) { listeners.set(type, handler); },
        dispatch(type, event = {}) { listeners.get(type)?.(event); },
      });
    }
    return elements.get(id);
  };

  const requests = [];
  let rendered = 0;
  const windowListeners = new Map();
  const payload = {
    seq: 1,
    scene: {
      boxes: [
        { cell: 0, level: 0, u: 0.4, v: 0.4, side_a: 0.2, side_b: 0.2, z0: 0, height: 0.1 },
        { cell: 1, level: 1, u: 0.6, v: 0.6, side_a: 0.2, side_b: 0.2, z0: 0.1, height: 0.1 },
      ],
      overlaps, provisional_boxes: provisionalBoxes,
      level_tops: [0, 0.1, 0.2], total_height: 0.2,
      total: 2, initial: 1, placed: 1, levels: 2,
    },
    view: { pallet_width: 1, pallet_length: 1, fill_margin: 0.85, level_gap_ratio: 0 },
    colors: { background: [20, 20, 20], title: [0, 255, 255], pallet: [0, 200, 255], levels: [[255, 0, 0]] },
  };
  const browserWindow = {
    devicePixelRatio: 1,
    addEventListener(type, handler) { windowListeners.set(type, handler); },
    requestAnimationFrame(callback) { rendered += 1; callback(); },
    setTimeout() {},
  };
  const context = {
    document: {
      body: { dataset: { camId: '3', az0: '35', el0: '35' } },
      getElementById: element,
      querySelectorAll() { return []; },
    },
    window: browserWindow,
    fetch(url) {
      requests.push(url);
      return Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
    },
  };

  vm.runInNewContext(fs.readFileSync('src/boxarm/web/static/iso.js', 'utf8'), context);
  return { element, requests, rendered: () => rendered, drawnTexts: () => drawnTexts };
}

test('drag follows the pointer direction without posting camera angles', async () => {
  const viewer = loadViewer();
  await new Promise((done) => setImmediate(done));
  const canvas = viewer.element('iso');
  const before = viewer.rendered();

  canvas.dispatch('pointerdown', { clientX: 0, clientY: 0, pointerId: 1, preventDefault() {} });
  canvas.dispatch('pointermove', { clientX: 10, clientY: 5 });
  canvas.dispatch('pointermove', { clientX: 20, clientY: 10 });

  assert.deepEqual(viewer.requests, ['/cam/3/iso/scene']);
  assert.equal(viewer.element('az').textContent, '27');
  assert.equal(viewer.element('el').textContent, '39');
  assert.ok(viewer.rendered() > before, 'pointer movement schedules local canvas frames');
});

test('layer control cycles occupied levels and returns to all layers', async () => {
  const viewer = loadViewer();
  await new Promise((done) => setImmediate(done));
  const layer = viewer.element('capa');

  assert.equal(layer.textContent, 'TODAS LAS CAPAS');
  layer.dispatch('click');
  assert.equal(layer.textContent, 'NIVEL 0');
  layer.dispatch('click');
  assert.equal(layer.textContent, 'NIVEL 1');
  layer.dispatch('click');
  assert.equal(layer.textContent, 'TODAS LAS CAPAS');
});

test('compass-cube widget replaced the ISO/FRONTAL/LATERAL/PLANTA buttons', async () => {
  const html = fs.readFileSync('src/boxarm/web/templates/camera.html', 'utf8');
  assert.doesNotMatch(html, /class="vista(?:\s|")/);
  assert.match(html, /id="isometrica"[^>]*>ISOMÉTRICA</);
});

test('clicking the cube\'s top face focuses the highest occupied level, clicking the ring center restores all', async () => {
  const viewer = loadViewer();
  await new Promise((done) => setImmediate(done));
  const canvas = viewer.element('iso');

  // Centroide de la cara SUP del widget con az0=35/el0=35, canvas 960x720,
  // dpr=1 (ver dibujarBrujulaCubo en iso.js: cx=820, cy=128, s=34).
  canvas.dispatch('pointerdown', { clientX: 820, clientY: 100, pointerId: 1, preventDefault() {} });
  assert.equal(viewer.element('capa').textContent, 'NIVEL 1');

  // Click en el anillo, lejos de cualquier cara o letra cardinal: vuelve a
  // la isometrica por defecto (equivalente al viejo boton ISO).
  canvas.dispatch('pointerdown', { clientX: 820, clientY: 198, pointerId: 2, preventDefault() {} });
  assert.equal(viewer.element('capa').textContent, 'TODAS LAS CAPAS');
});

test('renders explicit level markers and same-level overlap warnings', async () => {
  const viewer = loadViewer([{
    cell_a: 0, cell_b: 1, level: 0, ratio: 0.25,
    u0: 0.36, v0: 0.36, u1: 0.44, v1: 0.44, z0: 0, height: 0.1,
  }]);
  await new Promise((done) => setImmediate(done));
  const texts = viewer.drawnTexts();

  assert.ok(texts.includes('NIVEL 0'), 'each rendered layer has an external marker');
  assert.ok(texts.includes('NIVEL 1'), 'each rendered layer has an external marker');
  assert.ok(texts.includes('SOLAPE N0  0x1'), 'same-level overlap is visually explicit');
});

test('renders a first-frame observation as confirming without counting it', async () => {
  const viewer = loadViewer([], [{
    cell: -1, level: 1, u: 0.5, v: 0.5,
    side_a: 0.2, side_b: 0.2, z0: 0.1, height: 0.1,
    box_class: 'coin_roll_100', status: 'confirming',
  }]);
  await new Promise((done) => setImmediate(done));

  assert.ok(viewer.drawnTexts().includes('CONFIRMANDO'));
});

test('hides the camera loading indicator when the first video frame arrives', async () => {
  const viewer = loadViewer();
  const loading = viewer.element('camera-loading');
  const video = viewer.element('camera-stream');

  assert.equal(loading.classList.contains('oculto'), false);
  video.dispatch('load');
  assert.equal(loading.classList.contains('oculto'), true);
});
