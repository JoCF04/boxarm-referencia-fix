(function () {
  'use strict';
  var grid = document.getElementById('camera-grid');
  var isPuzzle = grid.classList.contains('puzzle-grid');
  // Grid plano (dragable): tarjetas .camera-card. Puzzle (3 camaras fijas):
  // el iframe vive en ".puzzle-card[data-camera-id]", el conteo en un
  // ".puzzle-info[data-camera-id]" aparte que puede compartir el mismo id
  // (misma camara, dos piezas) -- nunca se mezcla con otra camara.
  var cards = Array.from(grid.querySelectorAll(isPuzzle ? '.puzzle-card' : '.camera-card'));
  // Lo que se oculta al pasar a inspeccion es el CONTENEDOR de la grilla, no
  // la grilla: en el puzzle el wrapper .puzzle-scene lleva el
  // height:calc(100vh - 112px), asi que ocultar solo #camera-grid dejaba una
  // pantalla entera de alto vacio empujando la inspeccion hacia abajo.
  var gridShell = grid.closest('.puzzle-scene') || grid;
  var inspectionView = document.getElementById('inspection-view');
  var inspectionFrame = document.getElementById('inspection-frame');
  var pickerButtons = Array.from(inspectionView.querySelectorAll('.camera-picker button'));
  var mode = 'iso';
  var inspectedCameraId = pickerButtons.length ? pickerButtons[0].dataset.cameraId : null;
  var dragged = null;

  function setInspectedCamera(cameraId) {
    inspectedCameraId = cameraId;
    inspectionFrame.src = '/cam/' + cameraId + '?view=inspection';
    pickerButtons.forEach(function (button) {
      button.classList.toggle('active', button.dataset.cameraId === cameraId);
    });
  }
  pickerButtons.forEach(function (button) {
    button.addEventListener('click', function () { setInspectedCamera(button.dataset.cameraId); });
  });

  function setMode(next) {
    mode = next;
    document.querySelectorAll('.mode').forEach(function (button) {
      button.classList.toggle('active', button.dataset.mode === mode);
    });
    gridShell.hidden = mode === 'inspection';
    inspectionView.hidden = mode !== 'inspection';
    if (mode === 'inspection') {
      setInspectedCamera(inspectedCameraId);
      return;
    }
    cards.forEach(function (card) {
      var iframe = card.querySelector('iframe');
      iframe.src = '/cam/' + card.dataset.cameraId + '?view=' + mode;
    });
  }
  document.querySelectorAll('.mode').forEach(function (button) {
    button.addEventListener('click', function () { setMode(button.dataset.mode); });
  });

  // El reordenamiento por drag solo tiene sentido en el grid plano -- las
  // piezas del puzzle tienen posicion fija por su propia forma (clip-path).
  if (!isPuzzle) {
    cards.forEach(function (card) {
      card.addEventListener('dragstart', function () { dragged = card; card.classList.add('dragging'); });
      card.addEventListener('dragend', function () { dragged = null; card.classList.remove('dragging'); cards.forEach(function (c) { c.classList.remove('drop-target'); }); });
      card.addEventListener('dragover', function (event) { event.preventDefault(); if (dragged !== card) card.classList.add('drop-target'); });
      card.addEventListener('dragleave', function () { card.classList.remove('drop-target'); });
      card.addEventListener('drop', function (event) {
        event.preventDefault(); card.classList.remove('drop-target');
        if (!dragged || dragged === card) return;
        var all = Array.from(grid.children); var from = all.indexOf(dragged); var to = all.indexOf(card);
        if (from < to) grid.insertBefore(dragged, card.nextSibling); else grid.insertBefore(dragged, card);
        cards = Array.from(grid.querySelectorAll('.camera-card'));
      });
    });
  }

  // Paleta RGB de /api/cameras, la MISMA que usa el renderer ISO para las
  // cajas. Global a todas las camaras, se refresca en cada poll.
  var classColors = {};
  function colorDeClase(nombre) {
    var rgb = classColors[nombre];
    return Array.isArray(rgb) && rgb.length === 3 ? 'rgb(' + rgb.join(',') + ')' : '';
  }

  // Todo lo que pinta esta funcion sale UNICAMENTE de payload.scene de esa
  // misma camara -- nunca se mezcla con datos de otra tarjeta.
  function renderConteo(card, scene) {
    var totalEl = card.querySelector('.conteo-total');
    if (!totalEl) { return; }
    var detalleEl = card.querySelector('.conteo-detalle');
    var nivelesEl = card.querySelector('.conteo-niveles');
    var clasesEl = card.querySelector('.conteo-clases');

    var confirmadas = (scene && scene.boxes) || [];
    var provisionales = (scene && scene.provisional_boxes) || [];
    var validando = !!(scene && scene.validating_initial);

    totalEl.textContent = scene ? (validando ? provisionales.length : scene.total) : '-';
    detalleEl.textContent = !scene ? ''
      : validando ? 'SIN CONFIRMAR'
      : 'INICIAL ' + scene.initial + ' · ROBOT ' + scene.placed;

    nivelesEl.innerHTML = '';
    var todas = confirmadas.concat(provisionales);
    var niveles = Array.from(new Set(todas.map(function (box) { return box.level; }))).sort(function (a, b) { return a - b; });
    niveles.forEach(function (level) {
      var enEsteNivel = todas.filter(function (box) { return box.level === level; }).length;
      var span = document.createElement('span');
      span.innerHTML = 'N' + level + ' <b>' + enEsteNivel + '</b>';
      nivelesEl.appendChild(span);
    });

    clasesEl.innerHTML = '';
    var clases = [];
    todas.forEach(function (box) {
      var nombre = box.box_class || 'sin_clase';
      if (clases.indexOf(nombre) === -1) { clases.push(nombre); }
    });
    clases.forEach(function (nombre) {
      var confirmadasClase = confirmadas.filter(function (box) { return (box.box_class || 'sin_clase') === nombre; }).length;
      var observadasClase = provisionales.filter(function (box) { return (box.box_class || 'sin_clase') === nombre; }).length;
      var fila = document.createElement('div');
      fila.className = 'conteo-clase';
      // El bullet lleva el color REAL con el que el renderer ISO pinta esa
      // clase (colors.classes de /iso/scene, ahora tambien en /api/cameras).
      // Antes era una paleta hasheada por nombre: un color inventado que no
      // coincidia con el de las cajas. Si la clase no esta en la paleta cae a
      // currentColor, o sea el mismo color que su nombre.
      var color = colorDeClase(nombre);
      fila.innerHTML = '<i' + (color ? ' style="background:' + color + '"' : '') + '></i>' +
        '<span>' + nombre.toUpperCase() + '</span>' +
        '<b>' + (validando ? observadasClase : confirmadasClase + (observadasClase ? '+' + observadasClase : '')) + '</b>';
      clasesEl.appendChild(fila);
    });
  }

  function renderCard(payload) {
    var status = payload.status || 'no_signal';
    var label = { online:'EN LÍNEA', disabled:'DESACTIVADA', no_signal:'SIN SEÑAL', error:'ERROR' }[status] || status.toUpperCase();
    var overlayText = status === 'disabled' ? 'CÁMARA DESACTIVADA' : status === 'error' ? 'ERROR DE CÁMARA' : 'SIN SEÑAL';
    // En el puzzle puede haber DOS elementos con el mismo data-camera-id (el
    // iframe y, aparte, su ficha de info) -- se actualizan los dos, cada uno
    // con lo que tenga (status/overlay en uno, conteo en el otro).
    document.querySelectorAll('[data-camera-id="' + payload.id + '"]').forEach(function (el) {
      el.classList.remove('online', 'disabled', 'no_signal', 'error');
      el.classList.add(status);
      var statusEl = el.querySelector('.card-status');
      if (statusEl) { statusEl.textContent = label; }
      var overlayEl = el.querySelector('.card-overlay');
      if (overlayEl) { overlayEl.textContent = overlayText; }
      renderConteo(el, payload.scene);
    });
  }
  async function poll() { try { var response = await fetch('/api/cameras', { cache:'no-store' }); if (response.ok) { var data = await response.json(); classColors = data.class_colors || {}; data.cameras.forEach(renderCard); } } catch (_) {} finally { window.setTimeout(poll, 1000); } }
  poll();
}());
