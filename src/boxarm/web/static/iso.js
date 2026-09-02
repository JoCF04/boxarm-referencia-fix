// Renderer ISO local. Python publica geometria compacta; la camara se mueve
// enteramente en el navegador, sin round-trip HTTP ni JPEG por cada pixel.
(function () {
  'use strict';

  var CAM_ID = document.body.dataset.camId;
  var VIEW = document.body.dataset.view;
  var NECESITA_VIDEO = VIEW !== 'iso';
  var NECESITA_ISO = VIEW !== 'camera';
  var AZ0 = parseFloat(document.body.dataset.az0);
  var EL0 = parseFloat(document.body.dataset.el0);
  var EL_MAX = 89;
  var GRADOS_POR_PIXEL = 0.4;
  var GRADOS_POR_RUEDA = 2;

  var lienzo = document.getElementById('lienzo');
  var canvas = document.getElementById('iso');
  var ctx = canvas.getContext('2d');
  var verAz = document.getElementById('az');
  var verEl = document.getElementById('el');
  var selectorCapa = document.getElementById('capa');
  var selectorExplosion = document.getElementById('explosion');
  var selectorIsometrica = document.getElementById('isometrica');
  var video = document.getElementById('camera-stream');
  var cameraLoading = document.getElementById('camera-loading');
  var isoLoading = document.getElementById('iso-loading');
  var videoZoomOut = document.getElementById('video-zoom-out');
  var videoZoomReset = document.getElementById('video-zoom-reset');
  var videoZoomIn = document.getElementById('video-zoom-in');

  var az = AZ0;
  var el = EL0;
  var escena = null;
  var vista = null;
  var colores = null;
  var seq = -1;
  var framePendiente = false;
  var arrastrando = false;
  var ultimoX = 0;
  var ultimoY = 0;
  var nivelVisible = null;
  // Vista explotada (niveles separados en Z) vs apilada a tope. Explotada
  // por defecto: con las capas pegadas, la de abajo queda tapada por la de
  // arriba y no se puede auditar el nivel 0. Apilada muestra la altura real
  // de la carga, que es como se ve el pallet en la vida real.
  var explotado = true;
  var videoMarco = document.querySelector('.video-marco');
  var videoZoom = 1;
  var VIDEO_ZOOM_MIN = 1;
  var VIDEO_ZOOM_MAX = 3;
  var VIDEO_ZOOM_STEP = 0.25;
  var videoPanX = 0;
  var videoPanY = 0;
  var arrastrandoVideo = false;
  var videoUltimoX = 0;
  var videoUltimoY = 0;
  // -- Orbita isometrica automatica -----------------------------------------
  // Animacion basada en tiempo real (performance.now), NO en frames: la
  // velocidad angular es identica a 30, 60 o 144 FPS.  Al pausar se captura
  // el offset para que al reanudar la camara continue exactamente donde
  // quedo, sin salto.
  var orbitando = false;
  var orbitT0 = performance.now() / 1000;
  var orbitOffsetAz = 0;
  var orbitOffsetEl = 0;
  var ORBIT_ISO_AZ = 45.0;
  var ORBIT_ISO_EL = 35.26438968;   // atan(1/sqrt(2)) en grados
  var ORBIT_OMEGA = 8.0;            // grados/segundo  (360/45 = vuelta en 45 s)
  var ORBIT_EL_AMP = 2.0;           // oscilacion vertical +/- grados
  var ORBIT_EL_PERIOD = 20.0;       // periodo de la oscilacion vertical en segundos
  var selectorOrbitar = document.getElementById('orbitar');

  function ocultarCargaVideo() {
    cameraLoading.classList.add('oculto');
  }
  video.addEventListener('load', ocultarCargaVideo);
  // Vista "iso" no muestra el video (queda display:none): no tiene sentido
  // pedirle el stream MJPEG al servidor solo para tenerlo oculto.
  if (NECESITA_VIDEO) {
    video.src = video.dataset.src;
  } else {
    cameraLoading.classList.add('oculto');
  }
  // Evita que una imagen ya cacheada deje visible el overlay antes de que
  // este script registre el evento load.
  if (video.complete && video.naturalWidth > 0) { ocultarCargaVideo(); }

  // El pan solo tiene sentido mientras hay zoom: al pasar de zoom 1 a mas se
  // corre la imagen contra los bordes del marco para no dejar ver fondo vacio.
  function limitarPanVideo() {
    var maxOffset = (videoZoom - 1) * 50; // % del tamano del marco, cada lado
    videoPanX = Math.max(-maxOffset, Math.min(maxOffset, videoPanX));
    videoPanY = Math.max(-maxOffset, Math.min(maxOffset, videoPanY));
  }

  function actualizarZoomVideo() {
    limitarPanVideo();
    video.style.transform = 'translate(' + videoPanX + '%, ' + videoPanY + '%) scale(' + videoZoom + ')';
    videoMarco.classList.toggle('zoom-activo', videoZoom > VIDEO_ZOOM_MIN);
    videoZoomOut.disabled = videoZoom <= VIDEO_ZOOM_MIN;
    videoZoomIn.disabled = videoZoom >= VIDEO_ZOOM_MAX;
  }

  function cambiarZoomVideo(delta) {
    videoZoom = Math.max(VIDEO_ZOOM_MIN,
      Math.min(VIDEO_ZOOM_MAX, Math.round((videoZoom + delta) * 100) / 100));
    actualizarZoomVideo();
  }

  videoZoomOut.addEventListener('click', function () { cambiarZoomVideo(-VIDEO_ZOOM_STEP); });
  videoZoomIn.addEventListener('click', function () { cambiarZoomVideo(VIDEO_ZOOM_STEP); });
  videoZoomReset.addEventListener('click', function () {
    videoZoom = VIDEO_ZOOM_MIN;
    videoPanX = 0;
    videoPanY = 0;
    actualizarZoomVideo();
  });

  // Rueda del mouse sobre el video: zoom in/out, igual que los botones +/-.
  videoMarco.addEventListener('wheel', function (event) {
    event.preventDefault();
    cambiarZoomVideo(event.deltaY < 0 ? VIDEO_ZOOM_STEP : -VIDEO_ZOOM_STEP);
  }, { passive: false });

  // Arrastrar con el mouse mueve la imagen (pan) solo si hay zoom aplicado.
  videoMarco.addEventListener('mousedown', function (event) {
    if (videoZoom <= VIDEO_ZOOM_MIN) return;
    arrastrandoVideo = true;
    videoUltimoX = event.clientX;
    videoUltimoY = event.clientY;
    videoMarco.classList.add('arrastrando');
    event.preventDefault();
  });
  window.addEventListener('mousemove', function (event) {
    if (!arrastrandoVideo) return;
    var rect = videoMarco.getBoundingClientRect();
    videoPanX += (event.clientX - videoUltimoX) / rect.width * 100 / videoZoom;
    videoPanY += (event.clientY - videoUltimoY) / rect.height * 100 / videoZoom;
    videoUltimoX = event.clientX;
    videoUltimoY = event.clientY;
    actualizarZoomVideo();
  });
  window.addEventListener('mouseup', function () {
    arrastrandoVideo = false;
    videoMarco.classList.remove('arrastrando');
  });

  var CARAS = [
    { normal: [1, 0, 0], shade: 0.55, ids: [4, 6, 7, 5] },
    { normal: [-1, 0, 0], shade: 0.55, ids: [0, 1, 3, 2] },
    { normal: [0, 1, 0], shade: 0.8, ids: [2, 3, 7, 6] },
    { normal: [0, -1, 0], shade: 0.8, ids: [0, 4, 5, 1] },
    { normal: [0, 0, 1], shade: 1, ids: [1, 5, 7, 3] },
    { normal: [0, 0, -1], shade: 0.4, ids: [0, 2, 6, 4] },
  ];

  function limitarElevacion(valor) {
    return Math.max(-EL_MAX, Math.min(EL_MAX, valor));
  }

  function mostrarAngulos() {
    verAz.textContent = az.toFixed(0);
    verEl.textContent = el.toFixed(0);
  }

  function cajasConfirmadas() {
    return escena && Array.isArray(escena.boxes) ? escena.boxes : [];
  }

  function cajasProvisionales() {
    return escena && Array.isArray(escena.provisional_boxes) ? escena.provisional_boxes : [];
  }

  function todasLasCajas() {
    return cajasConfirmadas().concat(cajasProvisionales());
  }

  function nivelesOcupados() {
    if (!escena) { return []; }
    return Array.from(new Set(todasLasCajas().map(function (box) { return box.level; })))
      .sort(function (a, b) { return a - b; });
  }

  function numeroNivelesEscena() {
    var levels = nivelesOcupados();
    var porGeometria = levels.length ? levels[levels.length - 1] + 1 : 0;
    return Math.max(escena?.levels || 0, porGeometria);
  }

  function enfocarNivel(level) {
    nivelVisible = level;
    selectorCapa.textContent = level === null ? 'TODAS LAS CAPAS' : 'NIVEL ' + level;
    selectorCapa.classList[level === null ? 'remove' : 'add']('activa');
    pedirRender();
  }

  function color(rgb, alpha) {
    return 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + alpha + ')';
  }

  function proyectar(x, y, z, camera) {
    var xp = x * camera.ct - y * camera.st;
    var yp = x * camera.st + y * camera.ct;
    return {
      u: xp,
      v: yp * camera.sp - z * camera.cp,
      depth: yp * camera.cp + z * camera.sp,
    };
  }

  function poligono(points, fill, stroke, width) {
    if (!points.length) { return; }
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (var i = 1; i < points.length; i += 1) { ctx.lineTo(points[i][0], points[i][1]); }
    ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width || 1; ctx.stroke(); }
  }

  function esquinas(cx, cy, z0, a, b, c) {
    var x0 = cx - a / 2;
    var y0 = cy - b / 2;
    return [
      [x0, y0, z0], [x0, y0, z0 + c],
      [x0, y0 + b, z0], [x0, y0 + b, z0 + c],
      [x0 + a, y0, z0], [x0 + a, y0, z0 + c],
      [x0 + a, y0 + b, z0], [x0 + a, y0 + b, z0 + c],
    ];
  }

  function dimensionar() {
    // Devuelve el DPR: es la unidad en la que se dibuja TODO el HUD.
    // El canvas mide en pixeles de dispositivo (clientWidth * dpr), asi que
    // dimensionar texto como una fraccion de canvas.width lo hacia crecer
    // con la resolucion de la pantalla -- en un monitor ancho con dpr 2 el
    // titulo salia a ~85 px y se comia el resto del HUD. Con `u = dpr`, un
    // "14 * u" son 14 pixeles CSS reales en cualquier pantalla.
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    var height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return dpr;
  }

  // -- Paleta del visor ----------------------------------------------------
  // Look "oscuro tecnico": azul cosmico profundo, acentos cian de baja
  // saturacion y volumenes casi neutros. La saturacion plena de
  // drawing.yaml sirve para el overlay 2D sobre video (tiene que gritar
  // sobre carton), pero en una escena 3D grande convierte la vista en un
  // La clase conserva su color de referencia; el nivel se expresa con altura
  // y etiquetas, evitando mezclar semanticas de nivel y producto.
  var TINTE_CLASE = 0.72;             // cuanto sobrevive del color de clase en las caras
  var NEUTRO_CARA = [96, 112, 128];   // gris pizarra al que se mezcla
  var ACENTO = [96, 178, 214];        // cian apagado del grid y del HUD
  var CONFLICTO = [255, 82, 70];      // rojo coral: solape real entre celdas
  var CONFIRMANDO = [245, 177, 56];   // ambar: observado, aun no contabilizado
  var INICIALIZANDO = [177, 116, 255]; // violeta: reconstruccion inicial en curso

  function mezclar(a, b, t) {
    return [
      Math.round(a[0] * (1 - t) + b[0] * t),
      Math.round(a[1] * (1 - t) + b[1] * t),
      Math.round(a[2] * (1 - t) + b[2] * t),
    ];
  }

  function escalar(rgb, factor) {
    return rgb.map(function (channel) {
      return Math.max(0, Math.min(255, Math.round(channel * factor)));
    });
  }

  function tonoClase(rgb) {
    return mezclar(NEUTRO_CARA, rgb, TINTE_CLASE);
  }

  // Aristas por nivel: el relleno sigue codificando la CLASE, pero con una
  // sola clase en la paleta toda la torre se leia igual. La arista es la
  // unica capa que cambia. Dos tonos intercalados (par/impar) en vez de un
  // hue por nivel: la etiqueta ya dice QUE nivel es, la arista solo tiene
  // que decir DONDE termina uno y empieza el siguiente.
  var ARISTAS_NIVEL = [
    [110, 200, 255],  // niveles pares: cian
    [255, 196, 92],   // niveles impares: ambar
  ];

  function rgbDeNivel(level) {
    return ARISTAS_NIVEL[((level % 2) + 2) % 2];
  }

  function colorDeNivel(level, alpha) {
    return color(rgbDeNivel(level), alpha);
  }

  function rgbDeClase(boxClass) {
    return colores?.classes?.[boxClass] || [160, 180, 200];
  }

  function fondoCosmico(width, height) {
    // Nunca negro plano: degradado azul profundo + halo suave arriba y
    // vineteado en los bordes. Es lo que separa "app de producto" de
    // "figura sobre fondo negro".
    var vertical = ctx.createLinearGradient(0, 0, 0, height);
    vertical.addColorStop(0, '#080d15');
    vertical.addColorStop(0.55, '#0a1522');
    vertical.addColorStop(1, '#060a11');
    ctx.fillStyle = vertical;
    ctx.fillRect(0, 0, width, height);

    var halo = ctx.createRadialGradient(width * 0.5, height * 0.18, 0,
                                        width * 0.5, height * 0.18, Math.max(width, height) * 0.75);
    halo.addColorStop(0, 'rgba(64,150,196,0.16)');
    halo.addColorStop(0.45, 'rgba(38,92,132,0.06)');
    halo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, width, height);

    var vineta = ctx.createRadialGradient(width * 0.5, height * 0.5, Math.min(width, height) * 0.28,
                                          width * 0.5, height * 0.5, Math.max(width, height) * 0.72);
    vineta.addColorStop(0, 'rgba(0,0,0,0)');
    vineta.addColorStop(1, 'rgba(2,4,8,0.55)');
    ctx.fillStyle = vineta;
    ctx.fillRect(0, 0, width, height);
  }

  function linea(p0, p1, stroke, ancho) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = ancho;
    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]);
    ctx.lineTo(p1[0], p1[1]);
    ctx.stroke();
  }

  function resplandorSuelo(screen, x0, x1, y0, y1, width) {
    // Halo bajo la paleta: la carga se ve apoyada sobre algo iluminado y no
    // flotando en el vacio. Es radial en PANTALLA, no proyectado: representa
    // luz, no geometria.
    var centro = screen((x0 + x1) / 2, (y0 + y1) / 2, 0);
    var radio = width * 0.34;
    var halo = ctx.createRadialGradient(centro[0], centro[1], 0, centro[0], centro[1], radio);
    halo.addColorStop(0, 'rgba(78,168,214,0.20)');
    halo.addColorStop(0.5, 'rgba(58,128,176,0.07)');
    halo.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = halo;
    ctx.fillRect(centro[0] - radio, centro[1] - radio, radio * 2, radio * 2);
  }

  function reticulaSuelo(screen, x0, x1, y0, y1, u) {
    // Grid tecnico que se extiende MAS ALLA de la paleta y se apaga con la
    // distancia: ubica la carga en un espacio en vez de recortarla contra el
    // fondo. Sin ese desvanecido el borde del grid se lee como un corte duro.
    var extra = 1.6;   // cuanto sobresale el grid del piso, por lado
    var divisiones = 26;
    var sx = x1 - x0;
    var sy = y1 - y0;
    var gx0 = x0 - sx * extra;
    var gx1 = x1 + sx * extra;
    var gy0 = y0 - sy * extra;
    var gy1 = y1 + sy * extra;
    var grosor = Math.max(1, 0.5 * u);
    for (var i = 0; i <= divisiones; i += 1) {
      var t = i / divisiones;
      var fade = 1 - Math.min(1, Math.abs(t - 0.5) * 2);
      var alpha = 0.045 + fade * 0.10;
      var gx = gx0 + (gx1 - gx0) * t;
      var gy = gy0 + (gy1 - gy0) * t;
      linea(screen(gx, gy0, 0), screen(gx, gy1, 0), color(ACENTO, alpha), grosor);
      linea(screen(gx0, gy, 0), screen(gx1, gy, 0), color(ACENTO, alpha), grosor);
    }
  }

  function dibujarMarcosDeNivel(screen, x0, x1, y0, y1, sz, u, gap, levelH, niveles, deckH) {
    // La separacion vertical por si sola no basta: en una proyeccion oblicua
    // las huellas de dos niveles pueden coincidir en pantalla. Estos marcos
    // y postes hacen explicita la cota de cada capa sin inventar geometria de
    // la deteccion: son una guia visual del nivel, no cajas adicionales.
    var margenX = (x1 - x0) * 0.035;
    var margenY = (y1 - y0) * 0.035;
    for (var level = 0; level < niveles; level += 1) {
      if (nivelVisible !== null && level !== nivelVisible) { continue; }
      var cajasNivel = todasLasCajas().filter(function (box) { return box.level === level; });
      var zObservado = cajasNivel.length
        ? Math.min.apply(null, cajasNivel.map(function (box) { return box.z0 || 0; }))
        : null;
      var zBaseNormalizado = escena.level_tops?.[level];
      if (!Number.isFinite(zBaseNormalizado)) {
        zBaseNormalizado = zObservado === null ? level * levelH / Math.max(sz, 1e-9) : zObservado;
      }
      var z0 = deckH + zBaseNormalizado * sz + gap * level;
      // No usar el promedio cuando los niveles tienen footprints distintos:
      // la tapa del marco debe coincidir con la altura visual real de ese
      // nivel, no con la altura media de toda la torre.
      var zNextReal = escena.level_tops?.[level + 1];
      var alturaObservada = cajasNivel.length
        ? Math.max.apply(null, cajasNivel.map(function (box) { return box.height || 0; })) * sz
        : 0;
      var alturaNivel = Number.isFinite(zNextReal)
        ? Math.max((zNextReal - zBaseNormalizado) * sz, alturaObservada)
        : Math.max(levelH, alturaObservada);
      var z1 = z0 + alturaNivel;
      var base = [
        screen(x0 - margenX, y0 - margenY, z0),
        screen(x1 + margenX, y0 - margenY, z0),
        screen(x1 + margenX, y1 + margenY, z0),
        screen(x0 - margenX, y1 + margenY, z0),
      ];
      var techo = [
        screen(x0 - margenX, y0 - margenY, z1),
        screen(x1 + margenX, y0 - margenY, z1),
        screen(x1 + margenX, y1 + margenY, z1),
        screen(x0 - margenX, y1 + margenY, z1),
      ];
      var tono = color(ACENTO, 0.50);
      poligono(base, null, tono, Math.max(1, 1.15 * u));
      poligono(techo, null, color(ACENTO, 0.28), Math.max(1, 0.85 * u));
      for (var i = 0; i < base.length; i += 1) {
        linea(base[i], techo[i], color(ACENTO, 0.34), Math.max(1, 0.85 * u));
      }

      // Etiqueta fuera de la carga, unida al marco: "NIVEL 0/1" no depende
      // de que el centroide de una caja quede visible entre otras cajas.
      var centroZ = z0 + (z1 - z0) * 0.52;
      var centroY = y0 + (y1 - y0) * 0.52;
      var ancla = screen(x0 - margenX * 2.6, centroY, centroZ);
      var destino = screen(x0 - margenX, centroY, centroZ);
      linea(ancla, destino, color(ACENTO, 0.65), Math.max(1, 0.8 * u));
      var texto = 'NIVEL ' + level;
      ctx.font = '700 ' + (10 * u) + 'px ui-monospace, monospace';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      var ancho = ctx.measureText(texto).width;
      ctx.fillStyle = 'rgba(6,12,20,0.88)';
      ctx.fillRect(ancla[0], ancla[1] - 9 * u, ancho + 12 * u, 18 * u);
      ctx.fillStyle = color(ACENTO, 0.98);
      ctx.fillText(texto, ancla[0] + 6 * u, ancla[1]);
    }
  }

  function caraGradiente(points, rgb, shade) {
    // Degradado por cara en vez de color plano: es lo que hace que la caja se
    // lea como material y no como un poligono relleno.
    var xs = points.map(function (p) { return p[0]; });
    var ys = points.map(function (p) { return p[1]; });
    var grad = ctx.createLinearGradient(Math.min.apply(null, xs), Math.min.apply(null, ys),
                                        Math.max.apply(null, xs), Math.max.apply(null, ys));
    // Opaco: con alpha < 1 se veian las aristas de las cajas de atras a
    // traves de las de adelante y la pila se leia como vidrio apilado. La
    // translucidez queda reservada a provisionales e inicializando, donde
    // "aun no es firme" es justamente lo que hay que comunicar.
    var luz = 0.55 + shade * 0.55;
    grad.addColorStop(0, color(escalar(rgb, luz * 1.18), 1));
    grad.addColorStop(1, color(escalar(rgb, luz * 0.72), 1));
    return grad;
  }

  function brilloCara(points, shade) {
    // Punto de luz especular sobre la cara superior. Muy tenue: sugiere
    // barniz, no un foco.
    if (shade < 0.95) { return; }
    var cx = points.reduce(function (s, p) { return s + p[0]; }, 0) / points.length;
    var cy = points.reduce(function (s, p) { return s + p[1]; }, 0) / points.length;
    var xs = points.map(function (p) { return p[0]; });
    var radio = (Math.max.apply(null, xs) - Math.min.apply(null, xs)) * 0.42;
    if (radio <= 0) { return; }
    var brillo = ctx.createRadialGradient(cx, cy, 0, cx, cy, radio);
    brillo.addColorStop(0, 'rgba(190,232,255,0.22)');
    brillo.addColorStop(1, 'rgba(190,232,255,0)');
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    for (var i = 1; i < points.length; i += 1) { ctx.lineTo(points[i][0], points[i][1]); }
    ctx.closePath();
    ctx.clip();
    ctx.fillStyle = brillo;
    ctx.fill();
    ctx.restore();
  }

  function cuboNavEsquinas(s) {
    // Mismo orden que esquinas(): index = 4*xi + 2*yi + zi, con xi/yi/zi en
    // {0,1} mapeando a {-s,+s}. Compartir el orden deja reusar CARAS tal cual.
    return [
      [-s, -s, -s], [-s, -s, s], [-s, s, -s], [-s, s, s],
      [s, -s, -s], [s, -s, s], [s, s, -s], [s, s, s],
    ];
  }

  // Zonas de click del widget, en pixeles de canvas (device px). Se llenan en
  // cada dibujarBrujulaCubo() y las lee el pointerdown del canvas -- misma
  // logica de "saltar a una vista" que los botones .vista de la barra, solo
  // que activada desde el propio cubo/brujula en vez de un boton HTML.
  var brujulaHit = null;

  function dibujarBrujulaCubo(width, height, u, camera, camWorld) {
    // Widget de orientacion estilo "ViewCube": un cubo 3D pequeno que gira
    // con la camara (mismo az/el que la escena) mas un anillo de brujula
    // debajo. Reemplaza al hueco que quedaba en la esquina -- antes no habia
    // referencia visual de hacia donde mira uno, solo los numeros de AZIMUT
    // / ELEVACION en el HUD de texto.
    // El widget se dibujaba a tamano fijo en pixeles CSS, asi que en una
    // tarjeta chica del dashboard (canvas angosto) se veia gigante
    // comparado con el mismo cubo en el panel grande -- con un techo de 700
    // ambos casos tocaban el mismo tope y terminaban IGUAL de grandes pese
    // a que el panel grande mide varias veces mas. Referencia mas alta
    // (1600) para que el techo solo lo alcance un panel realmente grande.
    // max(ancho, alto) y no solo ancho: una tarjeta angosta pero alta no
    // debe quedar subestimada solo porque mide poco de lado a lado.
    var dimensionMayor = Math.max(width, height) / u;
    var escala = Math.max(0.25, Math.min(0.7, dimensionMayor / 1600)) * 1.8;
    var uc = u * escala;
    var cx = width - 115 * uc;
    var cy = 105 * uc;
    var s = 28 * uc;
    var radioAnillo = 62 * uc;
    brujulaHit = { cx: cx, cy: cy, radioAnillo: radioAnillo, radioCubo: s * 1.6, cardinales: [], caras: [] };

    function proyCubo(p) {
      var xp = p[0] * camera.ct - p[1] * camera.st;
      var yp = p[0] * camera.st + p[1] * camera.ct;
      var v = yp * camera.sp - p[2] * camera.cp;
      return [cx + xp, cy + v];
    }

    var esq = cuboNavEsquinas(s).map(proyCubo);

    // Anillo de brujula: plano, gira solo con el azimut (sin inclinacion),
    // asi conserva forma circular como una brujula real.
    ctx.beginPath();
    ctx.arc(cx, cy, radioAnillo, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(124,195,228,0.28)';
    ctx.lineWidth = Math.max(1, 1 * uc);
    ctx.stroke();

    var cardinales = [['N', 0], ['E', 90], ['S', 180], ['O', 270]];
    ctx.font = '700 ' + (12 * uc) + 'px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    cardinales.forEach(function (par) {
      var rad = par[1] * Math.PI / 180;
      var wx = Math.sin(rad);
      var wy = Math.cos(rad);
      var xp = wx * camera.ct - wy * camera.st;
      var yp = wx * camera.st + wy * camera.ct;
      var px = cx + xp * radioAnillo;
      var py = cy + yp * radioAnillo;
      var esNorte = par[0] === 'N';
      ctx.fillStyle = esNorte ? 'rgba(190,232,255,0.92)' : 'rgba(150,169,184,0.75)';
      ctx.fillText(par[0], px, py);
      brujulaHit.cardinales.push({ x: px, y: py, az: par[1] });
    });

    // Caras del cubo: mismas normales/shade que las cajas, ocultando las que
    // miran en contra de la camara. La etiqueta de cada cara sale de su
    // normal, reusando el mismo mapeo direccion->cardinal que el anillo
    // (az=0 -> +Y es Norte, az=90 -> +X es Este).
    var ETIQUETA_CARA = ['E', 'O', 'N', 'S', 'SUP', ''];
    var caras = CARAS.map(function (face, index) {
      var dot = face.normal[0] * camWorld[0] + face.normal[1] * camWorld[1] + face.normal[2] * camWorld[2];
      return { face: face, dot: dot, etiqueta: ETIQUETA_CARA[index] };
    }).filter(function (item) { return item.dot > 1e-6; });

    caras.forEach(function (item) {
      var face = item.face;
      var puntos = face.ids.map(function (id) { return esq[id]; });
      var luz = 0.6 + face.shade * 0.5;
      var base = escalar([148, 176, 196], luz);
      poligono(puntos, color(base, 0.94), 'rgba(6,12,20,0.55)', Math.max(1, 0.8 * uc));
      if (item.etiqueta) { brujulaHit.caras.push({ etiqueta: item.etiqueta, puntos: puntos }); }
      if (!item.etiqueta) { return; }
      var cx2 = puntos.reduce(function (sum, p) { return sum + p[0]; }, 0) / puntos.length;
      var cy2 = puntos.reduce(function (sum, p) { return sum + p[1]; }, 0) / puntos.length;
      ctx.font = '700 ' + (11 * uc) + 'px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(18,28,38,0.72)';
      ctx.fillText(item.etiqueta, cx2, cy2);
    });
  }

  function crearSolido(box, sx, sy, sz, gap, camera, baseZ) {
    var cx = box.u * sx;
    var cy = box.v * sy;
    var z0 = (baseZ || 0) + box.z0 * sz + gap * box.level;
    var h = box.height * sz;
    var corners = esquinas(cx, cy, z0, box.side_a * sx, box.side_b * sy, h);
    // El orden de dibujado (pintor) NO puede decidirse con un solo punto
    // (el centro): dos cajas de niveles distintos que se acercan en
    // pantalla pueden "ganar" al reves si se compara solo su centro -- una
    // esquina de la caja de abajo puede estar mas cerca de camara que el
    // centro de la de arriba, y viceversa. Se usa la esquina MAS CERCANA
    // a camara (max depth de las 8) como profundidad del solido: si
    // cualquier parte de una caja esta mas cerca, esa caja se dibuja
    // despues (encima), que es lo correcto para que no se vea "atravesada".
    var depth = Math.max.apply(null, corners.map(function (p) {
      return proyectar(p[0], p[1], p[2], camera).depth;
    }));
    return {
      box: box,
      depth: depth,
      corners: corners,
    };
  }

  function dibujarVolumenAuxiliar(cx, cy, z0, a, b, h, rgb, screen, camWorld, u) {
    var projected = esquinas(cx, cy, z0, a, b, h).map(function (p) {
      return screen(p[0], p[1], p[2]);
    });
    CARAS.forEach(function (face) {
      var dot = face.normal[0] * camWorld[0] + face.normal[1] * camWorld[1] + face.normal[2] * camWorld[2];
      if (dot <= 1e-6) { return; }
      var points = face.ids.map(function (id) { return projected[id]; });
      var luz = 0.55 + face.shade * 0.55;
      // La paleta es estructura, no una guia translucida: debe tapar el grid
      // y leerse como un bloque solido desde arriba y de costado.
      poligono(points, color(escalar(rgb, luz), 1),
               'rgba(18,24,25,0.82)', Math.max(1, 0.85 * u));
    });
  }

  function dibujarProvisional(solid, screen, camWorld, u) {
    var projected = solid.corners.map(function (p) { return screen(p[0], p[1], p[2]); });
    var inicial = solid.box.status === 'initializing';
    var base = rgbDeClase(solid.box.box_class);
    // Etiquetas cortas: el chip se dibuja sobre la caja fantasma y con
    // varias en pantalla el texto largo tapaba las cajas vecinas.
    // SCAN = analisis inicial de la paleta; CONF = confirmando una nueva.
    // Nivel incluido (mismo formato "N0"/"N1" que las cajas confirmadas,
    // ver mas abajo box.cell + 'N' + box.level): sin esto dos fantasmas
    // apilados en niveles distintos se veian con el mismo chip "CONF" y no
    // se distinguia cual era cual.
    var etiqueta = (inicial ? 'SCAN' : 'CONF') + ' N' + solid.box.level;

    // El trazo discontinuo y la transparencia expresan incertidumbre. El
    // tono sigue siendo el de la clase, incluso durante la confirmacion.
    ctx.save();
    ctx.setLineDash([7 * u, 4 * u]);
    ctx.shadowColor = color(base, 0.55);
    ctx.shadowBlur = 9 * u;
    CARAS.forEach(function (face) {
      var dot = face.normal[0] * camWorld[0] + face.normal[1] * camWorld[1] + face.normal[2] * camWorld[2];
      if (dot <= 1e-6) { return; }
      var points = face.ids.map(function (id) { return projected[id]; });
      var luz = 0.48 + face.shade * 0.42;
      poligono(points, color(escalar(base, luz), inicial ? 0.18 : 0.22),
               color(base, inicial ? 0.92 : 0.98), Math.max(1.5, 1.45 * u));
    });
    ctx.restore();

    // Una segunda tapa solida evita que el fantasma desaparezca visualmente
    // entre la reticula y los marcos discontinuos.
    var top = CARAS[4].ids.map(function (id) { return projected[id]; });
    poligono(top, color(base, inicial ? 0.12 : 0.16), color(base, 0.98), Math.max(2, 1.8 * u));
    linea(top[0], top[2], color(base, 0.72), Math.max(1, 0.9 * u));
    linea(top[1], top[3], color(base, 0.72), Math.max(1, 0.9 * u));

    var labelX = projected.reduce(function (sum, p) { return sum + p[0]; }, 0) / projected.length;
    var labelY = projected.reduce(function (sum, p) { return sum + p[1]; }, 0) / projected.length;
    ctx.font = '800 ' + (8.5 * u) + 'px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    var ancho = ctx.measureText(etiqueta).width;
    ctx.fillStyle = 'rgba(8,12,18,0.90)';
    ctx.fillRect(labelX - ancho / 2 - 4 * u, labelY - 7 * u, ancho + 8 * u, 14 * u);
    ctx.fillStyle = color(base, 1);
    ctx.fillText(etiqueta, labelX, labelY);
  }

  function render() {
    framePendiente = false;
    var u = dimensionar();   // pixel CSS -> pixel de canvas
    var width = canvas.width;
    var height = canvas.height;
    fondoCosmico(width, height);

    // -- orbita isometrica: calcular angulos desde tiempo real ---------------
    if (orbitando) {
      var t = performance.now() / 1000 - orbitT0;
      az = (ORBIT_ISO_AZ + ORBIT_OMEGA * t + orbitOffsetAz) % 360;
      if (az < 0) { az += 360; }
      el = ORBIT_ISO_EL + ORBIT_EL_AMP * Math.sin((2 * Math.PI * t) / ORBIT_EL_PERIOD) + orbitOffsetEl;
      el = limitarElevacion(el);
      mostrarAngulos();
      pedirRender();
    }

    var theta = az * Math.PI / 180;
    var phi = el * Math.PI / 180;
    var camera = { ct: Math.cos(theta), st: Math.sin(theta), cp: Math.cos(phi), sp: Math.sin(phi) };
    var camWorld = [camera.cp * camera.st, camera.cp * camera.ct, camera.sp];
    dibujarBrujulaCubo(width, height, u, camera, camWorld);
    if (!escena || !vista) { return; }

    // Las cajas viven SIEMPRE en el cuadrado unidad [0,1]^2 -- ahi nace
    // box.u/box.v (homografia real, ver scene.py) y no se multiplican por
    // nada del pallet. sx/sy es el mundo de la caja, no el tamano del
    // pallet: si el pallet se reescalara aca, las cajas se reescalarian
    // junto con el (ese era el bug: una sola variable hacia las dos cosas).
    // El pallet es su PROPIO rectangulo (vista.pallet_x0..y1), manejado
    // aparte mas abajo (fx0/fx1/fy0/fy1).
    var sx = 1;
    var sy = 1;
    var sz = Math.min(sx, sy);
    var cajasEscena = todasLasCajas();

    // El piso de la paleta es SIEMPRE su propio rectangulo medido
    // (vista.pallet_x0..y1): configs/roi_cam_<id>.json campo "pallet_roi", proyectado
    // en streaming.py con la misma homografia que posiciona cada caja. Es
    // geometria fisica de la tarima, no una funcion de la carga -- una
    // paleta no cambia de tamano porque le pongan o saquen cajas.
    //
    // NO se recorta al bounding box de las cajas: eso lo expandia casi al
    // cuadrado unidad completo y hacia que el piso midiera lo mismo que el
    // ROI de deteccion, tapando el pallet_roi real que se acababa de medir.
    var fx0 = typeof vista.pallet_x0 === 'number' ? vista.pallet_x0 : 0;
    var fx1 = typeof vista.pallet_x1 === 'number' ? vista.pallet_x1 : sx;
    var fy0 = typeof vista.pallet_y0 === 'number' ? vista.pallet_y0 : 0;
    var fy1 = typeof vista.pallet_y1 === 'number' ? vista.pallet_y1 : sy;
    var floorW = fx1 - fx0;
    var floorH = fy1 - fy0;
    var floorSize = Math.min(floorW, floorH);
    // El grosor del deck/soportes es proporcion del tamano REAL del pallet
    // (floorSize), no del dominio [0,1] entero de las cajas -- si no, un
    // pallet_roi mas chico que el ROI (lo normal) deja el deck viendose
    // igual de grueso que antes, ahora desproporcionado contra un piso mas
    // angosto. deck_thickness_m/support_height_m (drawing.yaml) son la
    // fraccion configurable; el nombre "_m" es historico, ya no son metros.
    var deckRatio = typeof vista.pallet_deck_thickness === 'number' ? vista.pallet_deck_thickness : 0.02;
    var supportRatio = typeof vista.pallet_support_height === 'number' ? vista.pallet_support_height : 0.03;
    var deckH = floorSize * deckRatio;
    var supportH = floorSize * supportRatio;

    var nivelesEscena = numeroNivelesEscena();
    var alturaObservada = cajasEscena.length
      ? Math.max.apply(null, cajasEscena.map(function (box) {
          return (box.z0 || 0) + (box.height || 0);
        }))
      : 0;
    var totalH = deckH + Math.max(escena.total_height || 0, alturaObservada) * sz;
    var levelH = nivelesEscena ? totalH / nivelesEscena : 0;
    var gap = explotado ? vista.level_gap_ratio * levelH : 0;
    var totalDraw = Math.max(totalH + gap * Math.max(0, nivelesEscena - 1),
      cajasEscena.length ? Math.max.apply(null, cajasEscena.map(function (box) {
        return ((box.z0 || 0) + (box.height || 0)) * sz + gap * box.level;
      })) : 0);

    var boundsWorld = [];
    [fx0, fx1].forEach(function (x) {
      [fy0, fy1].forEach(function (y) {
        [-supportH, totalDraw].forEach(function (z) { boundsWorld.push([x, y, z]); });
      });
    });
    cajasEscena.forEach(function (box) {
      var solid = crearSolido(box, sx, sy, sz, gap, camera, deckH);
      boundsWorld = boundsWorld.concat(solid.corners);
    });
    var bounds = boundsWorld.map(function (p) { return proyectar(p[0], p[1], p[2], camera); });
    var minU = Math.min.apply(null, bounds.map(function (p) { return p.u; }));
    var maxUWorld = Math.max.apply(null, bounds.map(function (p) { return p.u; }));
    var minV = Math.min.apply(null, bounds.map(function (p) { return p.v; }));
    var maxVWorld = Math.max.apply(null, bounds.map(function (p) { return p.v; }));
    var center = { u: (minU + maxUWorld) / 2, v: (minV + maxVWorld) / 2 };
    var maxU = (maxUWorld - minU) / 2;
    var maxV = (maxVWorld - minV) / 2;
    var scale = Math.min(
      width * vista.fill_margin / Math.max(2 * maxU, 1e-9),
      height * vista.fill_margin / Math.max(2 * maxV, 1e-9),
    );
    function screen(x, y, z) {
      var p = proyectar(x, y, z, camera);
      return [width / 2 + (p.u - center.u) * scale,
              height / 2 + (p.v - center.v) * scale];
    }

    resplandorSuelo(screen, fx0, fx1, fy0, fy1, width);
    reticulaSuelo(screen, fx0, fx1, fy0, fy1, u);

    if (vista.pallet_visible !== false) {
      // Paleta 3D: un deck macizo y seis bloques de soporte, todos de madera.
      // El template queda encima del deck; esto reemplaza la antigua tabla plana.
      var soporteFW = floorW * 0.070;
      var soporteY = [fy0 + floorH * 0.18, fy0 + floorH * 0.50, fy0 + floorH * 0.82];
      soporteY.forEach(function (y) {
        dibujarVolumenAuxiliar(fx0 + soporteFW / 2, y, -supportH,
                               soporteFW, floorH * 0.18, supportH,
                             [118, 82, 50], screen, camWorld, u);
        dibujarVolumenAuxiliar(fx1 - soporteFW / 2, y, -supportH,
                               soporteFW, floorH * 0.18, supportH,
                             [118, 82, 50], screen, camWorld, u);
      });
    // Cuatro tablones superiores con juntas estrechas: sigue siendo un
    // volumen macizo, pero la silueta comunica "paleta" y no "caja enorme".
    var junta = floorH * 0.012;
    var tablonY = (floorH - 3 * junta) / 4;
    for (var tablon = 0; tablon < 4; tablon += 1) {
      var tablonYc = fy0 + tablonY / 2 + tablon * (tablonY + junta);
      dibujarVolumenAuxiliar((fx0 + fx1) / 2, tablonYc, 0, floorW, tablonY, deckH,
                             [118, 82, 50], screen, camWorld, u);
    }
      var pisoPts = [[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]].map(function (p) { return screen(p[0], p[1], 0); });
      poligono(pisoPts, null, 'rgba(219,174,111,0.92)', Math.max(1, 1.2 * u));
    }

    dibujarMarcosDeNivel(screen, fx0, fx1, fy0, fy1, sz, u, gap, levelH, nivelesEscena, deckH);

    var cajasVisibles = cajasConfirmadas().filter(function (box) {
      return nivelVisible === null || box.level === nivelVisible;
    });
    var provisionalesVisibles = cajasProvisionales().filter(function (box) {
      return nivelVisible === null || box.level === nivelVisible;
    });
    var solids = cajasVisibles.map(function (box) {
      return crearSolido(box, sx, sy, sz, gap, camera, deckH);
    }).sort(function (a, b) { return a.depth - b.depth; });

    var etiquetas = [];
    solids.forEach(function (solid) {
      var projected = solid.corners.map(function (p) { return screen(p[0], p[1], p[2]); });
      var base = tonoClase(rgbDeClase(solid.box.box_class));
      // Variacion de luminancia por celda: separa dos cajas vecinas del mismo
      // nivel sin cambiarles el tono, que es lo que codifica el nivel.
      var tonos = [0.92, 1.06, 0.98, 1.12, 0.88, 1.02, 0.95];
      var cuerpo = escalar(base, tonos[solid.box.cell % tonos.length]);
      CARAS.forEach(function (face) {
        var dot = face.normal[0] * camWorld[0] + face.normal[1] * camWorld[1] + face.normal[2] * camWorld[2];
        if (dot <= 1e-6) { return; }
        var facePoints = face.ids.map(function (id) { return projected[id]; });
        var edgeWidth = Math.max(1, 1.3 * u);
        // Relleno con degradado y UNA sola arista: clara arriba (luz de
        // canto), oscura en los laterales (separa caja de caja). Antes eran
        // dos contornos, uno casi negro y grueso, y la escena se leia como un
        // dibujo a tinta.
        poligono(facePoints, caraGradiente(facePoints, cuerpo, face.shade), null, 0);
        brilloCara(facePoints, face.shade);
        poligono(facePoints, null,
                 face.shade > 0.95
                   ? colorDeNivel(solid.box.level, 0.95)
                   : colorDeNivel(solid.box.level, 0.50),
                 edgeWidth);
      });
      // Segundo contorno solo para la cara superior: es la "tapa" inequivoca
      // de esta caja y refuerza el color del NIVEL, que es lo que hay que
      // distinguir de un vistazo. La clase sigue viviendo en el relleno.
      var topPoints = CARAS[4].ids.map(function (id) { return projected[id]; });
      poligono(topPoints, null, colorDeNivel(solid.box.level, 0.98), Math.max(1, 2.1 * u));
      // La etiqueta NO se pinta aqui: si se dibuja dentro de este bucle, el
      // siguiente solido (mas cercano a camara) la tapa a medias y quedan
      // textos cortados. Se acumula y se pinta cuando toda la pila ya esta.
      etiquetas.push({
        texto: solid.box.cell + 'N' + solid.box.level,
        x: projected.reduce(function (sum, p) { return sum + p[0]; }, 0) / 8,
        y: projected.reduce(function (sum, p) { return sum + p[1]; }, 0) / 8,
      });
    });

    // Segunda pasada: etiquetas discretas -- sombra suave en vez de contorno
    // negro grueso, y gris claro en vez de cian puro, para que se lean sin
    // robarle atencion a la geometria. Solo celda+nivel: el nombre de la
    // clase iba debajo de cada caja y con el pallet lleno son decenas de
    // textos repetidos; el dato ya esta en el color de la caja y en el
    // conteo por clase de la ficha de la camara.
    ctx.save();
    ctx.font = '700 ' + (12 * u) + 'px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.shadowColor = 'rgba(0,0,0,0.75)';
    ctx.shadowBlur = 3 * u;
    ctx.fillStyle = 'rgba(244,250,255,0.98)';
    etiquetas.forEach(function (etiqueta) {
      ctx.fillText(etiqueta.texto, etiqueta.x, etiqueta.y);
    });
    ctx.restore();

    provisionalesVisibles.map(function (box) {
      return crearSolido(box, sx, sy, sz, gap, camera, deckH);
    }).sort(function (a, b) {
      return a.depth - b.depth;
    }).forEach(function (solid) {
      dibujarProvisional(solid, screen, camWorld, u);
    });

    // Un solape intranivel no debe parecer una caja de otro piso. Se marca
    // como conflicto sobre la huella exacta que el cerebro reporto: rojo,
    // cruz y texto con las dos celdas. Si no hay solapes, no se dibuja nada.
    (escena.overlaps || []).filter(function (overlap) {
      return nivelVisible === null || overlap.level === nivelVisible;
    }).forEach(function (overlap) {
      var cx = ((overlap.u0 + overlap.u1) / 2) * sx;
      var cy = ((overlap.v0 + overlap.v1) / 2) * sy;
      var a = (overlap.u1 - overlap.u0) * sx;
      var b = (overlap.v1 - overlap.v0) * sy;
      var z0 = deckH + overlap.z0 * sz + gap * overlap.level;
      var h = overlap.height * sz;
      var conflict = esquinas(cx, cy, z0, a, b, h).map(function (p) {
        return screen(p[0], p[1], p[2]);
      });
      var top = CARAS[4].ids.map(function (id) { return conflict[id]; });
      poligono(top, color(CONFLICTO, 0.30), color(CONFLICTO, 0.98), Math.max(2, 1.8 * u));
      linea(top[0], top[2], color(CONFLICTO, 0.98), Math.max(1, 1.4 * u));
      linea(top[1], top[3], color(CONFLICTO, 0.98), Math.max(1, 1.4 * u));
      var ox = top.reduce(function (sum, p) { return sum + p[0]; }, 0) / top.length;
      var oy = top.reduce(function (sum, p) { return sum + p[1]; }, 0) / top.length;
      // "C2/C7", no "2x7": son los IDS de las dos celdas que se pisan, no
      // una dimension de grilla -- con la 'x' en medio se leia como 2 por 7.
      var aviso = 'SOLAPE N' + overlap.level + ' C' + overlap.cell_a + '/C' + overlap.cell_b;
      ctx.font = '700 ' + (9 * u) + 'px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      var avisoW = ctx.measureText(aviso).width;
      ctx.fillStyle = 'rgba(45,7,8,0.90)';
      ctx.fillRect(ox - avisoW / 2 - 5 * u, oy - 9 * u, avisoW + 10 * u, 18 * u);
      ctx.fillStyle = 'rgba(255,238,235,0.98)';
      ctx.fillText(aviso, ox, oy);
    });

    // HUD integrado, compacto y de bajo contraste. Todo en unidades `u`
    // (pixeles CSS) y el ancho del panel MEDIDO sobre el texto real: con un
    // ancho fijo, "PALLET DIGITAL - 15 CAJAS" se salia de la caja en cuanto
    // el total pasaba a dos digitos o se enfocaba una capa.
    // El Canvas queda dedicado al gemelo 3D. El nombre, el conteo y la
    // leyenda de clases viven fuera del render para no saturar la escena.
  }

  function pedirRender() {
    if (framePendiente) { return; }
    framePendiente = true;
    window.requestAnimationFrame(render);
  }

  function aplicar(nuevoAz, nuevoEl) {
    az = ((nuevoAz % 360) + 360) % 360;
    el = limitarElevacion(nuevoEl);
    mostrarAngulos();
    pedirRender();
  }

  async function actualizarEscena() {
    try {
      var response = await fetch('/cam/' + CAM_ID + '/iso/scene', { cache: 'no-store' });
      if (response.ok) {
        var payload = await response.json();
        if (payload.seq !== seq) {
          seq = payload.seq;
          escena = payload.scene;
          vista = payload.view;
          colores = payload.colors;
          if (escena && vista) { isoLoading.classList.add('oculto'); }
          if (nivelVisible !== null && nivelesOcupados().indexOf(nivelVisible) === -1) {
            enfocarNivel(null);
          }
          pedirRender();
        }
      }
    } catch (_error) {
      // sin conexion: se reintenta abajo, sin nada que mostrar en pantalla.
    } finally {
      window.setTimeout(actualizarEscena, 100);
    }
  }

  function irAVista(azValor, elValor, enfocarTop) {
    // Salta a una vista predefinida: enfoca o no la capa de arriba y anima
    // la camara al angulo pedido. Se dispara solo desde el widget de
    // brujula/cubo sobre el canvas -- las vistas ya no viven como botones.
    var levels = nivelesOcupados();
    enfocarNivel(enfocarTop && levels.length ? levels[levels.length - 1] : null);
    aplicar(azValor, elValor);
  }

  function dentroDePoligono(x, y, puntos) {
    // Ray casting estandar: cuenta cuantas aristas del poligono cruzan la
    // horizontal que pasa por (x, y) a la derecha del punto.
    var dentro = false;
    for (var i = 0, j = puntos.length - 1; i < puntos.length; j = i, i += 1) {
      var pi = puntos[i];
      var pj = puntos[j];
      var cruza = (pi[1] > y) !== (pj[1] > y)
        && x < (pj[0] - pi[0]) * (y - pi[1]) / (pj[1] - pi[1]) + pi[0];
      if (cruza) { dentro = !dentro; }
    }
    return dentro;
  }

  // Cara del cubo -> vista (misma semantica que tenian los botones ISO /
  // FRONTAL / LATERAL / PLANTA, ahora disparada al clickear la cara).
  var VISTA_POR_CARA = {
    SUP: { az: 0, el: 89, focusTop: true },
    N: { az: 0, el: 0, focusTop: false },
    S: { az: 180, el: 0, focusTop: false },
    E: { az: 90, el: 0, focusTop: false },
    O: { az: 270, el: 0, focusTop: false },
  };

  function detectarClickBrujula(clientX, clientY) {
    if (!brujulaHit) { return false; }
    var rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) { return false; }
    var escalaX = canvas.width / rect.width;
    var escalaY = canvas.height / rect.height;
    var x = (clientX - rect.left) * escalaX;
    var y = (clientY - rect.top) * escalaY;
    var margen = 20 * (canvas.width / rect.width);

    for (var i = 0; i < brujulaHit.caras.length; i += 1) {
      var cara = brujulaHit.caras[i];
      if (dentroDePoligono(x, y, cara.puntos)) {
        var destino = VISTA_POR_CARA[cara.etiqueta];
        if (destino) { irAVista(destino.az, destino.el, destino.focusTop); }
        return true;
      }
    }
    for (var j = 0; j < brujulaHit.cardinales.length; j += 1) {
      var c = brujulaHit.cardinales[j];
      if (Math.hypot(x - c.x, y - c.y) <= margen) {
        irAVista(c.az, el, false);
        return true;
      }
    }
    var dCentro = Math.hypot(x - brujulaHit.cx, y - brujulaHit.cy);
    if (dCentro <= brujulaHit.radioAnillo) {
      // Click en el anillo, fuera de toda cara o letra: vuelve a la
      // isometrica por defecto, igual que el viejo boton ISO.
      if (orbitando) { pausarOrbita(); }
      irAVista(AZ0, EL0, false);
      return true;
    }
    return false;
  }

  // -- Control de orbita: pausar / reanudar ---------------------------------
  // pausarOrbita() congela el estado actual y lo convierte en offsets para
  // que reanudarOrbita() pueda continuar exactamente desde ahi.
  function pausarOrbita() {
    orbitando = false;
    selectorOrbitar.textContent = 'ORBITAR';
    selectorOrbitar.classList.remove('activa');
  }

  function reanudarOrbita() {
    // Calcular offsets para que la formula produzca az/el actuales en t=0.
    orbitT0 = performance.now() / 1000;
    orbitOffsetAz = az - ORBIT_ISO_AZ;
    orbitOffsetEl = 0;  // la oscilacion empieza en fase 0 (sin(0)=0)
    orbitando = true;
    selectorOrbitar.textContent = 'PAUSAR';
    selectorOrbitar.classList.add('activa');
    pedirRender();
  }

  canvas.addEventListener('pointerdown', function (e) {
    if (detectarClickBrujula(e.clientX, e.clientY)) {
      e.preventDefault();
      return;
    }
    // El drag manual pausa la orbita: el usuario tomo el control.
    if (orbitando) { pausarOrbita(); }
    arrastrando = true;
    ultimoX = e.clientX;
    ultimoY = e.clientY;
    canvas.setPointerCapture?.(e.pointerId);
    lienzo.classList.add('girando');
    e.preventDefault();
  });
  canvas.addEventListener('pointerup', function () {
    arrastrando = false;
    lienzo.classList.remove('girando');
  });
  canvas.addEventListener('pointercancel', function () {
    arrastrando = false;
    lienzo.classList.remove('girando');
  });
  canvas.addEventListener('pointermove', function (e) {
    if (!arrastrando) { return; }
    aplicar(az - (e.clientX - ultimoX) * GRADOS_POR_PIXEL,
            el + (e.clientY - ultimoY) * GRADOS_POR_PIXEL);
    ultimoX = e.clientX;
    ultimoY = e.clientY;
  });
  canvas.addEventListener('wheel', function (e) {
    if (orbitando) { pausarOrbita(); }
    aplicar(az, el + (e.deltaY > 0 ? GRADOS_POR_RUEDA : -GRADOS_POR_RUEDA));
    e.preventDefault();
  }, { passive: false });

  selectorExplosion.addEventListener('click', function () {
    explotado = !explotado;
    selectorExplosion.textContent = explotado ? 'EXPANDIDO' : 'APILADO';
    selectorExplosion.classList[explotado ? 'add' : 'remove']('activa');
    pedirRender();
  });

  selectorIsometrica.addEventListener('click', function () {
    // La vista isometrica es una posicion fija: detener la orbita evita que
    // el siguiente frame la vuelva a mover inmediatamente.
    if (orbitando) { pausarOrbita(); }
    enfocarNivel(null);
    aplicar(AZ0, EL0);
  });

  selectorOrbitar.addEventListener('click', function () {
    if (orbitando) {
      pausarOrbita();
    } else {
      reanudarOrbita();
    }
  });

  selectorCapa.addEventListener('click', function () {
    var levels = nivelesOcupados();
    if (!levels.length || nivelVisible === levels[levels.length - 1]) {
      enfocarNivel(null);
      return;
    }
    var current = nivelVisible === null ? -1 : levels.indexOf(nivelVisible);
    enfocarNivel(levels[current + 1]);
  });

  window.addEventListener('resize', pedirRender);
  // El resize de window no alcanza cuando este documento vive en un iframe
  // embebido en un layout con calc()/aspect-ratio pesados (dashboard, ISO):
  // el iframe no cambia de tamano de VENTANA, cambia de tamano de CAJA
  // despues del primer layout, y sin esto el canvas quedaba con el tamano
  // (chico) que tenia en el primer pintado, con el resto del contenedor
  // vacio para siempre.
  if (window.ResizeObserver) {
    new ResizeObserver(pedirRender).observe(lienzo);
  }
  // Red de seguridad: si por lo que sea ni el ResizeObserver ni el primer
  // render agarraron el tamano final del layout (fuentes/iframe/host
  // terminando de acomodarse), se reintenta una vez mas apenas arranca.
  window.setTimeout(pedirRender, 250);

  // La animacion arranca activa: el texto del boton indica la accion
  // disponible, no el estado.
  selectorOrbitar.textContent = 'PAUSAR';
  mostrarAngulos();
  actualizarZoomVideo();
  enfocarNivel(null);
  pedirRender();
  // Vista "camera" no muestra el 3D (queda display:none): no tiene sentido
  // pedirle la escena al servidor cada 100ms solo para tenerla oculta.
  if (NECESITA_ISO) {
    actualizarEscena();
  } else {
    isoLoading.classList.add('oculto');
  }
})();
