import cv2
import glob
import os
import threading
import time
from datetime import datetime

from flask import Flask, Response, render_template, jsonify

# ── Configuración ──────────────────────────────────────────────
WIDTH, HEIGHT   = 1920, 1080
JPEG_QUALITY    = 75
FPS             = 20
PORT            = 5000
RECORD_ROOT     = "data/cam"          # data/cam/<i>/record/cam-<i>-<fecha>.mp4
RECORD_FOURCC   = cv2.VideoWriter_fourcc(*"mp4v")
# ───────────────────────────────────────────────────────────────

app          = Flask(__name__)
camera_paths = []   # se llena al escanear
cams         = {}   # idx -> dict con estado por cámara


# ── Escaneo de cámaras disponibles ────────────────────────────
def scan_cameras():
    """Devuelve lista de rutas /dev/videoN que abren correctamente."""
    found = []
    devices = sorted(glob.glob('/dev/video*'))
    print(f"[SCAN] Dispositivos encontrados: {devices}")

    for dev in devices:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.size > 0:
            found.append(dev)
            print(f"[OK]   {dev} — entrega frames")
        else:
            print(f"[--]   {dev} — abierta pero sin frames (metadata/control)")

    return found


# ── Apertura de cámara ─────────────────────────────────────────
def open_camera(dev_path):
    cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] No se puede abrir {dev_path}")
        return None

    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Descartar frames vacíos iniciales
    for _ in range(15):
        try:
            ret, frame = cap.read()
        except cv2.error:
            time.sleep(0.1)
            continue
        if ret and frame is not None and frame.size > 0:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[OK] {dev_path} listo — {w}x{h}")
            return cap
        time.sleep(0.1)

    cap.release()
    print(f"[ERROR] {dev_path}: no entrega frames")
    return None


def record_path(idx):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(RECORD_ROOT, str(idx), "record")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"cam-{idx}-{ts}.mp4")


# ── Hilo de captura (una por cámara; solo corre mientras está activa) ─
def capture_loop(idx, dev_path, stop_event):
    cam = cams[idx]

    cap = open_camera(dev_path)
    if cap is None:
        cam['enabled'] = False
        return

    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    while not stop_event.is_set():
        try:
            ret, frame = cap.read()
        except cv2.error:
            time.sleep(0.03)
            continue
        if not ret or frame is None:
            time.sleep(0.03)
            continue

        _, buf = cv2.imencode('.jpg', frame, params)
        with cam['lock']:
            cam['frame'] = buf.tobytes()

        with cam['writer_lock']:
            if cam['recording'] and cam['writer'] is not None:
                cam['writer'].write(frame)

    cap.release()
    with cam['writer_lock']:
        if cam['writer'] is not None:
            cam['writer'].release()
            cam['writer'] = None
    with cam['lock']:
        cam['frame'] = None


def enable_camera(idx):
    """Abre la cámara y arranca su hilo de captura (consume CPU/GPU/ancho de banda)."""
    cam = cams[idx]
    if cam['enabled']:
        return
    cam['enabled']    = True
    cam['stop_event'] = threading.Event()
    cam['thread'] = threading.Thread(
        target=capture_loop, args=(idx, camera_paths[idx], cam['stop_event']), daemon=True
    )
    cam['thread'].start()
    print(f"[CAM {idx}] activada")


def disable_camera(idx):
    """Libera la cámara y detiene su hilo — no consume recursos mientras esté apagada."""
    cam = cams[idx]
    if not cam['enabled']:
        return
    if cam['recording']:
        stop_recording(idx)
    cam['enabled'] = False
    cam['stop_event'].set()
    if cam['thread'] is not None:
        cam['thread'].join(timeout=3)
    cam['thread'] = None
    print(f"[CAM {idx}] desactivada — recursos liberados")


def start_recording(idx):
    cam = cams[idx]
    with cam['writer_lock']:
        if cam['recording']:
            return cam['path']
        path = record_path(idx)
        cam['writer'] = cv2.VideoWriter(path, RECORD_FOURCC, FPS, (WIDTH, HEIGHT))
        cam['recording'] = True
        cam['path'] = path
        print(f"[REC] Cámara {idx} -> {path}")
        return path


def stop_recording(idx):
    cam = cams[idx]
    with cam['writer_lock']:
        cam['recording'] = False
        if cam['writer'] is not None:
            cam['writer'].release()
            cam['writer'] = None
        print(f"[REC] Cámara {idx} detenida")


@app.route('/')
def index():
    return render_template('index.html', total=len(camera_paths))


@app.route('/feed/<int:idx>')
def feed(idx):
    if idx not in cams:
        return jsonify(ok=False, error='Cámara no disponible'), 404
    if not cams[idx]['enabled']:
        return jsonify(ok=False, error='Cámara apagada'), 409
    return Response(stream(idx), mimetype='multipart/x-mixed-replace; boundary=frame')


def stream(idx):
    cam = cams[idx]
    while True:
        with cam['lock']:
            f = cam['frame']
        if f is None:
            time.sleep(0.05)
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f + b'\r\n'
        time.sleep(1 / FPS)


@app.route('/cam/<int:idx>/enable', methods=['POST'])
def cam_enable(idx):
    if idx not in cams:
        return jsonify(ok=False, error='Cámara no disponible'), 404
    enable_camera(idx)
    return jsonify(ok=True)


@app.route('/cam/<int:idx>/disable', methods=['POST'])
def cam_disable(idx):
    if idx not in cams:
        return jsonify(ok=False, error='Cámara no disponible'), 404
    disable_camera(idx)
    return jsonify(ok=True)


@app.route('/record/<int:idx>/start', methods=['POST'])
def record_start(idx):
    if idx not in cams:
        return jsonify(ok=False, error='Cámara no disponible'), 404
    if not cams[idx]['enabled']:
        return jsonify(ok=False, error='Cámara apagada'), 409
    path = start_recording(idx)
    return jsonify(ok=True, path=path)


@app.route('/record/<int:idx>/stop', methods=['POST'])
def record_stop(idx):
    if idx not in cams:
        return jsonify(ok=False, error='Cámara no disponible'), 404
    stop_recording(idx)
    return jsonify(ok=True)


@app.route('/status')
def status():
    data = []
    for idx, cam in cams.items():
        data.append({
            'idx': idx,
            'path': camera_paths[idx],
            'res': f'{WIDTH}x{HEIGHT}',
            'enabled': cam['enabled'],
            'recording': cam['recording'],
        })
    return jsonify(cams=data, port=PORT)


# ── Main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    import socket

    print("\n[SCAN] Buscando cámaras en /dev/video* ...")
    camera_paths = scan_cameras()

    if not camera_paths:
        print("\n[ERROR] No se encontró ninguna cámara funcional.")
        print("  Comprueba con: ls /dev/video*")
        print("  Permisos:      sudo usermod -a -G video $USER")
        exit(1)

    print(f"\n[INFO] Cámaras disponibles: {camera_paths}")

    for idx in range(len(camera_paths)):
        cams[idx] = {
            'frame': None,
            'lock': threading.Lock(),
            'recording': False,
            'writer': None,
            'writer_lock': threading.Lock(),
            'path': None,
            'enabled': False,
            'stop_event': None,
            'thread': None,
        }
        enable_camera(idx)

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = '127.0.0.1'

    print(f"[INFO] Servidor en http://{ip}:{PORT}\n")

    time.sleep(1.5)
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False)
