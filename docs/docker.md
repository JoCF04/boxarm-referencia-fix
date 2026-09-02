# Ejecutar en la Jetson dentro de Docker

El contenedor necesita permisos explicitos sobre los dispositivos de video.
Sin ellos, `modo: camara` falla aunque el codigo sea correcto.

## El sintoma

`main.py` arranca bien, Flask levanta, pero las tres camaras entran en
reconexion perpetua:

```
[Camara 1] MJPEG + NVDEC fallo: pipeline no llego a PLAYING
[Camara 1] YUYV crudo fallo: pipeline no llego a PLAYING
[Camara 1] GStreamer nativo fallo -- usando V4L2 plano
[Camara 1] V4L2 no pudo abrir /dev/video0 (ocupado o sin permisos)
[Camara 1] no se pudo abrir /dev/video0, reintentando en 2.0s...
```

Lo confuso es que `/dev/video0` **existe** dentro del contenedor: por eso
`os.path.exists()` pasa y el lector sigue reintentando. Lo que falla es el
`open()`, con EPERM.

## La causa

Montar `/dev` en el contenedor (`--volume /dev:/dev`) hace visibles los nodos,
pero **no concede permiso para abrirlos**. Ese permiso lo da el cgroup de
dispositivos, y hay que pedirlo aparte:

- `--device /dev/videoN:/dev/videoN` por cada camara, o
- `--device-cgroup-rule='c 81:* rmw'` (major 81 = `video4linux`)

`docker/build.sh` hace las dos cosas, que es lo robusto.

## Comprobacion rapida

Dentro del contenedor:

```bash
ls -l /dev/video*
python3 -c "import cv2; c=cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2); print(c.isOpened())"
```

Si eso imprime `False` dentro del contenedor y `True` en el host, es
exactamente este problema. Los nodos impares (`/dev/video1`, `/dev/video3`,
...) son metadata UVC y no entregan frames -- las camaras reales son los pares,
que es lo que refleja `index` en `configs/pipeline.yaml`.

## Uso normal: un comando

En el host, desde cualquier directorio del repo:

```bash
bash docker/build.sh
```

Hace todo el ciclo:

1. Detecta la version de L4T (`nvidia-l4t-core`) y elige la imagen base y la
   version de Python acordes -- JetPack 5 (L4T 35.x) usa
   `dustynv/jetson-inference:r35.4.1` con Python 3.8.
2. Construye `cnm_boxarm` instalando **encima** de esa base. Los 12 GB de la
   base se reutilizan tal cual: solo se anaden las capas de apt y pip.
3. Si ya existe un contenedor `cnm_robot`, pregunta si rehacerlo (los permisos
   de dispositivo solo se fijan al crear, no se le pueden anadir despues).
4. Pregunta si arrancar **con o sin Jupyter Lab**.
5. Crea el contenedor con todos los `/dev/video*` del host y `--network=host`.

Luego:

```bash
docker exec -ti cnm_robot python3 main.py
```

Con Jupyter Lab (modo 1) queda en `http://<ip-jetson>:8888`, token `cnm-robot`.

### Que trae la imagen

Las dependencias se declaran **todas** en
`docker/requirements_container.txt`. Del repo no se copia nada mas: el codigo
se monta por volumen en `/robot_systm/src`, que es el `WORKDIR` de la imagen,
asi que al entrar al contenedor ya estas en el codigo y editar un `.py` en el
host no obliga a reconstruir.

El archivo se parte en dos por una marca:

```
pyyaml, flask, matplotlib, ..., lapx, jupyterlab   <- pip normal
# --- se instalan con --no-deps ---
ultralytics==8.4.127                                <- pip --no-deps
```

`ultralytics` va con `--no-deps` porque su arbol declara `torch`,
`torchvision` y `opencv-python`: dejar que pip los resuelva sustituye las
builds CUDA de la imagen base por wheels de CPU y **se pierde la GPU sin que
nada avise**. Sus dependencias reales estan declaradas arriba a mano. Va
fijada a 8.3.63 porque la base de JetPack 5 trae Python 3.8 y las versiones
recientes ya no lo soportan.

Por eso `torch`, `torchvision`, `numpy` y `cv2` no aparecen en el requirements:
ya vienen compilados contra CUDA en la base y no se tocan.

El runtime usa `model.predict()` y no requiere la asignacion hungara de un tracker. Por apt entran `v4l-utils` y `usbutils`, para diagnosticar
las camaras desde dentro (`v4l2-ctl --list-formats-ext`).

Anadir un paquete es editar el `.txt` y volver a construir: Docker cachea la
capa de apt y solo rehace la de pip.

### Variables

Todo parametrizable por entorno, sin editar el script:

| Variable | Por defecto | Que es |
|---|---|---|
| `BOXARM_IMAGE` | `cnm_boxarm` | etiqueta de la imagen |
| `BOXARM_CONTAINER` | `cnm_robot` | nombre del contenedor |
| `BOXARM_MOUNT_POINT` | `/robot_systm/src` | donde se monta el repo dentro |
| `BOXARM_JUPYTER_PORT` | `8888` | puerto de Jupyter Lab |
| `BOXARM_JUPYTER_TOKEN` | `cnm-robot` | token de Jupyter Lab |

```bash
BOXARM_CONTAINER=pruebas bash docker/build.sh
```

## El docker run a mano

Por si hace falta crear el contenedor sin pasar por el script. `--privileged`
mas los `--device` de `/dev/video0` a `/dev/video5` cubren las tres camaras y
sus nodos de metadata; `--network=host` expone todos los puertos sin enumerar
ninguno.

```bash
docker run --runtime nvidia --privileged -d --ipc=host \
      --device /dev/bus/usb:/dev/bus/usb \
      --device /dev/video0:/dev/video0 \
      --device /dev/video1:/dev/video1 \
      --device /dev/video2:/dev/video2 \
      --device /dev/video3:/dev/video3 \
      --device /dev/video4:/dev/video4 \
      --device /dev/video5:/dev/video5 \
      --device-cgroup-rule='c 81:* rmw' \
      --network=host \
      -v /tmp/argus_socket:/tmp/argus_socket \
      -v /run/udev:/run/udev:ro \
      -v $(pwd):/robot_systm/src \
      -w /robot_systm/src \
      --name cnm_robot \
      --restart unless-stopped cnm_boxarm
```

Que hace cada parte que importa:

| Flag | Para que |
|---|---|
| `--privileged` | concede los permisos de dispositivo que faltaban (el EPERM) |
| `--device /dev/videoN` | mapea cada nodo de camara; **0 a 5**, pares reales e impares metadata |
| `--device /dev/bus/usb` | permite que la UVC se reenumere sin reiniciar el contenedor |
| `--device-cgroup-rule='c 81:* rmw'` | major 81 = `video4linux`; redundante con `--privileged`, util si algun dia lo quitas |
| `--network=host` | **todos los puertos**; cambiar `port` en `pipeline.yaml` no obliga a tocar nada |
| `--ipc=host` | memoria compartida para los procesos por camara |
| `--restart unless-stopped` | vuelve solo tras un reinicio de la Jetson |
| `-d` | queda vivo en segundo plano (el `CMD` de la imagen es `tail -f /dev/null`) |

## Recrear un contenedor que ya existe

Los permisos de dispositivo se fijan **al crear** el contenedor: no se pueden
anadir a uno que ya corre. Si el que tienes se creo sin los `--device` (el caso
tipico: `docker run ... tail -f /dev/null` a secas), hay que rehacerlo.

No se pierde nada al recrearlo: el repo va montado por volumen, no vive dentro
del contenedor.

Conviene parar el viejo antes de levantar el nuevo. No compite por las camaras
(no puede abrirlas), pero con `--network=host` en los dos chocarian en el mismo
puerto, y `--restart unless-stopped` lo resucita en cada arranque de la Jetson:

```bash
docker stop <contenedor-viejo>
```

Primero mira que monta el actual, para conservar la ruta del host:

```bash
docker ps -a --format '{{.Names}}	{{.Image}}'
docker inspect <contenedor-viejo> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"
"}}{{end}}'
```

Luego basta con volver a lanzar el script: detecta que el contenedor existe,
pregunta si rehacerlo y lo recrea ya con los dispositivos.

```bash
bash docker/build.sh
```

Comprobar que ahora si:

```bash
docker exec -ti cnm_robot python3 -c "import cv2; c=cv2.VideoCapture('/dev/video0',cv2.CAP_V4L2); print(c.isOpened())"
```

`True` -- ya se puede correr `docker exec -ti cnm_robot python3 main.py`.

## Distinguir EPERM de EBUSY

Cuando `isOpened()` da `False` siendo root dentro del contenedor, el errno dice
cual de los dos problemas es:

```bash
dd if=/dev/video0 of=/dev/null bs=1 count=1
cat /sys/fs/cgroup/devices/devices.list 2>/dev/null || cat /sys/fs/cgroup/devices.allow 2>/dev/null
```

- `Operation not permitted` -- es el cgroup del contenedor: recrearlo como
  arriba. Que la lista de devices no incluya `c 81:* rwm` lo confirma.
- `Device or resource busy` -- algo lo tiene tomado; se busca **en el host**
  con `sudo fuser -v /dev/video0`.

Ser root dentro del contenedor no cambia nada aqui: los `crw-rw---- root video`
de `ls -l /dev/video*` los cumple root de sobra, pero el cgroup de dispositivos
se aplica igual.

## Si aun asi no abre

El orden de descarte, de mas a menos probable:

1. **Dispositivo ocupado.** Otro proceso lo tiene tomado -- tipicamente un run
   anterior que quedo colgado, o el mismo `main.py` corriendo en el host.
   `sudo fuser -v /dev/video0` lo dice.
2. **Resolucion no soportada.** `v4l2-ctl -d /dev/video0 --list-formats-ext`
   lista lo que la camara da de verdad. Es comun que ofrezca MJPEG a 1280x720
   pero YUYV solo hasta 640x480. Se ajusta con `cap_width` / `cap_height` en
   `configs/pipeline.yaml`.
3. **Permisos de usuario en el host.** `sudo usermod -a -G video $USER` y
   volver a entrar.

## Sin camaras

Para trabajar sin hardware, `modo: video` en `configs/pipeline.yaml` lee los
`.mp4` de `videos/` en vez de `/dev/videoN`. No toca GStreamer ni V4L2, asi
que corre en cualquier PC y sirve para descartar si un problema es de captura
o del resto del pipeline.
