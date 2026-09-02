# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : Imagen con las dependencias ya instaladas. Antes se metian a
#              mano dentro del contenedor y se perdian al recrearlo (que es
#              justo lo que hay que hacer para concederle los /dev/video*).
# -----------------------------------------------------------------------
ARG BASE_IMAGE=dustynv/jetson-inference:r35.4.1
ARG PYTHON_VERSION=3.8

FROM ${BASE_IMAGE}

ARG PYTHON_VERSION
WORKDIR /robot_systm/external

# v4l-utils es lo que permite diagnosticar las camaras desde dentro del
# contenedor (v4l2-ctl --list-formats-ext), que es justo lo que hace falta
# cuando una no abre. El resto es zona horaria y pip al dia.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        v4l-utils \
        usbutils && \
    ln -sf /usr/share/zoneinfo/America/Lima /etc/localtime && \
    dpkg-reconfigure -fnoninteractive tzdata && \
    python3 -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Solo el requirements: el codigo va por volumen, no dentro de la imagen.
COPY requirements_container.txt .

# El requirements se parte en dos por la marca "--- se instalan con --no-deps":
# lo de arriba se resuelve normal, lo de abajo va con --no-deps para que pip no
# reemplace torch/torchvision/opencv de la imagen base por wheels de CPU. Una
# sola lista de dependencias, dos formas de instalarla.
RUN MARK='--- se instalan con --no-deps' && \
    sed "/$MARK/,\$d" requirements_container.txt > /tmp/req-deps.txt && \
    sed "1,/$MARK/d"  requirements_container.txt > /tmp/req-nodeps.txt && \
    pip3 install --no-cache-dir --default-timeout=1000 -r /tmp/req-deps.txt && \
    pip3 install --no-cache-dir --default-timeout=1000 --no-deps -r /tmp/req-nodeps.txt && \
    rm -f /tmp/req-deps.txt /tmp/req-nodeps.txt

# Chromium headless renderiza exactamente /cam/<id> para producir los videos
# `iso` y `dashboard`; no se sustituye por el ISO raster legado de OpenCV.
RUN python3 -m playwright install --with-deps chromium

# Ultralytics comprueba versiones y escribe ajustes al importar: sin esto
# intenta salir a internet en cada arranque.
ENV YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    PYTHONUNBUFFERED=1

# Falla el build si falta alguna dependencia. CUDA no se comprueba aqui: el
# build no tiene GPU asignada, eso se verifica al correr el contenedor.
# polars y la version de ultralytics se imprimen a proposito: son las dos
# piezas que deciden si los pesos YOLO26 cargan en este Python 3.8.
RUN python3 -c "import torch, cv2, yaml, flask, lap, polars, playwright, ultralytics; from ultralytics import YOLO; print('torch', torch.__version__, 'cv2', cv2.__version__, 'ultralytics', ultralytics.__version__, 'polars', polars.__version__)"

# El codigo se monta aqui por volumen al arrancar (ver docker/build.sh),
# asi editarlo en el host no obliga a reconstruir la imagen.
WORKDIR /robot_systm/src

CMD ["tail", "-f", "/dev/null"]
