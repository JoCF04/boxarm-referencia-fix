#!/bin/bash
# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 24-08-26
# Reason     : Construye la imagen del proyecto sobre la de jetson-inference
#              y crea el contenedor con acceso a las camaras. Sin los
#              --device los nodos /dev/videoN se ven dentro pero no se
#              pueden abrir (EPERM) y main.py cae en reconexion perpetua.
# -----------------------------------------------------------------------

# Colores de whiptail en tema oscuro, para que no se altere la paleta.
export NEWT_COLORS="root=,black;window=,black;border=white,black;textbox=white,black;button=black,lightgray;actbutton=lightgray,black;compactbutton=black,lightgray;title=yellow,black;listbox=white,black;actlistbox=black,lightgray;actsellistbox=black,lightgray;checkbox=white,black;actcheckbox=black,lightgray"

# La raiz del repo sale de la ubicacion de este script, no del cwd.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

IMAGE_NAME="${BOXARM_IMAGE:-cnm_boxarm}"
CONTAINER_NAME="${BOXARM_CONTAINER:-cnm_robot}"
MOUNT_POINT="${BOXARM_MOUNT_POINT:-/robot_systm/src}"
JUPYTER_PORT="${BOXARM_JUPYTER_PORT:-8888}"
JUPYTER_TOKEN="${BOXARM_JUPYTER_TOKEN:-cnm-robot}"
# BOXARM_NO_CACHE=1 rehace el build entero ignorando capas cacheadas. Hace
# falta cuando cambia una version dentro de requirements_container.txt y
# docker reusa la capa de pip vieja porque el nombre del archivo no cambio.
NO_CACHE_ARGS=()
if [ "${BOXARM_NO_CACHE:-0}" = "1" ]; then
    echo "BOXARM_NO_CACHE=1 -- build sin cache (tarda mucho mas)"
    NO_CACHE_ARGS=(--no-cache --pull)
fi

# --- UI Helpers (Whiptail TUI) ---
ask_yes_no() {
    local prompt="$1"
    if command -v whiptail >/dev/null 2>&1; then
        if whiptail --title "CNM Robotic Box Arm" --yes-button "Si" --no-button "No" --yesno "$prompt" 10 65 3>&1 1>&2 2>&3; then
            echo "s"
        else
            echo "n"
        fi
    else
        read -p "$prompt (s/n): " ans
        echo "$ans"
    fi
}

# --- Deteccion de JetPack ---------------------------------------------------
L4T_VERSION_STRING=$(dpkg -l | grep "nvidia-l4t-core" | awk '{print $3}')

if [ -z "$L4T_VERSION_STRING" ]; then
    echo "Error: No se pudo detectar la version de JetPack (nvidia-l4t-core no encontrado)."
    echo "Asegurate de estar ejecutando este script en un dispositivo Jetson."
    exit 1
fi

echo "Version detectada de L4T: $L4T_VERSION_STRING"

if [[ "$L4T_VERSION_STRING" == 35.* ]]; then
    echo "Detectado JetPack 5 (L4T 35.x)"
    BASE_IMAGE="dustynv/jetson-inference:r35.4.1"
    PYTHON_VERSION="3.8"
elif [[ "$L4T_VERSION_STRING" == 36.3.* ]] || [[ "$L4T_VERSION_STRING" == 36.4.* ]]; then
    echo "Detectado JetPack 6.1+ (L4T 36.3.x/36.4.x)"
    BASE_IMAGE="dustynv/jetson-inference:r36.4.0"
    PYTHON_VERSION="3.10"
elif [[ "$L4T_VERSION_STRING" == 36.* ]]; then
    echo "Detectado JetPack 6.0 (L4T 36.0.x-36.2.x)"
    BASE_IMAGE="dustynv/jetson-inference:r36.2.0"
    PYTHON_VERSION="3.10"
else
    echo "ERROR: Version de L4T no soportada: $L4T_VERSION_STRING"
    echo "Versiones soportadas:"
    echo "  - JetPack 5: L4T 35.x"
    echo "  - JetPack 6.0: L4T 36.0.x - 36.2.x"
    echo "  - JetPack 6.1+: L4T 36.3.x - 36.4.x"
    echo "Por favor actualiza build.sh para agregar soporte para tu version."
    exit 1
fi

echo "Construyendo imagen con:"
echo "  BASE_IMAGE: $BASE_IMAGE"
echo "  PYTHON_VERSION: $PYTHON_VERSION"

# El contexto es docker/ y no la raiz: lo unico que se copia a la imagen es
# requirements_container.txt. El codigo entra por volumen al arrancar.
docker build \
    "${NO_CACHE_ARGS[@]}" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    -f docker/Dockerfile \
    -t "$IMAGE_NAME" \
    docker

if [ $? -ne 0 ]; then
    echo "Error: El build de Docker fallo"
    exit 1
fi

echo ""
echo "Build completado exitosamente"
echo ""

# --- Contenedor -------------------------------------------------------------
# Los permisos de dispositivo se fijan al CREAR el contenedor: a uno que ya
# existe no se le pueden anadir, por eso la unica salida es recrearlo.
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "El contenedor '$CONTAINER_NAME' ya existe"
    respuesta=$(ask_yes_no "Deseas eliminarlo y crear uno nuevo?")
    if [[ "$respuesta" =~ ^[Ss]$ ]]; then
        echo "Eliminando contenedor existente..."
        docker rm -f "$CONTAINER_NAME"
    else
        echo "Manteniendo contenedor existente. Saliendo..."
        exit 0
    fi
fi

# Se mapea cada /dev/videoN que exista en el host (los pares son las camaras,
# los impares metadata UVC). La regla de cgroup del major 81 (video4linux) es
# la parte que de verdad concede el permiso.
DEVICE_ARGS=()
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    DEVICE_ARGS+=(--device "$dev:$dev")
done

if [ ${#DEVICE_ARGS[@]} -eq 0 ]; then
    echo "AVISO: no hay ningun /dev/video* en el host -- el contenedor arranca sin camaras"
fi

echo ""
echo "Selecciona el modo de ejecucion:"
if command -v whiptail >/dev/null 2>&1; then
    modo=$(whiptail --title "CNM Robotic Box Arm" --menu "Selecciona el modo de ejecucion:" 15 65 2 \
        "1" "Con Jupyter Lab (puerto $JUPYTER_PORT)" \
        "2" "Sin Jupyter Lab (modo estandar)" \
        3>&1 1>&2 2>&3)

    if [ $? -ne 0 ]; then
        echo "Operacion cancelada por el usuario."
        exit 0
    fi
else
    echo "1) Con Jupyter Lab (puerto $JUPYTER_PORT)"
    echo "2) Sin Jupyter Lab (modo estandar)"
    read -p "Opcion [1-2]: " modo
fi

echo ""
echo "Creando contenedor '$CONTAINER_NAME'..."

# --network=host expone TODOS los puertos sin enumerarlos, asi que cambiar
# "port" en configs/pipeline.yaml no obliga a tocar este script.
COMMON_ARGS=(
    --runtime nvidia --privileged -d --ipc=host
    --device /dev/bus/usb:/dev/bus/usb
    "${DEVICE_ARGS[@]}"
    --device-cgroup-rule='c 81:* rmw'
    --network=host
    --volume /run/udev:/run/udev:ro
    --volume /tmp/argus_socket:/tmp/argus_socket
    --volume "$REPO_ROOT:$MOUNT_POINT"
    --workdir "$MOUNT_POINT"
    --name "$CONTAINER_NAME"
    --restart unless-stopped
)

if [[ "$modo" == "1" ]]; then
    docker run "${COMMON_ARGS[@]}" "$IMAGE_NAME" \
      /bin/bash -c "jupyter lab --ip 0.0.0.0 --port $JUPYTER_PORT --allow-root --no-browser --ServerApp.token '$JUPYTER_TOKEN' --ServerApp.password='' &> /var/log/jupyter.log & tail -f /var/log/jupyter.log"

    echo ""
    echo "Contenedor creado con Jupyter Lab"
    echo "JupyterLab: http://$(hostname -I | awk '{print $1}'):$JUPYTER_PORT"
    echo "Token: $JUPYTER_TOKEN"
else
    docker run "${COMMON_ARGS[@]}" "$IMAGE_NAME"

    echo ""
    echo "Contenedor creado en modo estandar"
fi

echo ""
echo "  $REPO_ROOT  ->  $MOUNT_POINT"
echo "  camaras: ${#DEVICE_ARGS[@]} nodo(s) /dev/video*"
echo ""
echo "Comandos utiles:"
echo "  docker exec -ti $CONTAINER_NAME bash"
echo "  docker exec -ti $CONTAINER_NAME python3 main.py"
echo "  docker logs -f $CONTAINER_NAME"
echo "  docker restart $CONTAINER_NAME"
echo "  docker stop $CONTAINER_NAME"
echo "  docker rm -f $CONTAINER_NAME"
