from __future__ import annotations

# -----------------------------------------------------------------------
# CNM-Robotic_Box_Arm
# Programmer : gerald
# Date       : 23-08-26
# Reason     : Trains the box detector: pulls the dataset from Roboflow
#              and trains YOLO26 on it. Separate from the inference
#              pipeline (src/boxarm) -- this does not run in production,
#              it only produces the .pt consumed by configs/pipeline.yaml.
# -----------------------------------------------------------------------
"""Trains the box detector with YOLO26 on the Roboflow dataset.

No CLI arguments on purpose: the values that matter are the constants
below -- edit them here if something needs to change.

The API key is NOT in the code: it's read from the ROBOFLOW_API_KEY
environment variable.

Usage:
    set ROBOFLOW_API_KEY=...            (or export in bash)
    python scripts/train_yolo26.py
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("train")

ROOT = Path(__file__).resolve().parent.parent

# -- Roboflow dataset ----------------------------------------------------
# Ver el docstring del modulo: la key NUNCA va hardcodeada aca, este
# archivo se trackea con git y una key en el codigo queda en el historial
# para siempre aunque se borre despues.
RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not RF_API_KEY:
    raise RuntimeError(
        "falta la variable de entorno ROBOFLOW_API_KEY -- "
        "set ROBOFLOW_API_KEY=... (o export en bash) antes de correr este script"
    )
RF_WORKSPACE = "alys-peru"
RF_PROJECT = "cnm_palletscajas"
RF_VERSION = 6
RF_FORMAT = "yolo26"

# -- Training --------------------------------------------------------------
BASE_MODEL = "yolo26m.pt"        # base weights (yolo26n/s/m/l/x)
IMGSZ = 512
EPOCHS = 150
BATCH = 8                        # what fits on a 6GB GPU (RTX 4050 laptop) at imgsz=512
PATIENCE = 20                    # epochs without improvement before early stopping
WORKERS = 0                       # dataloader workers=2 deadlocked mid-training on this machine
                                  # (WinError 1455 in a worker, main process hangs waiting for it
                                  # forever -- GPU stays at 0% util but VRAM held). 0 = load in the
                                  # main process, no multiprocessing, slower but can't deadlock.
RUN_NAME = f"cajas_yolo26m_512_v{RF_VERSION}"


def download_dataset(dest: Path) -> Path:
    """Downloads the dataset from Roboflow and returns the path to its data.yaml."""
    from roboflow import Roboflow

    logger.info("downloading %s/%s v%d from Roboflow...", RF_WORKSPACE, RF_PROJECT, RF_VERSION)
    project = Roboflow(api_key=RF_API_KEY).workspace(RF_WORKSPACE).project(RF_PROJECT)
    dataset = project.version(RF_VERSION).download(RF_FORMAT, location=str(dest))
    data_yaml = Path(dataset.location) / "data.yaml"
    logger.info("dataset at %s", data_yaml)
    return data_yaml


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    import torch
    if not torch.cuda.is_available():
        logger.error("no CUDA available in this interpreter (%s) -- training on CPU is not viable",
                     sys.executable)
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    logger.info("GPU: %s, %.1f GB VRAM", props.name, props.total_memory / 1e9)

    from ultralytics import YOLO

    # If a previous run under this exact RUN_NAME left a checkpoint, pick up
    # from there instead of starting over -- a crash/hang partway through
    # (e.g. the Windows dataloader deadlock this hit before) shouldn't cost
    # the epochs already trained. Only "last.pt" (not "best.pt") carries the
    # optimizer/scheduler state resume needs to continue correctly.
    checkpoint = ROOT / "runs" / RUN_NAME / "weights" / "last.pt"
    if checkpoint.is_file():
        logger.info("found checkpoint %s -- resuming instead of starting from scratch", checkpoint)
        model = YOLO(str(checkpoint))
        model.train(resume=True, workers=WORKERS)
    else:
        data_yaml = download_dataset(ROOT / "datasets" / RF_PROJECT)
        if not data_yaml.is_file():
            logger.error("%s does not exist", data_yaml)
            sys.exit(1)

        logger.info("training %s -- imgsz=%d epochs=%d batch=%d", BASE_MODEL, IMGSZ, EPOCHS, BATCH)
        model = YOLO(BASE_MODEL)
        model.train(
            data=str(data_yaml),
            imgsz=IMGSZ,
            epochs=EPOCHS,
            batch=BATCH,
            patience=PATIENCE,
            device=0,
            project=str(ROOT / "runs"),
            name=RUN_NAME,
            workers=WORKERS,
            # AMP cuts VRAM usage a lot; essential on a 6 GB GPU.
            amp=True,
            # The boxes sit on a pallet seen from above: mirroring or rotating
            # 90 degrees are perfectly valid views of the same layout, so
            # they're free augmentations. Scale, on the other hand, is kept
            # moderate: the bbox's APPARENT SIZE is the signal the pipeline
            # uses to infer the stacking level (see GridCounter), and
            # exaggerating it would teach the model to ignore that signal.
            fliplr=0.5,
            flipud=0.5,
            degrees=90.0,
            scale=0.3,
            mosaic=1.0,
            close_mosaic=10,
        )

    best = ROOT / "runs" / RUN_NAME / "weights" / "best.pt"
    logger.info("done. best weights: %s", best)
    logger.info("to use it, in configs/pipeline.yaml -> weights: %s",
                best.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
