"""Shared helpers for the CNM YOLO training notebooks (Colab + Drive).

Used by:
- a01_coin_box_detection_train.ipynb
- a02_robotic_arm_detection_train.ipynb
"""

import logging
import os
import shutil
from datetime import datetime


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def download_roboflow_dataset(api_key, workspace, project_name, version_num, fmt="yolo26"):
    from roboflow import Roboflow

    logging.info("Descargando dataset desde Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download(fmt)
    logging.info(f"Dataset descargado en: {dataset.location}")
    return dataset


def validate_data_yaml(dataset_location):
    data_yaml = os.path.join(dataset_location, "data.yaml")

    if not os.path.exists(data_yaml):
        logging.error(f"No se encontro data.yaml en: {dataset_location}")
        for f in os.listdir(dataset_location):
            logging.info(f" - {f}")
        raise FileNotFoundError("data.yaml no encontrado")

    logging.info(f"Usando data.yaml: {data_yaml}")
    with open(data_yaml, "r") as f:
        logging.info(f"\n{f.read()}")

    return data_yaml


def find_best_checkpoint(base_path, run_name):
    runs_detect = os.path.join(base_path, "runs", "detect")
    best_pt_path = None

    if os.path.exists(runs_detect):
        for folder in sorted(os.listdir(runs_detect), reverse=True):
            if folder.startswith(run_name):
                candidate = os.path.join(runs_detect, folder, "weights", "best.pt")
                if os.path.exists(candidate):
                    best_pt_path = candidate
                    logging.info(f"Modelo encontrado: {best_pt_path}")
                    break

    if best_pt_path is None:
        raise FileNotFoundError(f"No se encontro best.pt para el run '{run_name}' en {runs_detect}")

    return best_pt_path


def save_model_with_backup(best_pt_path, model_dir, model_name, class_names):
    """Copia best.pt a model_dir/model_name.pt, respaldando la version previa
    (si existe) con un timestamp, y escribe classes.txt junto al peso."""
    os.makedirs(model_dir, exist_ok=True)
    dest_path = os.path.join(model_dir, f"{model_name}.pt")

    if os.path.exists(dest_path):
        timestamp = os.path.getctime(dest_path)
        fecha = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        backup = f"{model_name}_{fecha}.pt"
        shutil.move(dest_path, os.path.join(model_dir, backup))
        logging.info(f"Backup creado: {backup}")

    shutil.copy2(best_pt_path, dest_path)
    logging.info(f"Modelo guardado en: {dest_path}")

    classes_path = os.path.join(model_dir, "classes.txt")
    with open(classes_path, "w", encoding="utf-8") as f:
        for idx, name in class_names.items() if isinstance(class_names, dict) else enumerate(class_names):
            f.write(f"{idx}: {name}\n")
    logging.info(f"Clases guardadas en: {classes_path}")

    return dest_path
