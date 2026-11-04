
"""
Gestor reproducible de ejecuciones del TFM.

Los entrenamientos se desarrollan en /content y se archivan
posteriormente en Google Drive sin sobrescribir ejecuciones previas.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import platform
import re
import shutil
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from entorno_tfm import PATHS


RUN_SUBDIRECTORIES = (
    "configuracion",
    "modelo",
    "escaladores",
    "predicciones",
    "metricas",
    "tablas",
    "tiempos",
    "logs",
)


def _utc_now():
    return datetime.now(timezone.utc)


def _slug(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    value = value.strip("-._")

    if not value:
        raise ValueError("El identificador no puede quedar vacío.")

    return value


def _json_default(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    if hasattr(value, "tolist"):
        return value.tolist()

    raise TypeError(
        f"Objeto no serializable en JSON: {type(value).__name__}"
    )


def _safe_destination(run_directory, relative_path):
    run_directory = Path(run_directory).resolve()
    destination = (run_directory / relative_path).resolve()

    if destination != run_directory and run_directory not in destination.parents:
        raise ValueError(
            "La ruta solicitada queda fuera de la ejecución."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _write_json_exclusive(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )

    return path


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for chunk in iter(
            lambda: file.read(chunk_size),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def start_run(
    architecture,
    variant,
    configuration,
    seeds,
    selected_epoch=None,
    split_policy=None,
    notes=None,
):
    """
    Crea una ejecución nueva en /content.

    No escribe todavía resultados de entrenamiento en Google Drive.
    """
    architecture_slug = _slug(architecture)
    variant_slug = _slug(variant)

    created_at = _utc_now()
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")

    run_id = (
        f"{timestamp}__{architecture_slug}__{variant_slug}"
    )

    run_directory = PATHS["temp_runs"] / run_id

    # Garantiza que nunca se reutiliza una ejecución previa.
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    for subdirectory in RUN_SUBDIRECTORIES:
        (run_directory / subdirectory).mkdir(
            parents=False,
            exist_ok=False,
        )

    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "initialized",
        "created_at_utc": created_at.isoformat(),
        "architecture": architecture,
        "variant": variant,
        "selected_epoch": selected_epoch,
        "split_policy": split_policy,
        "notes": notes,
        "storage": {
            "temporary_directory": str(run_directory),
            "archive_root": str(PATHS["runs"]),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "test_accessed": False,
    }

    _write_json_exclusive(
        run_directory / "configuracion" / "metadata_ejecucion.json",
        metadata,
    )

    _write_json_exclusive(
        run_directory / "configuracion" / "configuracion.json",
        configuration,
    )

    _write_json_exclusive(
        run_directory / "configuracion" / "semillas.json",
        seeds,
    )

    return {
        "run_id": run_id,
        "run_directory": run_directory,
        "paths": {
            name: run_directory / name
            for name in RUN_SUBDIRECTORIES
        },
    }


def save_json(run_directory, relative_path, data):
    destination = _safe_destination(
        run_directory,
        relative_path,
    )
    return _write_json_exclusive(destination, data)


def save_text(run_directory, relative_path, text):
    destination = _safe_destination(
        run_directory,
        relative_path,
    )

    with destination.open("x", encoding="utf-8") as file:
        file.write(str(text))

    return destination


def save_pickle(run_directory, relative_path, obj):
    """
    Adecuado para escaladores y objetos auxiliares controlados.
    """
    destination = _safe_destination(
        run_directory,
        relative_path,
    )

    with destination.open("xb") as file:
        pickle.dump(
            obj,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return destination


def save_table(
    run_directory,
    relative_path,
    dataframe,
    index=False,
):
    """
    Guarda DataFrames en CSV o Parquet según la extensión.
    """
    destination = _safe_destination(
        run_directory,
        relative_path,
    )

    suffix = destination.suffix.lower()

    if suffix == ".csv":
        dataframe.to_csv(
            destination,
            index=index,
            encoding="utf-8",
        )

    elif suffix == ".parquet":
        dataframe.to_parquet(
            destination,
            index=index,
        )

    else:
        raise ValueError(
            "Las tablas deben guardarse como .csv o .parquet."
        )

    return destination


def record_timing(
    run_directory,
    phase,
    seconds,
    extra=None,
):
    """
    Registra una fase: selección, reentrenamiento, inferencia, etc.
    """
    phase_slug = _slug(phase)

    payload = {
        "phase": phase,
        "seconds": float(seconds),
        "recorded_at_utc": _utc_now().isoformat(),
        "extra": extra or {},
    }

    return save_json(
        run_directory,
        f"tiempos/{phase_slug}.json",
        payload,
    )


def save_model_information(
    run_directory,
    architecture,
    hyperparameters,
    selected_epoch,
    number_parameters=None,
    trainable_parameters=None,
    estimated_parameter_memory_bytes=None,
    checkpoint_path=None,
    extra=None,
):
    """
    Registra arquitectura, hiperparámetros, época, parámetros,
    memoria estimada y tamaño del checkpoint.
    """
    checkpoint_size_bytes = None
    checkpoint_relative_path = None

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint_size_bytes = checkpoint_path.stat().st_size

        try:
            checkpoint_relative_path = str(
                checkpoint_path.relative_to(
                    Path(run_directory)
                )
            )
        except ValueError:
            checkpoint_relative_path = str(checkpoint_path)

    payload = {
        "architecture": architecture,
        "hyperparameters": hyperparameters,
        "selected_epoch": selected_epoch,
        "number_parameters": number_parameters,
        "trainable_parameters": trainable_parameters,
        "estimated_parameter_memory_bytes": (
            estimated_parameter_memory_bytes
        ),
        "checkpoint": {
            "path": checkpoint_relative_path,
            "size_bytes": checkpoint_size_bytes,
            "size_mib": (
                checkpoint_size_bytes / 1024**2
                if checkpoint_size_bytes is not None
                else None
            ),
        },
        "extra": extra or {},
    }

    return save_json(
        run_directory,
        "modelo/informacion_modelo.json",
        payload,
    )


def build_manifest(run_directory):
    """
    Genera el manifiesto de integridad de la ejecución.
    """
    run_directory = Path(run_directory)

    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)

    manifest_path = run_directory / "manifest.json"

    if manifest_path.exists():
        raise FileExistsError(
            "La ejecución ya tiene un manifiesto."
        )

    files = []

    for path in sorted(run_directory.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(
            run_directory
        ).as_posix()

        files.append({
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "schema_version": 1,
        "generated_at_utc": _utc_now().isoformat(),
        "run_id": run_directory.name,
        "file_count": len(files),
        "total_size_bytes": sum(
            item["size_bytes"] for item in files
        ),
        "files": files,
    }

    _write_json_exclusive(
        manifest_path,
        manifest,
    )

    return manifest_path, manifest


def _file_size_map(directory):
    directory = Path(directory)

    return {
        path.relative_to(directory).as_posix(): path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    }


def archive_run(
    run_directory,
    status="completed",
    summary=None,
):
    """
    Archiva una ejecución desde /content a Google Drive.

    La copia se realiza primero en una ruta temporal de Drive.
    Solo después de verificar archivos y tamaños se renombra
    a su nombre definitivo. Nunca sobrescribe ejecuciones.
    """
    run_directory = Path(run_directory)

    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)

    if PATHS["temp_runs"].resolve() not in run_directory.resolve().parents:
        raise ValueError(
            "Solo se archivan ejecuciones creadas dentro de "
            "/content/tfm_runs."
        )

    run_id = run_directory.name
    final_destination = PATHS["runs"] / run_id
    copying_destination = PATHS["runs"] / f"{run_id}__copiando"

    if final_destination.exists():
        raise FileExistsError(
            f"La ejecución ya está archivada: {final_destination}"
        )

    if copying_destination.exists():
        raise FileExistsError(
            "Existe una copia incompleta anterior que debe revisarse: "
            f"{copying_destination}"
        )

    finalization = {
        "run_id": run_id,
        "status": status,
        "archived_at_utc": _utc_now().isoformat(),
        "summary": summary or {},
        "test_accessed": False,
    }

    _write_json_exclusive(
        run_directory / "finalizacion.json",
        finalization,
    )

    manifest_path, manifest = build_manifest(
        run_directory
    )

    shutil.copytree(
        run_directory,
        copying_destination,
    )

    source_sizes = _file_size_map(run_directory)
    copied_sizes = _file_size_map(copying_destination)

    if source_sizes != copied_sizes:
        raise RuntimeError(
            "La verificación por rutas y tamaños ha fallado. "
            f"La copia incompleta permanece en: {copying_destination}"
        )

    copying_destination.rename(final_destination)

    return {
        "run_id": run_id,
        "archive_directory": final_destination,
        "manifest": final_destination / manifest_path.name,
        "file_count": manifest["file_count"] + 1,
        "total_size_bytes_before_manifest": (
            manifest["total_size_bytes"]
        ),
        "verified": True,
    }
