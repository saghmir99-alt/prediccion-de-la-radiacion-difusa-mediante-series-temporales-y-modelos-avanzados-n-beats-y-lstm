"""
Configuración persistente del entorno del TFM.

Este módulo no carga datasets, modelos ni resultados.
Tampoco accede al conjunto de test.
"""

from pathlib import Path

PROJECT_FOLDER_ID = '1RAdZp9VKXm_-GrwgchKEL_YY-olFP0oz'
NOTEBOOK_ID = '1lOtlZn1Z3DJrQzt9pYhUJnYYkdiap15M'

PROJECT_ROOT = Path('/content/drive/MyDrive/TFM Aghmir/TFM/02. Modelado Multivariante/03. Modelado')
NOTEBOOK_PATH = Path('/content/drive/MyDrive/Colab Notebooks/Modelado.ipynb')

PATHS = {
    "root": PROJECT_ROOT,
    "data": PROJECT_ROOT / "datos",
    "models_promoted": PROJECT_ROOT / "modelos",
    "results": PROJECT_ROOT / "resultados",
    "runs": PROJECT_ROOT / "resultados" / "ejecuciones",
    "comparisons": PROJECT_ROOT / "comparativas",
    "exports": PROJECT_ROOT / "exportaciones",
    "documentation": PROJECT_ROOT / "documentacion",
    "inventories": PROJECT_ROOT / "documentacion" / "inventarios",
    "notebooks": NOTEBOOK_PATH.parent,
    "notebook_active": NOTEBOOK_PATH,
    "temp_runs": Path("/content/tfm_runs"),
}

PERSISTENT_KEYS = (
    "root",
    "data",
    "models_promoted",
    "results",
    "runs",
    "comparisons",
    "exports",
    "documentation",
    "inventories",
    "notebooks",
)


def initialize_environment(create_temp=True):
    """
    Valida las rutas persistentes y reconstruye el área temporal.

    Debe llamarse después de montar Google Drive.
    No crea ni modifica archivos persistentes.
    """
    missing = [
        str(PATHS[key])
        for key in PERSISTENT_KEYS
        if not PATHS[key].is_dir()
    ]

    if not NOTEBOOK_PATH.is_file():
        missing.append(str(NOTEBOOK_PATH))

    if missing:
        raise FileNotFoundError(
            "Faltan rutas persistentes o Google Drive no está montado:\n"
            + "\n".join(missing)
        )

    if create_temp:
        PATHS["temp_runs"].mkdir(
            parents=True,
            exist_ok=True,
        )

    return PATHS.copy()


def environment_status():
    """
    Devuelve un resumen sin leer el contenido de ningún archivo.
    """
    return {
        "project_root_available": PROJECT_ROOT.is_dir(),
        "notebook_available": NOTEBOOK_PATH.is_file(),
        "persistent_paths_available": {
            key: PATHS[key].is_dir()
            for key in PERSISTENT_KEYS
        },
        "temporary_area_available": (
            PATHS["temp_runs"].is_dir()
        ),
        "test_accessed": False,
    }
