import os
from pathlib import Path

# Raíz del proyecto (dos niveles arriba de src/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_RELEASE = "pipeline_v4_production"

# Directorios de Entrada
INPUTS_DIR = PROJECT_ROOT / "Inputs"
INFRA_DIR = INPUTS_DIR / "Infrastructure"
CACHE_INFRA_DIR = INFRA_DIR / "Cache_Optimizado"
GPS_DIR = INPUTS_DIR / "GPS User Data"
RATES_DIR = INPUTS_DIR / "Emission rates"

# Directorios de Salida
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
FINAL_DIR = OUTPUTS_DIR / "Final Outputs"

# Rutas de Archivos de Entrada
FILE_GPS_ORIGINAL = GPS_DIR / "top_20users_1_month.parquet"
FILE_GRAFO = INFRA_DIR / "monterrey_drive_network_V1.pkl"
FILE_GRAFO_WALK = INFRA_DIR / "monterrey_walk_network_EydanV1.pkl"
FILE_METRO = INFRA_DIR / "lineas_metrorrey.csv"
FILE_BUS = INFRA_DIR / "rutas_buses_ZMM_oficial.geojson"

# Rutas de Caché de Redes
FILE_CACHE_EDGES_DRIVE = CACHE_INFRA_DIR / "edges_drive.parquet"
FILE_CACHE_EDGES_WALK = CACHE_INFRA_DIR / "edges_walk.parquet"
FILE_CACHE_IG_DRIVE = CACHE_INFRA_DIR / "ig_drive_y_map.pkl"
FILE_CACHE_IG_WALK = CACHE_INFRA_DIR / "ig_walk_y_map.pkl"

# Rutas de Factores de Emisión de MOVES
FILE_MOVES = RATES_DIR / "cleaned_emission_rates_formatted_SB.parquet"
# Supuesto operativo provisional: los factores de la lookup están en g/km.
# WARNING: confirmar contra la exportación MOVES original en una revisión futura.
EMISSION_RATE_DISTANCE_UNIT = "g/km"
EMISSION_DISTANCE_UNIT = "km"
EMISSION_TOTAL_UNIT = "g"

# Clasificador modal seleccionable sin modificar el orquestador.
# Valores válidos: hybrid (oficial), random_forest (rollback), bayes.
MODAL_CLASSIFIER = os.getenv("MODAL_CLASSIFIER", "hybrid").strip().lower()
ENABLE_BAYES_FALLBACK = False

FILE_MODAL_HYBRID = GPS_DIR / "modal_classifier_hybrid_v1.pkl"
FILE_MODAL_RANDOM_FOREST = GPS_DIR / "random_forest_modal.pkl"
