import os
import numpy as np
import pandas as pd
import joblib
import fastf1

# Parámetros del circuito y carrera
YEAR = 2025
GP = "Monza"
DRIVER = "VER"  # Para obtener datos de referencia
CANT_VUELTAS = 53  # Número de vueltas en la carrera (ajustar según circuito)

# Parámetros del algoritmo evolutivo
POP_SIZE = 100
NGEN = 100
PMUT = 0.9

# Tiempo de pit stop 
PIT_STOP_TIME = 25.0 

# NOTA: Se eliminaron los límites explícitos de vueltas por compuesto.
# El modelo de predicción aprenderá implícitamente que mantener un neumático
# por mucho tiempo resulta en tiempos de vuelta peores debido a la degradación.

# Cargar modelo
# El modelo se guarda en utils/XGBoost/models/, así que buscamos ahí
base_dir = os.path.dirname(os.path.dirname(__file__))  # Sube a 'utils'
models_dir = os.path.join(base_dir, 'XGBoost', 'models')
MODEL_FILENAME = f"model_{YEAR}_gp{GP}.pkl"
model_path = os.path.join(models_dir, MODEL_FILENAME)

if not os.path.exists(model_path):
    raise SystemExit(f"[X] Modelo no encontrado en: {model_path}")

print(f"\n[OK] Cargando modelo: {MODEL_FILENAME}")
model_data = joblib.load(model_path)
model = model_data['model']
features = model_data['features']
metadata = model_data['metadata']

print(f"  Circuito: {metadata['circuit']}")
print(f"  R² Validación: {metadata['r2_val']:.3f}")
print(f"  MAE: {metadata['mae_val']:.3f}s")

# Configurar cache de FastF1
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Cargar datos de referencia de las prácticas/qualifying
print(f"\n[OK] Cargando datos de referencia de {YEAR} {GP}...")
reference_conditions = {}

# Intentar cargar sesiones de práctica y qualifying (CON telemetría)
sessions_to_try = ["Q", "FP3", "FP2", "FP1"]
loaded_session = None
loaded_session_type = None

for session_type in sessions_to_try:
    try:
        print(f"  Intentando {session_type}...", end=" ")
        session = fastf1.get_session(YEAR, GP, session_type)
        # Cargar CON telemetría para poder obtener datos de velocidad
        session.load(telemetry=True, weather=True, messages=False)
        loaded_session = session
        loaded_session_type = session_type
        print("OK")
        break
    except Exception as e:
        print(f"SKIP")

if loaded_session is None:
    print("\n[X] No se pudo cargar ninguna sesión de práctica/qualifying")
    print(f"[!] Verifica que existen datos para {YEAR} {GP}")
    raise SystemExit("[X] Ejecución cancelada")

print(f"  Sesión cargada: {loaded_session_type}")

# Extraer condiciones promedio de la sesión
if hasattr(loaded_session, "weather_data") and loaded_session.weather_data is not None and len(loaded_session.weather_data) > 0:
    reference_conditions["TrackTemp"] = loaded_session.weather_data["TrackTemp"].mean()
    reference_conditions["AirTemp"] = loaded_session.weather_data["AirTemp"].mean()
    reference_conditions["Humidity"] = loaded_session.weather_data["Humidity"].mean()
else:
    # Valores por defecto si no hay datos meteorológicos
    reference_conditions["TrackTemp"] = 30.0
    reference_conditions["AirTemp"] = 25.0
    reference_conditions["Humidity"] = 50.0

# Obtener telemetría promedio del piloto de referencia
print(f"\n[OK] Obteniendo telemetría de referencia de {DRIVER}...")

telemetry_obtained = False
error_message = ""

try:
    driver_laps = loaded_session.laps.pick_driver(DRIVER).pick_quicklaps()
    
    if len(driver_laps) == 0:
        error_message = f"No se encontraron vueltas rápidas para {DRIVER}"
        raise Exception(error_message)
    
    # Calcular promedios de telemetría
    speed_values = []
    throttle_values = []
    
    print(f"  Procesando {min(5, len(driver_laps))} vueltas...", end=" ")
    
    for idx in driver_laps.index[:5]:  # Solo primeras 5 vueltas rápidas
        try:
            telem = driver_laps.loc[idx].get_car_data()
            if telem is not None and len(telem) > 0:
                if 'Speed' in telem.columns and 'Throttle' in telem.columns:
                    max_speed = telem['Speed'].max()
                    avg_throttle = telem['Throttle'].mean()
                    
                    # Validar que los datos sean razonables
                    if max_speed > 0 and not np.isnan(max_speed):
                        speed_values.append(max_speed)
                    if avg_throttle >= 0 and not np.isnan(avg_throttle):
                        throttle_values.append(avg_throttle)
        except Exception as e:
            continue
    
    if len(speed_values) == 0 or len(throttle_values) == 0:
        error_message = f"No se pudo extraer telemetría válida de las vueltas de {DRIVER}"
        raise Exception(error_message)
    
    # Asignar valores promedio
    reference_conditions["MaxSpeed"] = np.mean(speed_values)
    reference_conditions["AvgSpeed"] = np.mean(speed_values) * 0.75  # Estimación conservadora
    reference_conditions["AvgThrottle"] = np.mean(throttle_values)
    
    print(f"OK ({len(speed_values)} vueltas procesadas)")
    telemetry_obtained = True
    
except Exception as e:
    print(f"ERROR")
    if error_message:
        print(f"\n[X] {error_message}")
    else:
        print(f"\n[X] Error al obtener telemetría: {str(e)}")
    print(f"\n[!] El algoritmo requiere datos de telemetría reales para funcionar correctamente.")
    print(f"[!] Verifica que:")
    print(f"    - El piloto '{DRIVER}' participó en la sesión")
    print(f"    - Hay datos de telemetría disponibles")
    print(f"    - La sesión se cargó correctamente")
    raise SystemExit("\n[X] Ejecución cancelada: No hay telemetría válida")

# Obtener equipo del piloto
try:
    team = loaded_session.results[loaded_session.results["Abbreviation"] == DRIVER]["TeamName"].iloc[0]
    reference_conditions["Team"] = team
except:
    reference_conditions["Team"] = "Red Bull Racing"

print(f"\n[OK] Condiciones de referencia:")
print(f"  Sesión: {loaded_session_type}")
print(f"  Temperatura pista: {reference_conditions['TrackTemp']:.1f}°C")
print(f"  Temperatura aire: {reference_conditions['AirTemp']:.1f}°C")
print(f"  Humedad: {reference_conditions['Humidity']:.1f}%")
print(f"  Velocidad máxima: {reference_conditions['MaxSpeed']:.1f} km/h")
print(f"  Velocidad promedio: {reference_conditions['AvgSpeed']:.1f} km/h")
print(f"  Throttle promedio: {reference_conditions['AvgThrottle']:.1f}%")
print(f"  Equipo: {reference_conditions['Team']}")

print(f"\n[OK] Vueltas en carrera: {CANT_VUELTAS}")

# Validación final antes de continuar
if not telemetry_obtained:
    raise SystemExit("\n[X] No se puede continuar sin telemetría válida")

print("="*60)
print("TODOS LOS DATOS CARGADOS CORRECTAMENTE")
print("="*60 + "\n")

# ============================================================
# FUNCIONES AUXILIARES PARA PREDICCIÓN
# ============================================================

def create_features_for_lap(lap_number, compound, tyre_life, fuel_load, conditions):
    """
    Crea features para una vuelta específica, similar a create_features de model_improved_v2.py
    """
    # Mapeo de compuestos
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    compound_hardness = compound_map.get(compound, 2)
    
    features_dict = {
        # Categóricas
        "Compound": compound,
        "SessionType": "R",  # Predecimos para carrera
        "Team": conditions["Team"],
        
        # Neumáticos básicas
        "TyreLife": tyre_life,
        "TyreWearRate": tyre_life / 50.0,  # Normalizar igual que en entrenamiento
        "TyreLifeSquared": tyre_life ** 2,
        "CompoundHardness": compound_hardness,
        
        # Interacciones compuesto-edad (el modelo aprenderá la degradación naturalmente)
        "TyreLifeByCompound": tyre_life * compound_hardness,
        "RelativeTyreAge": tyre_life / compound_hardness,
        
        # Combustible
        "FuelLoad": fuel_load,
        "FuelPenalty": fuel_load * 3.0,
        
        # Temperatura
        "TrackTemp": conditions["TrackTemp"],
        "AirTemp": conditions["AirTemp"],
        "Humidity": conditions["Humidity"],
        "TempDiff": conditions["TrackTemp"] - conditions["AirTemp"],
        
        # Telemetría
        "MaxSpeed": conditions["MaxSpeed"],
        "AvgSpeed": conditions["AvgSpeed"],
        "AvgThrottle": conditions["AvgThrottle"]
    }
    
    return features_dict

def predict_lap_time(lap_number, compound, tyre_life, fuel_load, conditions, model, features):
    """
    Predice el tiempo de una vuelta usando el modelo
    """
    features_dict = create_features_for_lap(lap_number, compound, tyre_life, fuel_load, conditions)
    
    # Crear DataFrame con las features en el orden correcto
    X = pd.DataFrame([features_dict])[features]
    
    # Predecir (el modelo fue entrenado con log1p, así que usamos expm1)
    log_pred = model.predict(X)[0]
    lap_time = np.expm1(log_pred)
    
    return lap_time