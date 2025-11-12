import os
import numpy as np
import pandas as pd
import joblib
import fastf1

# Parámetros del circuito y carrera
YEAR = 2025
GP = "Monza"
CANT_VUELTAS = 53  # Número de vueltas en la carrera

# Parámetros del algoritmo evolutivo
POP_SIZE = 200
NGEN = 1000
PMUT = 0.90

# Tiempo de pit stop 
PIT_STOP_TIME = 25.0 

# Cargar modelo
# El modelo se guarda en utils/XGBoost/models/, así que buscamos ahí
base_dir = os.path.dirname(os.path.dirname(__file__)) 
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
if metadata.get('r2_val') is not None:
    print(f"  R² Validación: {metadata['r2_val']:.3f}")
    print(f"  MAE Validación: {metadata['mae_val']:.3f}s")
else:
    r2_train = metadata.get('r2_train')
    mae_train = metadata.get('mae_train')
    if r2_train is not None:
        print(f"  R² Entrenamiento: {r2_train:.3f}")
    else:
        print(f"  R² Entrenamiento: N/A")
    if mae_train is not None:
        print(f"  MAE Entrenamiento: {mae_train:.3f}s")
    else:
        print(f"  MAE Entrenamiento: N/A")
    print(f"  [NOTA] Modelo entrenado sin conjunto de validación")

# Configurar cache de FastF1
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Cargar datos de referencia de las prácticas/qualifying (solo datos meteorológicos)
print(f"\n[OK] Cargando datos de referencia de {YEAR} {GP}...")
reference_conditions = {}

# Intentar cargar sesiones de práctica y qualifying (solo datos meteorológicos, sin telemetría)
sessions_to_try = ["Q", "FP3", "FP2", "FP1"]
loaded_session = None
loaded_session_type = None

for session_type in sessions_to_try:
    try:
        print(f"  Intentando {session_type}...", end=" ")
        session = fastf1.get_session(YEAR, GP, session_type)
        # Cargar solo datos meteorológicos (sin telemetría, ya no se necesita)
        session.load(telemetry=False, weather=True, messages=False)
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

# Extraer condiciones promedio de la sesión (solo datos meteorológicos)
if hasattr(loaded_session, "weather_data") and loaded_session.weather_data is not None and len(loaded_session.weather_data) > 0:
    reference_conditions["TrackTemp"] = loaded_session.weather_data["TrackTemp"].mean()
    reference_conditions["AirTemp"] = loaded_session.weather_data["AirTemp"].mean()
    reference_conditions["Humidity"] = loaded_session.weather_data["Humidity"].mean()
else:
    # Valores por defecto si no hay datos meteorológicos
    reference_conditions["TrackTemp"] = 30.0
    reference_conditions["AirTemp"] = 25.0
    reference_conditions["Humidity"] = 50.0

print(f"\n[OK] Condiciones de referencia:")
print(f"  Sesión: {loaded_session_type}")
print(f"  Temperatura pista: {reference_conditions['TrackTemp']:.1f}°C")
print(f"  Temperatura aire: {reference_conditions['AirTemp']:.1f}°C")
print(f"  Humedad: {reference_conditions['Humidity']:.1f}%")

print(f"\n[OK] Vueltas en carrera: {CANT_VUELTAS}")

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
        # Neumáticos básicas
        "TyreLife": tyre_life,
        #"TyreLifeSquared": tyre_life ** 2,
        #"TyreLifeCubed": tyre_life ** 3,
        
        # Interacciones compuesto-edad (el modelo aprenderá la degradación naturalmente)
        # Esta feature permite al modelo aprender diferentes tasas de degradación por compuesto
        "TyreLifeByCompound": tyre_life * compound_hardness,
        
        # Combustible
        "FuelLoad": fuel_load,
        "FuelPenalty": fuel_load * 3.0,
        
        # Temperatura
        #"TrackTemp": conditions["TrackTemp"],
        #"AirTemp": conditions["AirTemp"],
        #"Humidity": conditions["Humidity"],
    }
    
    return features_dict

def predict_lap_time(lap_number, compound, tyre_life, fuel_load, conditions, model, features):
    """
    Predice el tiempo de una vuelta usando el modelo
    """
    features_dict = create_features_for_lap(lap_number, compound, tyre_life, fuel_load, conditions)
    
    # Crear DataFrame con las features en el orden correcto
    X = pd.DataFrame([features_dict])[features]
    
    # Verificar que todas las features estén presentes
    missing_features = set(features) - set(X.columns)
    if missing_features:
        raise ValueError(f"Features faltantes: {missing_features}")
    
    # Verificar que no haya NaN
    if X.isna().any().any():
        # Rellenar NaN con valores por defecto
        for col in X.columns:
            if X[col].isna().any():
                if col in ["Compound"]:
                    X[col] = X[col].fillna("Unknown")
                else:
                    X[col] = X[col].fillna(0.0)
    
    # DEBUG: Verificar que el modelo sea un Pipeline
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        # El modelo es un Pipeline, debería funcionar correctamente
        # Pero vamos a verificar que las features categóricas estén bien
        categorical_features = ["Compound"]
        for cat_feat in categorical_features:
            if cat_feat in X.columns:
                # Verificar que el valor sea válido
                if X[cat_feat].iloc[0] not in ["SOFT", "MEDIUM", "HARD"]:
                    # Si no es válido, usar un valor por defecto
                    if cat_feat == "Compound":
                        X[cat_feat] = "MEDIUM"
    
    # Predecir (el modelo fue entrenado con log1p, así que usamos expm1)
    # El modelo es un Pipeline, así que aplicará el preprocesamiento automáticamente
    try:
        # Verificar que el Pipeline tenga los pasos correctos
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            # Obtener el preprocesador y el regresor
            preprocessor = model.named_steps.get('pre', None)
            regressor = model.named_steps.get('reg', None)
            
            if preprocessor is not None:
                # Transformar las features con el preprocesador
                X_transformed = preprocessor.transform(X)
                
                # Verificar que las features transformadas varíen
                if X_transformed.shape[0] > 0:
                    # Predecir con el regresor directamente
                    if regressor is not None:
                        log_pred = regressor.predict(X_transformed)
                        if isinstance(log_pred, np.ndarray):
                            log_pred = log_pred[0] if len(log_pred) > 0 else log_pred
                    else:
                        # Si no hay regresor, usar el Pipeline completo
                        predictions = model.predict(X)
                        if isinstance(predictions, np.ndarray):
                            log_pred = predictions[0] if len(predictions) > 0 else predictions
                        else:
                            log_pred = predictions
                else:
                    raise ValueError("Features transformadas vacías")
            else:
                # Si no hay preprocesador, usar el Pipeline completo
                predictions = model.predict(X)
                if isinstance(predictions, np.ndarray):
                    log_pred = predictions[0] if len(predictions) > 0 else predictions
                else:
                    log_pred = predictions
        else:
            # Si no es un Pipeline, predecir directamente
            predictions = model.predict(X)
            if isinstance(predictions, np.ndarray):
                log_pred = predictions[0] if len(predictions) > 0 else predictions
            else:
                log_pred = predictions
        
        # Convertir de log a tiempo real
        lap_time = np.expm1(log_pred)
        
        # Verificar si el tiempo es válido
        if np.isnan(lap_time) or np.isinf(lap_time) or lap_time <= 0:
            raise ValueError(f"Tiempo inválido: {lap_time}")
            
    except Exception as e:
        # Si hay un error, mostrar el error y lanzar excepción
        raise RuntimeError(f"Error en predicción del modelo: {e}")
    
    return lap_time