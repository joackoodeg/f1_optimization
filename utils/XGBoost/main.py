import fastf1
import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Cache y directorios
cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

models_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(models_dir, exist_ok=True)

year = 2025
gp = "Monza"
historical_years = [2024, 2023, 2022] 
top_n_drivers = 25  
FORCE_RETRAIN = True
MODEL_FILENAME = f"model_{year}_gp{gp}.pkl"

def save_model(model, features, metadata, filename=MODEL_FILENAME):
    """Guarda el modelo entrenado"""
    model_path = os.path.join(models_dir, filename)
    
    model_data = {
        'model': model,
        'features': features,
        'metadata': metadata,
        'saved_at': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, model_path, compress=3)
    print(f"\n[OK] Modelo guardado en: {model_path}")
    print(f"  Tamaño: {os.path.getsize(model_path) / 1024:.2f} KB")
    return model_path

def get_circuit_name(year, gp):
    """Obtiene nombre del circuito"""
    try:
        session = fastf1.get_session(year, gp, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        return session.event['EventName']
    except:
        return None

def detect_outliers_iqr(df, column='LapTimeSec', factor=2.0):
    """Detecta y elimina outliers (más conservador: factor=2.0)"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    mask = (df[column] >= lower) & (df[column] <= upper)
    n_outliers = (~mask).sum()
    
    if n_outliers > 0:
        print(f"    [OUTLIERS] {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
    
    return df[mask]

def load_session_data(year, gp, session_type, drivers=None, circuit_name=None):
    """Carga datos de sesión (versión simplificada)"""
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load()
    except:
        return None, None
    
    if circuit_name is None:
        circuit_name = session.event['EventName']
    
    if drivers is None:
        drivers = session.results["Abbreviation"].iloc[:top_n_drivers].tolist()
    elif isinstance(drivers, str):
        drivers = [drivers]
    
    all_laps = []
    
    for driver in drivers:
        try:
            # Solo vueltas rápidas
            laps = session.laps.pick_driver(driver).pick_quicklaps()
            if len(laps) == 0:
                continue
            
            # Solo neumáticos secos
            laps = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
            if len(laps) == 0:
                continue
            
            # Meteorología
            if hasattr(session, "weather_data") and session.weather_data is not None and len(session.weather_data) > 0:
                laps["TrackTemp"] = session.weather_data["TrackTemp"].mean()
                laps["AirTemp"] = session.weather_data["AirTemp"].mean()
                laps["Humidity"] = session.weather_data["Humidity"].mean()
            else:
                laps["TrackTemp"] = 30.0
                laps["AirTemp"] = 25.0
                laps["Humidity"] = 50.0
            
            # Team
            try:
                team = session.results[session.results["Abbreviation"] == driver]["TeamName"].iloc[0]
                laps["Team"] = team
            except:
                laps["Team"] = "Unknown"
            
            # Telemetría básica
            print(f"    {driver}...", end=" ")
            telemetry_list = []
            
            for idx in laps.index:
                try:
                    telem = laps.loc[idx].get_car_data()
                    if telem is not None and len(telem) > 0:
                        telemetry_list.append({
                            'MaxSpeed': telem['Speed'].max(),
                            'AvgSpeed': telem['Speed'].mean(),
                            'AvgThrottle': telem['Throttle'].mean()
                        })
                    else:
                        telemetry_list.append({
                            'MaxSpeed': 300.0,
                            'AvgSpeed': 200.0,
                            'AvgThrottle': 70.0
                        })
                except:
                    telemetry_list.append({
                        'MaxSpeed': 300.0,
                        'AvgSpeed': 200.0,
                        'AvgThrottle': 70.0
                    })
            
            telem_df = pd.DataFrame(telemetry_list, index=laps.index)
            laps = pd.concat([laps.reset_index(drop=True), telem_df.reset_index(drop=True)], axis=1)
            
            # Features básicas
            required = [
                "LapTime", "LapNumber", "TyreLife", "Compound", 
                "TrackTemp", "AirTemp", "Humidity", "TrackStatus", "Team",
                "MaxSpeed", "AvgSpeed", "AvgThrottle"
            ]
            
            laps = laps[required].dropna()
            if len(laps) == 0:
                continue
            
            # Flags y conversiones
            laps["IsSC"] = laps["TrackStatus"].apply(lambda x: 1 if "4" in str(x) or "5" in str(x) else 0)
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps["SessionType"] = session_type
            
            # Combustible estimado
            if session_type == "R":
                total = session.laps["LapNumber"].max()
                laps["FuelLoad"] = 1.0 - (laps["LapNumber"] / total)
            elif session_type.startswith("FP"):
                laps["FuelLoad"] = 0.25
            elif session_type in ["Q", "SQ"]:
                laps["FuelLoad"] = 0.05
            else:
                laps["FuelLoad"] = 0.5
            
            # Filtros
            laps = laps[(laps["IsSC"] == 0) & (laps["LapTimeSec"].between(60, 120))]
            
            # Outliers
            if len(laps) > 10:
                laps = detect_outliers_iqr(laps, 'LapTimeSec', factor=2.0)
            
            all_laps.append(laps)
            
        except:
            continue
    
    if all_laps:
        print()
        combined = pd.concat(all_laps, ignore_index=True)
        return combined, drivers
    else:
        print()
        return None, None

def create_features(df):
    """
    Features MÍNIMAS y ROBUSTAS
    Solo las que realmente importan
    """
    # Compuesto numérico
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    df["CompoundHardness"] = df["Compound"].map(compound_map)
    
    # Básicas de neumáticos
    # TyreWearRate: normalizar por edad máxima razonable (más representativo)
    df["TyreWearRate"] = df["TyreLife"] / 50.0  # Normalizar a ~50 vueltas máximo
    
    # Degradación cuadrática
    df["TyreLifeSquared"] = df["TyreLife"] ** 2
    
    # Interacción compuesto-edad (el modelo aprenderá que SOFT con alta edad = peor)
    # Esta es una feature natural que el modelo puede aprender, no una penalización explícita
    df["TyreLifeByCompound"] = df["TyreLife"] * df["CompoundHardness"]
    
    # Degradación relativa: edad del neumático relativa a su dureza
    # Los neumáticos blandos (1) con alta edad se degradan más rápido
    df["RelativeTyreAge"] = df["TyreLife"] / df["CompoundHardness"]
    
    # Combustible
    df["FuelPenalty"] = df["FuelLoad"] * 3.0
    
    # Temperatura
    df["TempDiff"] = df["TrackTemp"] - df["AirTemp"]
    
    return df

# Solo ejecutar esto si se ejecuta directamente, no cuando se importa
if __name__ == "__main__":
    print(f"[I] Circuito: {gp} ({year})")
    target_circuit = get_circuit_name(year, gp)
    print(f"[OK] {target_circuit}\n")

