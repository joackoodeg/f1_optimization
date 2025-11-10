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

def detect_outliers_iqr(df, column='LapTimeSec', factor=2.0, protect_pitstops=True, preserve_continuity=True):
    """
    Detecta y elimina outliers preservando pitstops y continuidad de degradación.
    
    Args:
        df: DataFrame con datos de vueltas
        column: Columna para detectar outliers
        factor: Factor IQR para detectar outliers
        protect_pitstops: Si True, protege vueltas con pitstop de ser eliminadas
        preserve_continuity: Si True, preserva continuidad de degradación por driver
    """
    if len(df) == 0:
        return df
    
    # Detectar pitstops antes de eliminar outliers
    # Pitstop se detecta cuando TyreLife == 0 o cambia el compuesto
    if protect_pitstops and 'TyreLife' in df.columns and 'Compound' in df.columns:
        # Detectar pitstops: TyreLife == 0
        pitstop_mask = (df['TyreLife'] == 0)
        
        # También detectar cambios de compuesto (puede indicar pitstop)
        if 'LapNumber' in df.columns and 'Driver' in df.columns:
            # Ordenar por driver y número de vuelta para detectar cambios de compuesto
            df_sorted_pitstop = df.sort_values(['Driver', 'LapNumber']).copy()
            # Detectar cambios de compuesto dentro del mismo driver
            df_sorted_pitstop['CompoundChange'] = (df_sorted_pitstop.groupby('Driver')['Compound'].shift() != df_sorted_pitstop['Compound'])
            compound_change_mask_sorted = df_sorted_pitstop['CompoundChange'].fillna(False)
            # Restaurar índice original
            compound_change_mask = compound_change_mask_sorted.reindex(df.index).fillna(False)
            pitstop_mask = pitstop_mask | compound_change_mask
        
        n_pitstops = pitstop_mask.sum()
    else:
        pitstop_mask = pd.Series([False] * len(df), index=df.index)
        n_pitstops = 0
    
    # Calcular límites IQR
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    
    # Detectar outliers
    outlier_mask = (df[column] < lower) | (df[column] > upper)
    
    # Proteger pitstops: no eliminar vueltas con pitstop aunque sean outliers
    if protect_pitstops:
        outlier_mask = outlier_mask & ~pitstop_mask
    
    # Preservar continuidad de degradación por driver
    if preserve_continuity and 'Driver' in df.columns and 'LapNumber' in df.columns and 'TyreLife' in df.columns:
        # Ordenar por driver y número de vuelta
        df_sorted = df.sort_values(['Driver', 'LapNumber']).copy()
        
        # Crear máscara de outliers en el orden ordenado
        outlier_mask_sorted = outlier_mask.reindex(df_sorted.index)
        
        # Para cada driver, verificar continuidad de TyreLife
        protected_indices = set()
        
        for driver in df_sorted['Driver'].unique():
            driver_data = df_sorted[df_sorted['Driver'] == driver].copy()
            if len(driver_data) < 2:
                continue
            
            # Verificar secuencia de TyreLife
            tyre_life_values = driver_data['TyreLife'].values
            lap_numbers = driver_data['LapNumber'].values
            driver_indices = driver_data.index.values
            
            # Proteger vueltas que mantienen continuidad
            for i in range(1, len(tyre_life_values)):
                prev_tyre_life = tyre_life_values[i-1]
                curr_tyre_life = tyre_life_values[i]
                prev_lap = lap_numbers[i-1]
                curr_lap = lap_numbers[i]
                
                # Verificar si la secuencia es continua
                is_continuous = (
                    curr_tyre_life == 0 or  # Pitstop (reseteo)
                    curr_tyre_life == prev_tyre_life + 1 or  # Degradación normal
                    (curr_lap == prev_lap + 1 and curr_tyre_life >= prev_tyre_life)  # Vuelta siguiente con degradación válida
                )
                
                # Si la secuencia es continua y alguna vuelta está marcada como outlier, proteger ambas
                if is_continuous:
                    prev_idx = driver_indices[i-1]
                    curr_idx = driver_indices[i]
                    if outlier_mask_sorted.loc[prev_idx] or outlier_mask_sorted.loc[curr_idx]:
                        protected_indices.add(prev_idx)
                        protected_indices.add(curr_idx)
        
        # No eliminar vueltas protegidas
        if protected_indices:
            protected_mask = pd.Series([idx in protected_indices for idx in df.index], index=df.index)
            outlier_mask = outlier_mask & ~protected_mask
    
    # Aplicar máscara
    mask = ~outlier_mask
    n_outliers = outlier_mask.sum()
    
    if n_outliers > 0:
        protected_count = n_pitstops if protect_pitstops else 0
        print(f"    [OUTLIERS] {n_outliers} ({n_outliers/len(df)*100:.1f}%)", end="")
        if protected_count > 0:
            print(f" | [PROTECTED] {protected_count} pitstops preservados", end="")
        print()
    
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
            
            # Añadir columna Driver para preservar continuidad por driver
            laps["Driver"] = driver
            
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
            
            # Ordenar por número de vuelta para preservar continuidad
            if len(laps) > 0 and 'LapNumber' in laps.columns:
                laps = laps.sort_values('LapNumber').reset_index(drop=True)
            
            # Outliers: proteger pitstops y preservar continuidad de degradación
            if len(laps) > 10:
                laps = detect_outliers_iqr(
                    laps, 
                    column='LapTimeSec', 
                    factor=2.0, 
                    protect_pitstops=True, 
                    preserve_continuity=True
                )
            
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
    # Compuesto numérico (solo para calcular TyreLifeByCompound, no se incluye en features)
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    df["CompoundHardness"] = df["Compound"].map(compound_map)
    
    # Básicas de neumáticos
    # TyreWearRate ELIMINADA: correlación perfecta (1.0) con TyreLife
    
    # Degradación cuadrática
    df["TyreLifeSquared"] = df["TyreLife"] ** 2
    
    # Degradación cúbica (para capturar mejor la degradación no lineal, especialmente en SOFT)
    df["TyreLifeCubed"] = df["TyreLife"] ** 3
    
    # Interacción compuesto-edad (el modelo aprenderá la degradación naturalmente)
    # Esta feature permite al modelo aprender diferentes tasas de degradación por compuesto
    df["TyreLifeByCompound"] = df["TyreLife"] * df["CompoundHardness"]
    
    # RelativeTyreAge ELIMINADA: correlacionada con TyreLife y CompoundHardness
    
    # Combustible
    df["FuelPenalty"] = df["FuelLoad"] * 3.0
    
    return df

# Solo ejecutar esto si se ejecuta directamente, no cuando se importa
if __name__ == "__main__":
    print(f"[I] Circuito: {gp} ({year})")
    target_circuit = get_circuit_name(year, gp)
    print(f"[OK] {target_circuit}\n")

