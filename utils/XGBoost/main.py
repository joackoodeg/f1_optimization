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
# Reducir a años más recientes para tener condiciones más homogéneas
# Más datos no siempre es mejor si las condiciones son muy diferentes
historical_years = [2024, 2023, 2022]  # Solo 3 años más recientes
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
    """Carga datos de sesión con filtros avanzados para eliminar ruido"""
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
    
    # Estadísticas de filtrado
    filter_stats = {
        'total': 0,
        'not_quicklaps': 0,
        'wet_tyres': 0,
        'sc_vsc': 0,
        'yellow_flags': 0,
        'first_lap_stint': 0,
        'time_range': 0,
        'pit_outlap': 0,
        'outliers': 0,
        'final': 0
    }
    
    for driver in drivers:
        try:
            # Cargar todas las vueltas del driver
            laps = session.laps.pick_driver(driver)
            filter_stats['total'] += len(laps)
            
            if len(laps) == 0:
                continue
            
            # 1. FILTRO CRÍTICO: Solo quicklaps (vueltas rápidas y válidas)
            # Este filtro elimina: outlaps, inlaps, vueltas con errores, pit stops, etc.
            laps_before = len(laps)
            laps = laps.pick_quicklaps()
            filter_stats['not_quicklaps'] += (laps_before - len(laps))
            if len(laps) == 0:
                continue
            
            # 2. Solo neumáticos secos
            laps_before = len(laps)
            laps = laps[laps["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]
            filter_stats['wet_tyres'] += (laps_before - len(laps))
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
            
            # Features básicas
            print(f"    {driver}...", end=" ")
            required = [
                "LapTime", "LapNumber", "TyreLife", "Compound", 
                "TrackTemp", "AirTemp", "Humidity", "TrackStatus"
            ]
            
            laps = laps[required].dropna()
            if len(laps) == 0:
                continue
            
            # Añadir columna Driver para preservar continuidad por driver
            laps["Driver"] = driver
            
            # === FILTROS AVANZADOS PARA ELIMINAR RUIDO ===
            
            # TrackStatus códigos:
            # "1" = Track clear (verde)
            # "2" = Bandera amarilla
            # "4" = Safety Car
            # "5" = Bandera roja
            # "6" = Virtual Safety Car (VSC)
            # "7" = VSC ending
            laps_before = len(laps)
            laps["IsSC"] = laps["TrackStatus"].apply(
                lambda x: 1 if any(flag in str(x) for flag in ["4", "5", "6", "7"]) else 0
            )
            laps = laps[laps["IsSC"] == 0]
            filter_stats['sc_vsc'] += (laps_before - len(laps))
            
            # 4. Banderas amarillas (pueden indicar tráfico, accidentes, etc.)
            laps_before = len(laps)
            laps["IsYellow"] = laps["TrackStatus"].apply(lambda x: 1 if "2" in str(x) else 0)
            laps = laps[laps["IsYellow"] == 0]
            filter_stats['yellow_flags'] += (laps_before - len(laps))
            
            if len(laps) == 0:
                continue
            
            # 5. Convertir LapTime a segundos ANTES de filtros de tiempo
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps["SessionType"] = session_type
            
            if session_type == "R":
                total = session.laps["LapNumber"].max()
                laps["FuelLoad"] = 1.0 - (laps["LapNumber"] / total)
            elif session_type.startswith("FP"):
                laps["FuelLoad"] = 0.25
            elif session_type in ["Q", "SQ"]:
                laps["FuelLoad"] = 0.05
            else:
                laps["FuelLoad"] = 0.5
            
            # 7. Primera vuelta de cada stint (neumáticos fríos)
            # Eliminar primera vuelta con neumáticos nuevos (outlap efectivamente)
            laps_before = len(laps)
            laps = laps[laps["TyreLife"] > 1]
            filter_stats['first_lap_stint'] += (laps_before - len(laps))
            
            if len(laps) == 0:
                continue
            
            # 8. Filtro de tiempo razonable (60-120 segundos)
            # pick_quicklaps() ya debería haber eliminado la mayoría, pero aseguramos
            laps_before = len(laps)
            laps = laps[laps["LapTimeSec"].between(60, 120)]
            filter_stats['time_range'] += (laps_before - len(laps))
            
            if len(laps) == 0:
                continue
            
            # Ordenar por número de vuelta para preservar continuidad
            if len(laps) > 0 and 'LapNumber' in laps.columns:
                laps = laps.sort_values('LapNumber').reset_index(drop=True)
            
            # 9. Outliers: proteger pitstops y preservar continuidad de degradación
            # IMPORTANTE: Factor 2.0 (estricto) porque pick_quicklaps() ya limpió mucho
            if len(laps) > 10:
                laps_before = len(laps)
                laps = detect_outliers_iqr(
                    laps, 
                    column='LapTimeSec', 
                    factor=2.0,  # Más estricto porque pick_quicklaps() ya filtró
                    protect_pitstops=True, 
                    preserve_continuity=True
                )
                filter_stats['outliers'] += (laps_before - len(laps))
            
            if len(laps) > 0:
                all_laps.append(laps)
                filter_stats['final'] += len(laps)
            
        except Exception as e:
            continue
    
    if all_laps:
        print()
        combined = pd.concat(all_laps, ignore_index=True)
        
        # Mostrar estadísticas de filtrado
        print(f"    [FILTROS APLICADOS]")
        print(f"      Total vueltas:           {filter_stats['total']}")
        print(f"      - No quicklaps:          {filter_stats['not_quicklaps']} ({filter_stats['not_quicklaps']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      - Neumáticos mojados:    {filter_stats['wet_tyres']} ({filter_stats['wet_tyres']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      - SC/VSC/Bandera roja:   {filter_stats['sc_vsc']} ({filter_stats['sc_vsc']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      - Banderas amarillas:    {filter_stats['yellow_flags']} ({filter_stats['yellow_flags']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      - Primera vuelta stint:  {filter_stats['first_lap_stint']} ({filter_stats['first_lap_stint']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      - Fuera rango tiempo:    {filter_stats['time_range']} ({filter_stats['time_range']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      - Outliers:              {filter_stats['outliers']} ({filter_stats['outliers']/max(filter_stats['total'], 1)*100:.1f}%)")
        print(f"      = Vueltas finales:       {filter_stats['final']} ({filter_stats['final']/max(filter_stats['total'], 1)*100:.1f}%)\n")
        
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

def analyze_degradation(df):
    """Analiza el patrón de degradación en los datos"""
    print("\n" + "="*60)
    print("ANÁLISIS DE DEGRADACIÓN EN LOS DATOS")
    print("="*60)
    print("\nVERIFICANDO PATRÓN DE DEGRADACIÓN:")
    print("(Esperado: tiempos más lentos con más TyreLife)")
    
    # Crear bins de TyreLife
    df['TyreLifeBin'] = pd.cut(
        df['TyreLife'],
        bins=[0, 5, 15, 25, 35, 100],
        labels=['Muy nuevo', 'Nuevo', 'Medio', 'Usado', 'Muy usado']
    )
    
    print("\n1. ANÁLISIS SIN NORMALIZAR (puede estar afectado por combustible):")
    print("-" * 60)
    
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) == 0:
            continue
        
        grouped = compound_data.groupby('TyreLifeBin')['LapTimeSec'].agg(['count', 'mean', 'std'])
        print(f"\n{compound}:")
        print(grouped)
        
        if len(grouped) >= 2:
            degradation = grouped['mean'].iloc[-1] - grouped['mean'].iloc[0]
            print(f"  → Degradación aparente: {degradation:.3f}s ({degradation*1000:.0f}ms)")
            if degradation < 0:
                print(f"  ⚠  NEGATIVA - probablemente dominada por efecto de combustible")
    
    print("\n\n2. ANÁLISIS NORMALIZADO (eliminando efecto de combustible):")
    print("-" * 60)
    print("Restando la penalización de combustible para aislar degradación pura\n")
    
    # Crear tiempo normalizado
    df['LapTimeNormalized'] = df['LapTimeSec'] - df['FuelPenalty']
    
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) == 0:
            continue
        
        grouped = compound_data.groupby('TyreLifeBin')['LapTimeNormalized'].agg(['count', 'mean', 'std'])
        print(f"\n{compound}:")
        print(grouped)
        
        if len(grouped) >= 2:
            degradation_real = grouped['mean'].iloc[-1] - grouped['mean'].iloc[0]
            print(f"  → Degradación REAL (sin combustible): {degradation_real:.3f}s ({degradation_real*1000:.0f}ms)")
            if degradation_real > 0:
                print(f"  ✓ Degradación detectada correctamente.")
            else:
                print(f"  ⚠  ADVERTENCIA: Degradación aún negativa. Revisar datos.")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE DEGRADACIÓN DETECTADA:")
    print("="*60)
    
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) == 0:
            continue
        
        grouped = compound_data.groupby('TyreLifeBin')['LapTimeNormalized'].agg(['mean'])
        if len(grouped) >= 2:
            degradation = grouped['mean'].iloc[-1] - grouped['mean'].iloc[0]
            sign = "✓" if degradation > 0 else "⚠"
            print(f"  {sign} {compound:7s}: +{degradation:.3f}s (+{degradation*1000:.0f}ms)")
    
    # Calcular degradación promedio
    all_degradations = []
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) > 0:
            grouped = compound_data.groupby('TyreLifeBin')['LapTimeNormalized'].agg(['mean'])
            if len(grouped) >= 2:
                degradation = grouped['mean'].iloc[-1] - grouped['mean'].iloc[0]
                all_degradations.append(degradation)
    
    if all_degradations:
        avg_degradation = np.mean(all_degradations)
        print(f"\n  Degradación promedio: {avg_degradation:.3f}s")
        if avg_degradation > 0.3:  # Si la degradación promedio es > 0.3s
            print(f"  ✓ Degradación suficiente para que el modelo aprenda correctamente.")
        else:
            print(f"  ⚠  ADVERTENCIA: Degradación muy baja. El modelo puede no aprender bien.")
    
    # Análisis detallado de bins sospechosos
    print("\n" + "="*60)
    print("ANÁLISIS DETALLADO DE DATOS SOSPECHOSOS")
    print("="*60)
    
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) == 0:
            continue
        
        for bin_name in ['Muy usado', 'Usado']:
            bin_data = compound_data[compound_data['TyreLifeBin'] == bin_name]
            if len(bin_data) > 0:
                # Verificar si hay muchas vueltas con poco combustible
                low_fuel = (bin_data['FuelLoad'] < 0.2).sum()
                pct_low_fuel = low_fuel / len(bin_data) * 100
                
                if pct_low_fuel > 80:  # Si >80% tienen poco combustible
                    print(f"\n{compound} - Bin '{bin_name}' (n={len(bin_data)}):")
                    print(f"  TyreLife range: [{bin_data['TyreLife'].min()}, {bin_data['TyreLife'].max()}]")
                    print(f"  LapTimeSec:     mean={bin_data['LapTimeSec'].mean():.3f}s, std={bin_data['LapTimeSec'].std():.3f}s")
                    print(f"  FuelLoad:       mean={bin_data['FuelLoad'].mean():.3f}, std={bin_data['FuelLoad'].std():.3f}")
                    print(f"  → {pct_low_fuel:.1f}% de vueltas con FuelLoad < 0.2 (final de carrera)")
                    print(f"  ⚠  ADVERTENCIA: Muchas vueltas con poco combustible en '{bin_name}'")
                    print(f"      Esto puede sesgar los tiempos hacia valores más rápidos.")
                    print(f"      Solución: Filtrar vueltas por rango de TyreLife más específico.")
    
    print("="*60)
    
    # Análisis de correlación entre features
    print("\nCorrelación entre features de neumáticos:")
    tyre_features = ['TyreLife', 'TyreLifeSquared', 'TyreLifeCubed', 'TyreLifeByCompound']
    
    # Crear un subset con solo estas features
    if all(f in df.columns for f in tyre_features):
        corr_matrix = df[tyre_features].corr()
        
        print("Matriz de correlación:")
        for i, feat1 in enumerate(tyre_features):
            for feat2 in tyre_features[i+1:]:
                corr_val = corr_matrix.loc[feat1, feat2]
                print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")
    
    print("\n" + "="*60)

def train_model(df):
    """Entrena modelo XGBoost"""
    print("\n" + "="*60)
    print("ENTRENANDO MODELO XGBOOST")
    print("="*60 + "\n")
    
    # Features para el modelo
    feature_cols = [
        'TyreLife',
        'TyreLifeSquared',
        'TyreLifeCubed',
        'TyreLifeByCompound',
        'FuelLoad',
        'FuelPenalty',
        'TrackTemp'
    ]
    
    categorical_cols = ['Compound']
    
    # Preparar datos
    X = df[feature_cols + categorical_cols].copy()
    y = df['LapTimeSec'].copy()
    
    # Pipeline de preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Modelo XGBoost
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Entrenando modelo XGBoost...")
    model.fit(X, y)
    print("OK Completado")
    
    # Obtener feature importance
    regressor = model.named_steps['regressor']
    feature_names_encoded = model.named_steps['preprocessor'].get_feature_names_out()
    
    importances = regressor.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names_encoded,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 features con mayor importancia:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {idx}: {row['feature']} = {row['importance']:.6f}")
    
    # Mostrar features que varían (no categóricas)
    print("\nFeatures que varían (TyreLife, TyreLifeSquared, FuelLoad, etc.):")
    varying_features = [f for f in feature_importance_df['feature'] if not f.startswith('cat')]
    for feat in varying_features:
        imp = feature_importance_df[feature_importance_df['feature'] == feat]['importance'].values[0]
        idx = feature_importance_df[feature_importance_df['feature'] == feat].index[0]
        print(f"  {idx}: {feat} = {imp:.6f}")
    
    return model, feature_cols + categorical_cols

# Solo ejecutar esto si se ejecuta directamente, no cuando se importa
if __name__ == "__main__":
    print(f"[I] Circuito: {gp} ({year})")
    target_circuit = get_circuit_name(year, gp)
    print(f"[OK] {target_circuit}\n")
    
    # Cargar datos históricos
    print("="*60)
    print(f"CARGANDO DATOS HISTÓRICOS ({len(historical_years)} años)")
    print("="*60)
    
    all_data = []
    
    for hist_year in historical_years:
        print(f"\n[{hist_year}] Cargando datos...")
        race_data, _ = load_session_data(hist_year, gp, 'R', circuit_name=target_circuit)
        
        if race_data is not None and len(race_data) > 0:
            all_data.append(race_data)
            print(f"  ✓ {len(race_data)} vueltas cargadas")
    
    if not all_data:
        print("\n⚠ ERROR: No se pudieron cargar datos históricos")
        exit(1)
    
    # Combinar todos los datos
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"DATOS TOTALES: {len(df_combined)} vueltas")
    print(f"{'='*60}\n")
    
    # Crear features
    df_combined = create_features(df_combined)
    
    # Analizar degradación
    analyze_degradation(df_combined)
    
    # Entrenar modelo
    model, feature_names = train_model(df_combined)
    
    # Guardar modelo
    metadata = {
        'years': historical_years,
        'circuit': target_circuit,
        'n_samples': len(df_combined),
        'compounds': df_combined['Compound'].unique().tolist()
    }
    
    save_model(model, feature_names, metadata)
    
    print("\n✓ Proceso completado exitosamente")

