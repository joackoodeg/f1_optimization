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
            # Es el filtro MÁS IMPORTANTE de FastF1
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
            
            # 9. Outliers: NO SE UTILIZA - pick_quicklaps() ya hizo el trabajo
            # detect_outliers_iqr() eliminado porque pick_quicklaps() es suficiente
            
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
    # Compuesto numérico (solo para calcular TyreLifeByCompound)
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    df["CompoundHardness"] = df["Compound"].map(compound_map)
    
    # Interacción compuesto-edad (degradación por tipo de neumático)
    df["TyreLifeByCompound"] = df["TyreLife"] * df["CompoundHardness"]
    
    return df

def analyze_degradation(df):
    """Analiza el patrón de degradación en los datos"""
    print("\n" + "="*60)
    print("ANÁLISIS DE DEGRADACIÓN EN LOS DATOS")
    print("="*60)
    print("\nVERIFICANDO PATRÓN DE DEGRADACIÓN:")
    print("(Esperado: tiempos más lentos con más TyreLife)")
    
    # Bins adaptativos por compuesto (basados en vida útil real)
    compound_bins = {
        'SOFT': {
            'bins': [0, 5, 10, 15, 20, 30],
            'labels': ['Muy nuevo (0-5)', 'Nuevo (5-10)', 'Medio (10-15)', 'Usado (15-20)', 'Muy usado (20-30)']
        },
        'MEDIUM': {
            'bins': [0, 5, 12, 20, 30, 45],
            'labels': ['Muy nuevo (0-5)', 'Nuevo (5-12)', 'Medio (12-20)', 'Usado (20-30)', 'Muy usado (30-45)']
        },
        'HARD': {
            'bins': [0, 5, 15, 25, 35, 55],
            'labels': ['Muy nuevo (0-5)', 'Nuevo (5-15)', 'Medio (15-25)', 'Usado (25-35)', 'Muy usado (35-55)']
        }
    }
    
    print("\nANÁLISIS DE TIEMPOS POR DEGRADACIÓN:")
    print("-" * 60)
    
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = df[df['Compound'] == compound]
        if len(compound_data) == 0:
            continue
        
        # Crear bins específicos para este compuesto
        bins_config = compound_bins[compound]
        compound_data['TyreLifeBin'] = pd.cut(
            compound_data['TyreLife'],
            bins=bins_config['bins'],
            labels=bins_config['labels']
        )
        
        grouped = compound_data.groupby('TyreLifeBin')['LapTimeSec'].agg(['count', 'mean', 'std'])
        print(f"\n{compound}:")
        print(grouped)
        
        # Calcular degradación usando el último bin con datos válidos
        if len(grouped) >= 2:
            # Filtrar bins con datos válidos (count > 0)
            valid_bins = grouped[grouped['count'] > 0]
            if len(valid_bins) >= 2:
                degradation = valid_bins['mean'].iloc[-1] - valid_bins['mean'].iloc[0]
                print(f"  → Degradación aparente: {degradation:.3f}s ({degradation*1000:.0f}ms)")
                if degradation < 0:
                    print(f"  ⚠️  NEGATIVA - dominada por efecto de combustible")
            else:
                print(f"  → Degradación aparente: N/A (insuficientes datos)")
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN:")
    print("="*60)
    print("  Nota: Degradación aparente puede ser negativa debido al efecto")
    print("  del combustible. El modelo aprenderá ambos efectos por separado.")
    
    # Análisis de correlación
    print("\n" + "="*60)
    print("Correlación entre features de neumáticos:")
    tyre_features = ['TyreLife', 'TyreLifeByCompound']
    
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
        'TyreLifeByCompound',
        'FuelLoad',
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
    
    # Mostrar features numéricas
    print("\nFeatures numéricas:")
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

