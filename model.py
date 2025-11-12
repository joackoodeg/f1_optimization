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

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils', 'XGBoost'))
from main import (
    load_session_data,
    create_features,
    save_model,
    get_circuit_name,
    year,
    gp,
    historical_years,
    MODEL_FILENAME
)

# Obtener nombre del circuito
print(f"[I] Circuito: {gp} ({year})")
target_circuit = get_circuit_name(year, gp)
if target_circuit is None:
    raise SystemExit(f"[X] No se pudo obtener el nombre del circuito para {year} {gp}")
print(f"[OK] {target_circuit}\n")

training_data_list = []
# SOLO CARGAR CARRERAS (SessionType == "R")
# Es crítico usar solo carreras porque las prácticas/qualifying tienen patrones diferentes
# (en qualifying los neumáticos SOFT pueden durar más porque no hay tanto desgaste)
# Aunque los datos de carrera tienen más ruido, son los únicos que reflejan la realidad de una carrera
sessions = ["R"]

print("[1/4] Cargando datos históricos (SOLO CARRERAS)...")
for hist_year in historical_years:
    try:
        schedule = fastf1.get_event_schedule(hist_year)
        match = schedule[schedule["EventName"] == target_circuit]
        if len(match) == 0:
            print(f"  {hist_year}: NO encontrado")
            continue
        
        hist_round = match.iloc[0]["RoundNumber"]
        print(f"  {hist_year} (R{hist_round}):")
        
        for sess in sessions:
            try:
                print(f"    {sess}: ", end="")
                data, _ = load_session_data(hist_year, hist_round, sess, circuit_name=target_circuit)
                if data is not None and len(data) > 0:
                    training_data_list.append(data)
                    print(f"OK {len(data)} laps")
                else:
                    print("SKIP")
            except:
                print("SKIP")
    except Exception as e:
        print(f"  {hist_year}: Error")

if not training_data_list:
    raise SystemExit("[X] No hay datos históricos de carreras")

all_train_data = pd.concat(training_data_list, ignore_index=True)
print(f"\n  Total histórico (carreras): {len(all_train_data)} laps\n")

# NO cargar datos del año actual (2025) - solo usar datos históricos
print("[2/4] Verificando que solo hay datos de carrera...")
print(f"  Total de datos: {len(all_train_data)}")
if 'SessionType' in all_train_data.columns:
    session_counts = all_train_data['SessionType'].value_counts()
    print(f"  Distribución por sesión:")
    for session, count in session_counts.items():
        print(f"    {session}: {count}")
    
    # Filtrar SOLO carreras - sin excepciones
    race_data = all_train_data[all_train_data['SessionType'] == 'R'].copy()
    print(f"\n  Datos de carrera (SessionType='R'): {len(race_data)}")
    
    if len(race_data) == 0:
        raise SystemExit("[X] ERROR: No hay datos de carrera. El modelo solo puede entrenarse con carreras.")
    
    if len(race_data) < 100:
        print(f"  [!] Advertencia: Pocos datos de carrera ({len(race_data)}). Continuando de todos modos.")
    
    # Verificar que no hay datos de otras sesiones
    non_race_data = all_train_data[all_train_data['SessionType'] != 'R']
    if len(non_race_data) > 0:
        print(f"  [!] ADVERTENCIA: Se encontraron {len(non_race_data)} datos de sesiones no-carrera. Serán eliminados.")
        print(f"      Sesiones encontradas: {non_race_data['SessionType'].unique()}")
else:
    raise SystemExit("[X] ERROR: No hay columna SessionType. No se puede verificar que solo hay carreras.")

# NO separar validación - usar TODOS los datos para entrenar
print("\n[3/4] Preparando datos para entrenamiento (SIN validación)...")
train_data = race_data.copy()
print(f"  Train: {len(train_data)} laps (100% de los datos)\n")

# Features
print("[4/4] Feature engineering...")
train_data = create_features(train_data)

# OPCIÓN: Eliminar features de temperatura para que el modelo se enfoque en degradación
# Las temperaturas varían mucho entre años (30-49°C) y dominan la predicción
# Al eliminarlas, forzamos al modelo a aprender de TyreLife y Compound
NORMALIZE_TEMPERATURES = True

if NORMALIZE_TEMPERATURES:
    print("  [EXPERIMENTAL] Normalizando temperaturas para reducir variabilidad...")
    
    # Reemplazar temperaturas variables por valores constantes (de referencia 2025)
    # Esto obliga al modelo a aprender degradación en lugar de temperatura
    train_data['TrackTemp'] = 42.8  # Temperatura de referencia (2025 Monza Q)
    train_data['AirTemp'] = 26.6    # Temperatura de referencia (2025 Monza Q)
    train_data['Humidity'] = 35.8   # Humedad de referencia (2025 Monza Q)
    
    print(f"    ✓ Temperaturas fijadas a valores de referencia (2025 Monza Qualifying)")
    print(f"      Pista: 42.8°C | Aire: 26.6°C | Humedad: 35.8%")
    print(f"    → Esto obliga al modelo a aprender degradación en lugar de temperatura")
print()

# Verificación final: asegurar que solo tenemos datos de carrera
if 'SessionType' in train_data.columns:
    non_race_train = train_data[train_data['SessionType'] != 'R']
    if len(non_race_train) > 0:
        print(f"  [!] ADVERTENCIA CRÍTICA: Se encontraron datos no-carrera después del feature engineering.")
        print(f"      Train: {len(non_race_train)} datos no-carrera")
        print(f"      Eliminando datos no-carrera...")
        train_data = train_data[train_data['SessionType'] == 'R'].copy()
        print(f"      Train después de filtro: {len(train_data)}")
    else:
        print(f"  [OK] Verificado: Solo datos de carrera (Train: {len(train_data)})")
print()

# FEATURES MEJORADAS PARA CAPTURAR DEGRADACIÓN
# IMPORTANTE: Eliminar features constantes (AvgSpeed, AvgThrottle, MaxSpeed) porque
# tienen correlación muy alta durante el entrenamiento pero son constantes durante la predicción
# Esto hace que el modelo devuelva siempre el mismo valor
# También eliminar features redundantes:
# - TyreWearRate: correlación perfecta (1.0) con TyreLife
# - RelativeTyreAge: correlacionada con TyreLife y CompoundHardness
# - CompoundHardness: redundante con Compound (categórica), pero se usa para calcular TyreLifeByCompound
features = [
    # Categóricas
    "Compound",
    # SessionType ELIMINADA: Solo usamos datos de carrera (SessionType="R"), por lo que es constante
    # Team ELIMINADA: No es relevante para predecir degradación de neumáticos
    # Neumáticos básicas
    "TyreLife", 
    #"TyreLifeSquared", "TyreLifeCubed",
    # Interacciones compuesto-edad (el modelo aprenderá la degradación naturalmente)
    "TyreLifeByCompound",
    # Combustible
    "FuelLoad",
    # Temperatura (mantener porque varían entre carreras)
    #"TrackTemp", "AirTemp", "Humidity"
    # Telemetría ELIMINADA: MaxSpeed, AvgSpeed, AvgThrottle
    # Features redundantes ELIMINADAS: TyreWearRate, RelativeTyreAge, CompoundHardness, SoftDegradation, SessionType, Team
]

print(f"  Features: {len(features)}\n")

# Preparar datos
y_train = np.log1p(train_data["LapTimeSec"])

X_train = train_data[features]

# Rellenar NaN solo en columnas numéricas
numerical_features = [f for f in features if f not in ["Compound"]]
X_train[numerical_features] = X_train[numerical_features].fillna(X_train[numerical_features].median())

# Debug: Verificar variación de features
print("Variación de features en el conjunto de entrenamiento:")
varying_features = [#'TyreLifeSquared', 'TyreLifeCubed',
                    'TyreLife', 'TyreLifeByCompound', 'FuelLoad']
constant_features = ['TrackTemp', 'AirTemp', 'Humidity']
for feat_name in varying_features + constant_features:
    if feat_name in X_train.columns:
        std_val = X_train[feat_name].std()
        mean_val = X_train[feat_name].mean()
        cv = std_val / mean_val if mean_val != 0 else 0  # Coeficiente de variación
        min_val = X_train[feat_name].min()
        max_val = X_train[feat_name].max()
        print(f"  {feat_name}: std={std_val:.3f}, mean={mean_val:.3f}, CV={cv:.3f}, range=[{min_val:.1f}, {max_val:.1f}]")
print()

# Debug: Análisis detallado de TyreLife
print("Análisis de TyreLife en datos de entrenamiento:")
if 'TyreLife' in train_data.columns:
    print(f"  Min: {train_data['TyreLife'].min()}")
    print(f"  Max: {train_data['TyreLife'].max()}")
    print(f"  Mean: {train_data['TyreLife'].mean():.2f}")
    print(f"  Median: {train_data['TyreLife'].median():.2f}")
    print(f"  Percentiles: 25%={train_data['TyreLife'].quantile(0.25):.1f}, 50%={train_data['TyreLife'].quantile(0.50):.1f}, 75%={train_data['TyreLife'].quantile(0.75):.1f}, 95%={train_data['TyreLife'].quantile(0.95):.1f}")
    
    # Verificar distribución por compuesto
    print("\n  Distribución de TyreLife por compuesto:")
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = train_data[train_data['Compound'] == compound]
        if len(compound_data) > 0:
            print(f"    {compound}: count={len(compound_data)}, TyreLife range=[{compound_data['TyreLife'].min():.0f}, {compound_data['TyreLife'].max():.0f}], mean={compound_data['TyreLife'].mean():.2f}")
    
    # Verificar si hay correlación entre TyreLife y LapTimeSec por compuesto
    print("\n  Correlación TyreLife vs LapTimeSec por compuesto:")
    for compound in ['SOFT', 'MEDIUM', 'HARD']:
        compound_data = train_data[train_data['Compound'] == compound]
        if len(compound_data) > 10:  # Solo si hay suficientes datos
            corr = compound_data['TyreLife'].corr(compound_data['LapTimeSec'])
            print(f"    {compound}: corr={corr:.4f} (n={len(compound_data)})")
print()

# Debug: Verificar correlación con el target
print("Correlación de features con el target (LapTimeSec):")
train_data_with_target = train_data.copy()
train_data_with_target['LapTimeSec'] = train_data['LapTimeSec']
for feat_name in varying_features + constant_features:
    if feat_name in train_data_with_target.columns:
        corr = train_data_with_target[feat_name].corr(train_data_with_target['LapTimeSec'])
        print(f"  {feat_name}: corr={corr:.4f}")
print()

# ANÁLISIS CRÍTICO: Verificar que la degradación esté presente en los datos
print("="*60)
print("ANÁLISIS DE DEGRADACIÓN EN LOS DATOS")
print("="*60)
print("\nVERIFICANDO PATRÓN DE DEGRADACIÓN:")
print("(Esperado: tiempos más lentos con más TyreLife)\n")

# Análisis SIN normalizar (raw)
print("1. ANÁLISIS SIN NORMALIZAR (puede estar afectado por combustible):")
print("-" * 60)
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    compound_data = train_data[train_data['Compound'] == compound].copy()
    if len(compound_data) < 50:
        continue
    
    # Dividir en 5 bins de TyreLife
    compound_data['TyreLifeBin'] = pd.cut(compound_data['TyreLife'], bins=5, labels=['Muy nuevo', 'Nuevo', 'Medio', 'Usado', 'Muy usado'])
    
    print(f"\n{compound}:")
    bin_stats = compound_data.groupby('TyreLifeBin')['LapTimeSec'].agg(['count', 'mean', 'std'])
    print(bin_stats)
    
    # Calcular diferencia entre muy nuevo y muy usado
    if 'Muy nuevo' in bin_stats.index and 'Muy usado' in bin_stats.index:
        diff = bin_stats.loc['Muy usado', 'mean'] - bin_stats.loc['Muy nuevo', 'mean']
        print(f"  → Degradación aparente: {diff:.3f}s ({diff*1000:.0f}ms)")
        if diff < 0:
            print(f"  ⚠️  NEGATIVA - probablemente dominada por efecto de combustible")

# Análisis NORMALIZADO por combustible
print("\n\n2. ANÁLISIS NORMALIZADO (eliminando efecto de combustible):")
print("-" * 60)
print("Restando la penalización de combustible para aislar degradación pura\n")

degradation_detected = {}
for compound in ['SOFT', 'MEDIUM', 'HARD']:
    compound_data = train_data[train_data['Compound'] == compound].copy()
    if len(compound_data) < 50:
        continue
    
    # NORMALIZAR: Restar efecto de combustible (3s por carga completa)
    # LapTime normalizado = LapTime real - penalización de combustible
    compound_data['LapTimeNormalized'] = compound_data['LapTimeSec'] - (compound_data['FuelLoad'] * 3.0)
    
    # Dividir en bins de TyreLife
    compound_data['TyreLifeBin'] = pd.cut(compound_data['TyreLife'], bins=5, labels=['Muy nuevo', 'Nuevo', 'Medio', 'Usado', 'Muy usado'])
    
    print(f"\n{compound}:")
    bin_stats = compound_data.groupby('TyreLifeBin')['LapTimeNormalized'].agg(['count', 'mean', 'std'])
    print(bin_stats)
    
    # Calcular degradación real
    if 'Muy nuevo' in bin_stats.index and 'Muy usado' in bin_stats.index:
        diff = bin_stats.loc['Muy usado', 'mean'] - bin_stats.loc['Muy nuevo', 'mean']
        print(f"  → Degradación REAL (sin combustible): {diff:.3f}s ({diff*1000:.0f}ms)")
        degradation_detected[compound] = diff
        
        if diff < 0:
            print(f"  ⚠️  ADVERTENCIA: Degradación NEGATIVA incluso sin combustible.")
        elif diff < 0.3:
            print(f"  ⚠️  ADVERTENCIA: Degradación muy baja ({diff:.3f}s).")
        else:
            print(f"  ✓ Degradación detectada correctamente.")

# Resumen
print("\n" + "="*60)
print("RESUMEN DE DEGRADACIÓN DETECTADA:")
print("="*60)
for compound, deg in degradation_detected.items():
    status = "✓" if deg > 0.3 else "⚠️"
    print(f"  {status} {compound:8s}: {deg:+.3f}s ({deg*1000:+.0f}ms)")

# Verificar si el modelo puede aprender degradación
avg_degradation = np.mean(list(degradation_detected.values()))
print(f"\n  Degradación promedio: {avg_degradation:.3f}s")
if avg_degradation < 0.1:
    print("  ❌ PROBLEMA: Degradación muy baja. El modelo no podrá aprender correctamente.")
    print("     → Posibles causas:")
    print("       - Datos muy filtrados (eliminamos las vueltas con degradación)")
    print("       - Mezcla de condiciones muy diferentes entre años")
    print("       - pick_quicklaps() elimina vueltas lentas (las más degradadas)")
elif avg_degradation < 0.3:
    print("  ⚠️  ADVERTENCIA: Degradación baja. El modelo tendrá dificultad para aprender.")
else:
    print("  ✓ Degradación suficiente para que el modelo aprenda correctamente.")

# ANÁLISIS ADICIONAL: Verificar outliers en "Muy usado" de HARD
print("\n" + "="*60)
print("ANÁLISIS DETALLADO DE DATOS SOSPECHOSOS")
print("="*60)

# Verificar el bin "Muy usado" de HARD que mostró degradación negativa
hard_data = train_data[train_data['Compound'] == 'HARD'].copy()
if len(hard_data) > 0:
    # Crear bins
    hard_data['TyreLifeBin'] = pd.cut(hard_data['TyreLife'], bins=5, labels=['Muy nuevo', 'Nuevo', 'Medio', 'Usado', 'Muy usado'])
    
    # Analizar el bin "Muy usado"
    muy_usado = hard_data[hard_data['TyreLifeBin'] == 'Muy usado']
    
    if len(muy_usado) > 0:
        print(f"\nHARD - Bin 'Muy usado' (n={len(muy_usado)}):")
        print(f"  TyreLife range: [{muy_usado['TyreLife'].min():.0f}, {muy_usado['TyreLife'].max():.0f}]")
        print(f"  LapTimeSec:     mean={muy_usado['LapTimeSec'].mean():.3f}s, std={muy_usado['LapTimeSec'].std():.3f}s")
        print(f"  FuelLoad:       mean={muy_usado['FuelLoad'].mean():.3f}, std={muy_usado['FuelLoad'].std():.3f}")
        
        # Verificar si hay un patrón de combustible bajo
        low_fuel_pct = (muy_usado['FuelLoad'] < 0.2).sum() / len(muy_usado) * 100
        print(f"  → {low_fuel_pct:.1f}% de vueltas con FuelLoad < 0.2 (final de carrera)")
        
        if low_fuel_pct > 50:
            print(f"  ⚠️  ADVERTENCIA: Muchas vueltas con poco combustible en 'Muy usado'")
            print(f"      Esto puede sesgar los tiempos hacia valores más rápidos.")
            print(f"      Solución: Filtrar vueltas por rango de TyreLife más específico.")

print("="*60)
print()

# Debug: Verificar correlación entre features de neumáticos
print("Correlación entre features de neumáticos:")
tyre_features = [#'TyreLifeSquared', 'TyreLifeCubed',
    'TyreLife', 'TyreLifeByCompound']
tyre_corr_matrix = train_data[tyre_features].corr()
print("Matriz de correlación:")
for i, feat1 in enumerate(tyre_features):
    for j, feat2 in enumerate(tyre_features):
        if i < j:  # Solo mostrar la mitad superior
            corr_val = tyre_corr_matrix.loc[feat1, feat2]
            if abs(corr_val) > 0.9:  # Solo mostrar correlaciones muy altas
                print(f"  {feat1} <-> {feat2}: {corr_val:.4f}")
print()

print("="*60)
print("ENTRENANDO MODELO XGBOOST")
print("="*60 + "\n")

# Preprocessor
categorical = ["Compound"]
numerical = [f for f in features if f not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ("num", StandardScaler(), numerical)
])

# MODELO XGBOOST
# Modelo robusto para datos de carrera con mucho ruido
# Necesitamos que aprenda la degradación a pesar del ruido
model = Pipeline([
    ("pre", preprocessor),
    ("reg", XGBRegressor(
        n_estimators=500,        # Más árboles para capturar patrones a pesar del ruido
        learning_rate=0.03,      # Tasa de aprendizaje más baja para aprender mejor
        max_depth=6,             # Profundidad moderada-alta para capturar degradación
        subsample=0.8,           # Usar 80% de muestras para cada árbol (reducir sobreajuste)
        colsample_bytree=0.8,    # Usar 80% de features para cada árbol
        colsample_bylevel=0.8,   # Usar 80% de features en cada nivel
        min_child_weight=2,      # Peso mínimo bajo para permitir splits en degradación
        gamma=0.0,               # Sin penalización mínima para permitir splits
        reg_alpha=0.0,           # Sin regularización L1 para no penalizar features de degradación
        reg_lambda=0.5,          # Regularización L2 baja para no penalizar tanto las features
        random_state=42,
        verbosity=0,
        importance_type='gain'  # Usar 'gain' en lugar de 'weight' para mejor importancia de features
    ))
])

print("Entrenando modelo XGBoost...")
model.fit(X_train, y_train)
print("OK Completado\n")

# Verificar importancia de features
regressor = model.named_steps['reg']
feature_importance = regressor.feature_importances_

# Obtener nombres de features después del preprocesamiento
feature_names_after_preprocessing = []
preprocessor = model.named_steps['pre']
for name, transformer, columns in preprocessor.transformers_:
    if name == 'cat':
        if hasattr(transformer, 'categories_'):
            for i, col in enumerate(columns):
                for cat in transformer.categories_[i]:
                    feature_names_after_preprocessing.append(f"{col}_{cat}")
    elif name == 'num':
        feature_names_after_preprocessing.extend(columns)

# Mostrar top 10 features con mayor importancia
top_indices = np.argsort(feature_importance)[-10:][::-1]
print("Top 10 features con mayor importancia:")
for idx in top_indices:
    if idx < len(feature_names_after_preprocessing):
        print(f"  {idx}: {feature_names_after_preprocessing[idx]} = {feature_importance[idx]:.6f}")
print()

# Mostrar todas las features que varían y su importancia
print("Features que varían (TyreLife, TyreLifeByCompound, FuelLoad, etc.):")
varying_features = [#'TyreLifeSquared', 'TyreLifeCubed', 
    'TyreLife', 'TyreLifeByCompound', 'FuelLoad']
for feat_name in varying_features:
    if feat_name in feature_names_after_preprocessing:
        idx = feature_names_after_preprocessing.index(feat_name)
        print(f"  {idx}: {feat_name} = {feature_importance[idx]:.6f}")
print()

# Entrenar con TODOS los datos (sin validación)
print("Entrenando con todos los datos disponibles...")
model.fit(X_train, y_train)

# Evaluación solo en datos de entrenamiento
r2_train = model.score(X_train, y_train)
y_pred_train = model.predict(X_train)
mae_train = mean_absolute_error(np.expm1(y_train), np.expm1(y_pred_train))

print("="*60)
print("RESULTADOS FINALES")
print("="*60)
print(f"\nENTRENAMIENTO (100% de los datos):")
print(f"  R²:   {r2_train:.3f}")
print(f"  MAE:  {mae_train:.3f}s")
print(f"\n[NOTA] No se reservó conjunto de validación - modelo entrenado con todos los datos disponibles")

# Guardar con el mismo nombre que antes para compatibilidad con ES.py
metadata = {
    'year': year,
    'gp': gp,
    'circuit': target_circuit,
    'version': 'v2_conservative',
    'train_samples': len(train_data),
    'val_samples': 0,  # Sin validación
    'r2_train': r2_train,
    'r2_val': None,  # Sin validación
    'mae_train': mae_train,
    'mae_val': None,  # Sin validación
    'features_count': len(features),
    'note': 'Modelo entrenado sin datos de 2025 y sin conjunto de validación'
}

# Asegurar que se guarde con el nombre exacto que espera ES.py: model_2025_gpMonza.pkl
save_model(model, features, metadata, MODEL_FILENAME)
print(f"[OK] Modelo guardado como: {MODEL_FILENAME}")

print("\nOK MODELO XGBOOST LISTO\n")
