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
sessions = ["R", "Q", "FP3", "FP2"]

print("[1/4] Cargando datos históricos...")
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
    raise SystemExit("[X] No hay datos históricos")

all_train_data = pd.concat(training_data_list, ignore_index=True)
print(f"\n  Total histórico: {len(all_train_data)} laps\n")

# Datos año actual
print("[2/4] Cargando datos año actual...")
current_sessions = ["FP1", "FP2", "FP3", "Q"]
current_data_list = []

for sess in current_sessions:
    try:
        print(f"  {sess}: ", end="")
        data, _ = load_session_data(year, gp, sess, circuit_name=target_circuit)
        if data is not None and len(data) > 0:
            current_data_list.append(data)
            print(f"OK {len(data)} laps")
        else:
            print("SKIP")
    except:
        print("SKIP")

if current_data_list:
    current_data = pd.concat(current_data_list, ignore_index=True)
    all_train_data = pd.concat([all_train_data, current_data], ignore_index=True)
    print(f"\n  Total con {year}: {len(all_train_data)} laps\n")
else:
    print(f"\n  [!] Sin datos de {year}, solo histórico\n")

# Validación: usar último 20%
print("[3/4] Preparando validación...")
split_idx = int(len(all_train_data) * 0.8)
train_data = all_train_data.iloc[:split_idx].copy()
test_data = all_train_data.iloc[split_idx:].copy()
print(f"  Train: {len(train_data)} | Val: {len(test_data)}\n")

# Features
print("[4/4] Feature engineering...")
train_data = create_features(train_data)
test_data = create_features(test_data)

# SOLO FEATURES BÁSICAS
features = [
    # Categóricas
    "Compound", "SessionType", "Team",
    # Neumáticos
    "TyreLife", "TyreWearRate", "TyreLifeSquared", "CompoundHardness",
    # Combustible
    "FuelLoad", "FuelPenalty",
    # Temperatura
    "TrackTemp", "AirTemp", "Humidity", "TempDiff",
    # Telemetría
    "MaxSpeed", "AvgSpeed", "AvgThrottle"
]

print(f"  Features: {len(features)}\n")

# Preparar datos
y_train = np.log1p(train_data["LapTimeSec"])
y_test = np.log1p(test_data["LapTimeSec"])

X_train = train_data[features]
X_test = test_data[features]

# Rellenar NaN solo en columnas numéricas
numerical_features = [f for f in features if f not in ["Compound", "SessionType", "Team"]]
X_train[numerical_features] = X_train[numerical_features].fillna(X_train[numerical_features].median())
X_test[numerical_features] = X_test[numerical_features].fillna(X_train[numerical_features].median())

print("="*60)
print("ENTRENANDO MODELO XGBOOST")
print("="*60 + "\n")

# Preprocessor
categorical = ["Compound", "SessionType", "Team"]
numerical = [f for f in features if f not in categorical]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ("num", StandardScaler(), numerical)
])

# MODELO XGBOOST
model = Pipeline([
    ("pre", preprocessor),
    ("reg", XGBRegressor(
        n_estimators=300,        # Número de árboles en el modelo (más árboles = mejor ajuste pero más tiempo)
        learning_rate=0.05,      # Tasa de aprendizaje: controla qué tan rápido aprende el modelo (más bajo = más conservador)
        max_depth=4,             # Profundidad máxima de cada árbol (más profundo = más complejo, riesgo de overfitting)
        subsample=0.6,           # Fracción de muestras usadas para entrenar cada árbol (previene overfitting)
        colsample_bytree=0.6,    # Fracción de features usadas para cada árbol (reduce correlación entre árboles)
        min_child_weight=5,      # Peso mínimo requerido en nodos hijo (mayor valor = más conservador, previene overfitting)
        gamma=0.2,               # Penalización mínima de pérdida para hacer splits (mayor valor = árboles más simples)
        reg_alpha=0.2,           # Regularización L1 (Lasso) - penaliza features poco importantes
        reg_lambda=2.0,          # Regularización L2 (Ridge) - penaliza pesos grandes para evitar overfitting
        random_state=42,
        verbosity=0
    ))
])

print("Entrenando modelo XGBoost...")
model.fit(X_train, y_train)
print("OK Completado\n")

# Evaluación
print("="*60)
print("EVALUACIÓN")
print("="*60 + "\n")

# Cross-validation temporal
tscv = TimeSeriesSplit(n_splits=3)
cv_scores = []

print("Validación cruzada:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    model.fit(X_fold_train, y_fold_train)
    score = model.score(X_fold_val, y_fold_val)
    cv_scores.append(score)
    print(f"  Fold {fold}: R² = {score:.3f}")

print(f"\n  Media CV: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}\n")

# Re-entrenar con todos los datos
model.fit(X_train, y_train)

r2_train = model.score(X_train, y_train)
r2_val = model.score(X_test, y_test)

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_test)

mae_train = mean_absolute_error(np.expm1(y_train), np.expm1(y_pred_train))
mae_val = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_val))

print("="*60)
print("RESULTADOS FINALES")
print("="*60)
print(f"\nTRAIN:")
print(f"  R²:   {r2_train:.3f}")
print(f"  MAE:  {mae_train:.3f}s")
print(f"\nVALIDACIÓN:")
print(f"  R²:   {r2_val:.3f}")
print(f"  MAE:  {mae_val:.3f}s")

# Guardar
metadata = {
    'year': year,
    'gp': gp,
    'circuit': target_circuit,
    'version': 'v2_conservative',
    'train_samples': len(train_data),
    'val_samples': len(test_data),
    'r2_train': r2_train,
    'r2_val': r2_val,
    'mae_train': mae_train,
    'mae_val': mae_val,
    'features_count': len(features)
}

save_model(model, features, metadata, MODEL_FILENAME)

print("\nOK MODELO XGBOOST LISTO\n")
