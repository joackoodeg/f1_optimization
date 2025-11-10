"""
Script para debuggear el modelo XGBoost y entender qué features tiene importancia
y cómo predice para diferentes valores de degradación de neumáticos.
"""
import os
import numpy as np
import pandas as pd
import joblib
from utils.ES.main import reference_conditions, predict_lap_time, create_features_for_lap

# Cargar modelo
base_dir = os.path.dirname(__file__)
models_dir = os.path.join(base_dir, 'utils', 'XGBoost', 'models')
MODEL_FILENAME = "model_2025_gpMonza.pkl"
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
print(f"  MAE: {metadata['mae_val']:.3f}s\n")

# Obtener importancia de features
from sklearn.pipeline import Pipeline
if isinstance(model, Pipeline):
    regressor = model.named_steps.get('reg', None)
    if hasattr(regressor, 'feature_importances_'):
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
        
        print("="*60)
        print("IMPORTANCIA DE FEATURES")
        print("="*60)
        
        # Mostrar todas las features ordenadas por importancia
        feature_importance_dict = {name: imp for name, imp in zip(feature_names_after_preprocessing, feature_importance)}
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTodas las features ordenadas por importancia:")
        for i, (name, imp) in enumerate(sorted_features, 1):
            print(f"  {i:2d}. {name:30s} = {imp:.6f}")
        
        print("\n" + "="*60)
        print("FEATURES DE DEGRADACIÓN")
        print("="*60)
        degradation_features = ['TyreLife', 'TyreLifeSquared', 'TyreLifeCubed', 'TyreLifeByCompound']
        for feat_name in degradation_features:
            if feat_name in feature_names_after_preprocessing:
                idx = feature_names_after_preprocessing.index(feat_name)
                print(f"  {feat_name:25s} = {feature_importance[idx]:.6f}")
            else:
                print(f"  {feat_name:25s} = NO ENCONTRADA")

# Probar predicciones con diferentes valores de degradación
print("\n" + "="*60)
print("PREDICCIONES CON DIFERENTES VALORES DE DEGRADACIÓN")
print("="*60)

# Parámetros de prueba
lap_number = 25
fuel_load = 0.5  # Mitad de carrera

print(f"\nParámetros de prueba:")
print(f"  Lap: {lap_number}")
print(f"  Fuel Load: {fuel_load:.2f}")
print(f"  Condiciones de referencia:")
for key, value in reference_conditions.items():
    print(f"    {key}: {value}")

print("\n" + "-"*60)
print("SOFT - Degradación progresiva")
print("-"*60)
for tyre_life in [0, 10, 20, 30, 40, 50]:
    try:
        lap_time = predict_lap_time(
            lap_number, "SOFT", tyre_life, fuel_load,
            reference_conditions, model, features
        )
        # Obtener features usadas
        features_dict = create_features_for_lap(lap_number, "SOFT", tyre_life, fuel_load, reference_conditions)
        print(f"  TyreLife={tyre_life:2d}: {lap_time:.3f}s | "
              f"TyreLife²={features_dict['TyreLifeSquared']:4d} | "
              f"TyreLife³={features_dict['TyreLifeCubed']:5d} | "
              f"TyreLifeByComp={features_dict['TyreLifeByCompound']:3.1f}")
    except Exception as e:
        print(f"  TyreLife={tyre_life:2d}: ERROR - {e}")

print("\n" + "-"*60)
print("MEDIUM - Degradación progresiva")
print("-"*60)
for tyre_life in [0, 10, 20, 30, 40, 50]:
    try:
        lap_time = predict_lap_time(
            lap_number, "MEDIUM", tyre_life, fuel_load,
            reference_conditions, model, features
        )
        features_dict = create_features_for_lap(lap_number, "MEDIUM", tyre_life, fuel_load, reference_conditions)
        print(f"  TyreLife={tyre_life:2d}: {lap_time:.3f}s | "
              f"TyreLife²={features_dict['TyreLifeSquared']:4d} | "
              f"TyreLife³={features_dict['TyreLifeCubed']:5d} | "
              f"TyreLifeByComp={features_dict['TyreLifeByCompound']:3.1f}")
    except Exception as e:
        print(f"  TyreLife={tyre_life:2d}: ERROR - {e}")

print("\n" + "-"*60)
print("HARD - Degradación progresiva")
print("-"*60)
for tyre_life in [0, 10, 20, 30, 40, 50]:
    try:
        lap_time = predict_lap_time(
            lap_number, "HARD", tyre_life, fuel_load,
            reference_conditions, model, features
        )
        features_dict = create_features_for_lap(lap_number, "HARD", tyre_life, fuel_load, reference_conditions)
        print(f"  TyreLife={tyre_life:2d}: {lap_time:.3f}s | "
              f"TyreLife²={features_dict['TyreLifeSquared']:4d} | "
              f"TyreLife³={features_dict['TyreLifeCubed']:5d} | "
              f"TyreLifeByComp={features_dict['TyreLifeByCompound']:3.1f}")
    except Exception as e:
        print(f"  TyreLife={tyre_life:2d}: ERROR - {e}")

# Comparar predicciones para un stint completo de SOFT
print("\n" + "="*60)
print("SIMULACIÓN DE STINT COMPLETO: SOFT (52 vueltas)")
print("="*60)
total_time = 0.0
print("\nPrimeras 10 vueltas:")
for lap in range(10):
    tyre_life = lap
    fuel_load = 1.0 - (lap / 53)
    lap_time = predict_lap_time(
        lap, "SOFT", tyre_life, fuel_load,
        reference_conditions, model, features
    )
    total_time += lap_time
    print(f"  Lap {lap:2d}: TyreLife={tyre_life:2d}, Fuel={fuel_load:.2f} -> {lap_time:.3f}s")

print("\nVueltas intermedias (20, 30, 40, 50):")
for lap in [20, 30, 40, 50]:
    tyre_life = lap
    fuel_load = 1.0 - (lap / 53)
    lap_time = predict_lap_time(
        lap, "SOFT", tyre_life, fuel_load,
        reference_conditions, model, features
    )
    total_time += lap_time
    print(f"  Lap {lap:2d}: TyreLife={tyre_life:2d}, Fuel={fuel_load:.2f} -> {lap_time:.3f}s")

print(f"\nÚltimas 3 vueltas (51, 52):")
for lap in [51, 52]:
    tyre_life = lap
    fuel_load = 1.0 - (lap / 53)
    lap_time = predict_lap_time(
        lap, "SOFT", tyre_life, fuel_load,
        reference_conditions, model, features
    )
    total_time += lap_time
    print(f"  Lap {lap:2d}: TyreLife={tyre_life:2d}, Fuel={fuel_load:.2f} -> {lap_time:.3f}s")

print(f"\nTiempo total estimado para 52 vueltas con SOFT: {total_time:.3f}s ({total_time/60:.2f} minutos)")

print("\n" + "="*60)
print("ANÁLISIS COMPLETADO")
print("="*60)

