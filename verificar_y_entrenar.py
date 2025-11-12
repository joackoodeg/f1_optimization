"""
Script para verificar si el modelo necesita reentrenarse
y reentrenarlo si es necesario.
"""
import os
import joblib
import sys

# Verificar features esperadas del modelo actual
models_dir = os.path.join('utils', 'XGBoost', 'models')
MODEL_FILENAME = "model_2025_gpMonza.pkl"
model_path = os.path.join(models_dir, MODEL_FILENAME)

# Features nuevas (optimizadas)
NEW_FEATURES = [
    "Compound",
    "TyreLife",
    "TyreLifeNormalized",
    "FuelLoad",
    "FuelLoad_TyreLife"
]

# Features antiguas (del modelo guardado)
OLD_FEATURES = [
    "Compound",
    "TyreLife",
    "TyreLifeByCompound",
    "FuelLoad",
    "TrackTemp"
]

if os.path.exists(model_path):
    print(f"[I] Modelo encontrado: {MODEL_FILENAME}")
    model_data = joblib.load(model_path)
    saved_features = model_data.get('features', [])
    
    print(f"\n[I] Features del modelo guardado ({len(saved_features)}):")
    for feat in saved_features:
        print(f"  - {feat}")
    
    print(f"\n[I] Features nuevas ({len(NEW_FEATURES)}):")
    for feat in NEW_FEATURES:
        print(f"  - {feat}")
    
    # Verificar si coinciden
    if set(saved_features) == set(NEW_FEATURES):
        print("\n[OK] ✅ El modelo ya tiene las features correctas!")
        print("     Puedes ejecutar ES.py sin problemas.")
    else:
        print("\n[!] ⚠️  El modelo tiene features diferentes.")
        print("     Necesitas reentrenar el modelo.")
        print("\n     Ejecuta: python model.py")
        print("     Luego podrás ejecutar ES.py")
else:
    print(f"[!] Modelo no encontrado: {model_path}")
    print("     Necesitas entrenar el modelo primero.")
    print("\n     Ejecuta: python model.py")

