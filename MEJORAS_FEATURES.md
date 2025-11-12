# Propuestas de Mejora para Features del Modelo

## Análisis de Problemas Actuales

### 1. TrackTemp - Feature Constante (No aporta información)
- **Problema**: std=0.000, CV=0.000, range=[42.8, 42.8]
- **Causa**: Normalizado a valor constante en model.py línea 122
- **Impacto**: Feature inútil que ocupa espacio y puede confundir al modelo
- **Solución**: Eliminar completamente o permitir variación real

### 2. TyreLifeByCompound - Correlación Negativa Muy Baja
- **Problema**: corr=-0.0194 (prácticamente nula)
- **Causa**: Multiplicación simple `TyreLife * CompoundHardness` no captura la degradación real
- **Impacto**: Feature no informativa, desperdicio de capacidad del modelo
- **Solución**: Rediseñar la interacción compuesto-edad

### 3. TyreLife - Correlación Baja Globalmente
- **Problema**: corr=0.0692 global, pero mejor por compuesto (0.13-0.26)
- **Causa**: La degradación es no lineal y diferente por compuesto
- **Impacto**: El modelo no captura bien la degradación no lineal
- **Solución**: Agregar features polinómicas y normalización por compuesto

### 4. FuelLoad - Mejor Correlación pero Moderada
- **Problema**: corr=0.3645 (mejor pero aún moderada)
- **Causa**: Puede haber interacciones con otras features
- **Impacto**: Podría mejorarse con interacciones

## Mejoras Propuestas

### Mejora 1: Eliminar TrackTemp
**Razón**: No aporta información (constante)
**Acción**: Remover de features y del código

### Mejora 2: Agregar Features Polinómicas de TyreLife
**Razón**: La degradación es no lineal (más rápida al principio, luego se estabiliza)
**Acción**: Agregar `TyreLifeSquared` y `TyreLifeCubed`

### Mejora 3: Rediseñar TyreLifeByCompound
**Opciones**:
- **Opción A**: Usar degradación relativa por compuesto
  - `TyreLifeNormalized = TyreLife / MaxTyreLifeForCompound`
  - Captura mejor la degradación relativa
- **Opción B**: Usar función exponencial
  - `TyreLifeByCompound = TyreLife * exp(CompoundHardness / 3)`
  - Captura degradación acelerada en compuestos más blandos
- **Opción C**: Eliminar y dejar que el modelo aprenda la interacción
  - XGBoost puede aprender interacciones automáticamente con Compound (categórica)

### Mejora 4: Agregar Degradación Relativa por Compuesto
**Razón**: Los compuestos tienen rangos diferentes de vida útil
**Acción**: 
- `TyreLifeNormalized_SOFT = TyreLife / 25` (SOFT típicamente dura ~25 vueltas)
- `TyreLifeNormalized_MEDIUM = TyreLife / 35` (MEDIUM ~35 vueltas)
- `TyreLifeNormalized_HARD = TyreLife / 50` (HARD ~50 vueltas)

### Mejora 5: Agregar Interacciones FuelLoad-TyreLife
**Razón**: El efecto del combustible puede interactuar con la degradación
**Acción**: `FuelLoad * TyreLife` como feature adicional

## Implementación Recomendada

### Features Finales Propuestas:
1. **Compound** (categórica) - Mantener
2. **TyreLife** - Mantener
3. **TyreLifeSquared** - NUEVA (TyreLife²)
4. **TyreLifeCubed** - NUEVA (TyreLife³)
5. **TyreLifeNormalized** - NUEVA (TyreLife / max_typical_life_for_compound)
6. **FuelLoad** - Mantener
7. **FuelLoad * TyreLife** - NUEVA (interacción)
8. ~~**TyreLifeByCompound**~~ - ELIMINAR (no informativa)
9. ~~**TrackTemp**~~ - ELIMINAR (constante)

### Justificación:
- **TyreLifeSquared/Cubed**: Capturan degradación no lineal
- **TyreLifeNormalized**: Permite comparar degradación entre compuestos
- **FuelLoad * TyreLife**: Captura interacción combustible-degradación
- **Eliminar TyreLifeByCompound**: XGBoost puede aprender interacciones automáticamente con Compound
- **Eliminar TrackTemp**: No aporta información

## Resultados Esperados

1. **Mayor correlación** de features con el target
2. **Mejor R²** del modelo
3. **Mejor captura de degradación** no lineal
4. **Menos features redundantes** o inútiles

