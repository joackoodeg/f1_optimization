import os
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools
import fastf1
import joblib
import warnings
from scoop import futures ################### pip install scoop
warnings.filterwarnings("ignore")

# Importar funciones y variables desde utils/ES/main.py
from utils.ES.main import (
    YEAR, GP, CANT_VUELTAS,
    POP_SIZE, NGEN, PMUT,
    PIT_STOP_TIME,
    reference_conditions,
    model, features, metadata,
    predict_lap_time,
    create_features_for_lap
)

# Flag de debugging (cambiar a True para activar debugging detallado)
DEBUG_MODEL = False

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def create_individual():
    """
    Crea un individuo (estrategia) aleatorio.
    El modelo aprenderá implícitamente que mantener un neumático por mucho tiempo
    resulta en tiempos de vuelta peores.
    genTyreCompound: lista de compuestos por vuelta (SOFT, MEDIUM, HARD)
    """
    genTyreCompound = []
    PitStop = []
    TyreAge = []
    NumPitStop = 0

    # Mapeo de enteros a compuestos
    compound_names = ["SOFT", "MEDIUM", "HARD"]
    compound = compound_names[random.randint(0, 2)]
    genTyreCompound.append(compound)
    TyreAge.append(0)
    PitStop.append(0)
    
    # Contador de vueltas en el stint actual
    stint_laps = 1

    for lap in range(1, CANT_VUELTAS):
        # Probabilidad de hacer pit stop (aumenta con la edad del neumático)
        # El modelo aprenderá que neumáticos más viejos son peores
        prob_pit = 0.05 + (stint_laps / CANT_VUELTAS) * 0.15  # 5% base, hasta 20%
        newTyre = np.random.choice([True, False], p=[prob_pit, 1 - prob_pit])
        
        if newTyre:
            # Elegir nuevo compuesto (evitar el mismo)
            new_compound = random.choice([c for c in compound_names if c != compound])
            compound = new_compound
            genTyreCompound.append(compound)
            TyreAge.append(0)
            PitStop.append(1)
            NumPitStop += 1
            stint_laps = 1
        else:
            genTyreCompound.append(compound)
            TyreAge.append(TyreAge[-1] + 1)
            PitStop.append(0)
            stint_laps += 1

    ind = creator.Individual([genTyreCompound])
    ind.PitStop = PitStop
    ind.TyreAge = TyreAge
    ind.NumPitStop = NumPitStop
    ind.Valid = validar_estrategia(ind)
    return ind

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def validar_estrategia(individual):
    """
    Valida que la estrategia cumpla con las reglas básicas de F1:
    - Al menos un pit stop
    - Al menos dos tipos de neumáticos diferentes
    """
    gen = individual[0]
    tipos_usados = set(gen)
    
    # Validación básica: al menos un pit stop y dos tipos de neumáticos
    if individual.NumPitStop == 0 or len(tipos_usados) < 2:
        return False
    
    return True

def format_strategy_short(ind):
    """Formatea la estrategia en una línea corta"""
    compound_counts = {}
    for compound in ind[0]:
        compound_counts[compound] = compound_counts.get(compound, 0) + 1
    
    # Ordenar compuestos por cantidad (mayor primero)
    sorted_compounds = sorted(compound_counts.items(), key=lambda x: x[1], reverse=True)
    # Formato: "1pits: SOFT(71) HARD(7)"
    strategy_str = f"{ind.NumPitStop}pits: " + " ".join([f"{compound}({count})" for compound, count in sorted_compounds])
    return strategy_str

def debug_predictions(individual, max_laps_to_show=10):
    """
    Función de debugging para mostrar predicciones detalladas de una estrategia
    """
    print("\n" + "="*60)
    print("DEBUG: Predicciones detalladas de la estrategia")
    print("="*60)
    
    strategy_str = format_strategy_short(individual)
    print(f"Estrategia: {strategy_str}")
    print(f"Tiempo total: {individual.fitness.values[0]:.3f}s\n")
    
    # Mostrar predicciones para algunas vueltas clave
    laps_to_show = [0] + list(range(10, min(CANT_VUELTAS, 60), 10)) + [CANT_VUELTAS-1]
    laps_to_show = sorted(set(laps_to_show))[:max_laps_to_show]
    
    print("Predicciones por vuelta (muestra):")
    print(f"{'Lap':<4} {'Compound':<8} {'TyreLife':<9} {'Fuel':<6} {'LapTime':<8} {'Features clave'}")
    print("-" * 80)
    
    for lap in laps_to_show:
        fuel_load = 1.0 - (lap / CANT_VUELTAS)
        compound = individual[0][lap]
        tyre_life = individual.TyreAge[lap]
        
        try:
            lap_time = predict_lap_time(
                lap, compound, tyre_life, fuel_load,
                reference_conditions, model, features
            )
            
            # Obtener features usadas
            features_dict = create_features_for_lap(lap, compound, tyre_life, fuel_load, reference_conditions)
            
            features_str = f"T²={features_dict['TyreLifeSquared']:4d} T³={features_dict['TyreLifeCubed']:5d} "
            features_str += f"T×C={features_dict['TyreLifeByCompound']:3.1f}"
            
            print(f"{lap:<4} {compound:<8} {tyre_life:<9} {fuel_load:<6.2f} {lap_time:<8.3f} {features_str}")
        except Exception as e:
            print(f"{lap:<4} {compound:<8} {tyre_life:<9} {fuel_load:<6.2f} ERROR: {e}")
    
    print("="*60 + "\n")

def func_aptitud(individual):
    """
    Función de aptitud: minimizar tiempo total de carrera
    Usa el modelo predictivo para estimar tiempos de vuelta.
    El modelo ya incluye la degradación de neumáticos en sus predicciones,
    por lo que aprenderá implícitamente que neumáticos más viejos son peores.
    No hay penalizaciones explícitas: el algoritmo debe aprender por sí solo
    qué estrategias son racionales basándose únicamente en las predicciones del modelo.
    """
    # Validar primero
    individual.Valid = validar_estrategia(individual)
    
    if not individual.Valid:
        return (1e6,)  # Penalización enorme si no es válida
    
    total_time = 0.0
    
    for lap in range(CANT_VUELTAS):
        # Calcular carga de combustible (decrece linealmente)
        fuel_load = 1.0 - (lap / CANT_VUELTAS)
        
        # Obtener datos de la vuelta
        compound = individual[0][lap]
        tyre_life = individual.TyreAge[lap]
        is_pit = individual.PitStop[lap]
        
        # Predecir tiempo de vuelta
        # El modelo ya incluye la degradación de neumáticos (TyreLife, TyreLifeSquared, 
        # TyreLifeCubed, TyreLifeByCompound) por lo que aprenderá implícitamente que 
        # neumáticos más viejos son peores, y podrá aprender diferentes tasas de degradación
        # por compuesto usando la feature categórica Compound y TyreLifeByCompound.
        try:
            lap_time = predict_lap_time(
                lap, compound, tyre_life, fuel_load,
                reference_conditions, model, features
            )
            # Verificar si el tiempo es válido (no NaN ni infinito)
            if np.isnan(lap_time) or np.isinf(lap_time):
                lap_time = 90.0  # Tiempo por defecto si es inválido
            total_time += lap_time
        except Exception as e:
            # Si hay error en predicción, usar penalización
            total_time += 90.0  # Tiempo por defecto
        
        # Añadir tiempo de pit stop si corresponde
        if is_pit == 1:
            total_time += PIT_STOP_TIME
    
    return (total_time,)

toolbox.register("evaluate", func_aptitud)

# ============================================================
# OPERADORES EVOLUTIVOS
# ============================================================

def seleccion(population, k):
    """Selección + mutaciones"""
    sorted_pop = sorted(population, key=lambda ind: ind.fitness.values[0])
    mejoresK = sorted_pop[:k]

    mut1 = []
    mut2 = []
    mut3 = []
    mut123 = []

    for ind in mejoresK:
        if random.random() < PMUT:
            ind_aux = toolbox.clone(ind)
            toolbox.compound_mutation(ind_aux)
            del ind_aux.fitness.values
            mut1.append(ind_aux)

        if random.random() < PMUT and ind.NumPitStop > 1:
            ind_aux = toolbox.clone(ind)
            toolbox.remove_pit_mutation(ind_aux)
            del ind_aux.fitness.values
            mut2.append(ind_aux)

        if random.random() < PMUT:
            ind_aux = toolbox.clone(ind)
            toolbox.add_pit_mutation(ind_aux)
            del ind_aux.fitness.values
            mut3.append(ind_aux)
        
        if random.random() < PMUT:
            ind_aux = toolbox.clone(ind)
            toolbox.add_pit_mutation(ind_aux)
            toolbox.remove_pit_mutation(ind_aux)
            toolbox.compound_mutation(ind_aux)
            del ind_aux.fitness.values
            mut123.append(ind_aux)

    new_pop = mejoresK + mut1 + mut2 + mut3 + mut123

    #for _ in range(1,len(mejoresK)//2):
    #    toolbox.mate(random.choice(mejoresK), random.choice(mejoresK)) #######################

    num_random = POP_SIZE - len(new_pop)
    for _ in range(num_random):
        new_pop.append(toolbox.individual())

    return new_pop

def cruza(ind1, ind2):
    
    pass

toolbox.register("mate", cruza)

def compound_mutation(individual):
    """Cambia el compuesto de un stint completo.
    El modelo aprenderá implícitamente qué compuestos funcionan mejor
    para diferentes longitudes de stint."""
    gen = individual[0]
    n = len(gen)
    if n == 0:
        return
    
    idx = random.randint(0, n - 1)
    comp0 = gen[idx]

    # Buscar inicio y fin del stint
    start = idx
    while start > 0 and gen[start - 1] == comp0:
        start -= 1
    end = idx
    while end < n - 1 and gen[end + 1] == comp0:
        end += 1

    # Elegir nuevo compuesto (cualquiera excepto el actual)
    opciones = ["SOFT", "MEDIUM", "HARD"]
    opciones.remove(comp0)
    new_comp = random.choice(opciones)
    
    # Aplicar cambio
    for j in range(start, end + 1):
        gen[j] = new_comp

    # Recomputar atributos
    pits = individual.PitStop
    tyre_age = []
    last_age = 0
    for i in range(n):
        if pits[i] == 1:
            last_age = 0
            tyre_age.append(0)
        else:
            last_age += 1 if i > 0 else 0
            tyre_age.append(last_age)
    individual.TyreAge = tyre_age
    individual.NumPitStop = sum(pits)
    individual.Valid = validar_estrategia(individual)

def remove_pit_mutation(individual):
    """Elimina un pit stop aleatorio.
    El modelo aprenderá implícitamente si el stint extendido resulta
    en tiempos de vuelta peores debido a la degradación."""
    pits = individual.PitStop
    gen = individual[0]
    pit_indices = [i for i, p in enumerate(pits) if p == 1]
    
    if not pit_indices or len(pit_indices) <= 1:
        return
    
    rem_idx = random.choice(pit_indices)
    
    # Calcular longitud del stint extendido antes de hacer el cambio
    comp_prev = gen[rem_idx-1]
    stint_start = rem_idx - 1
    while stint_start > 0 and gen[stint_start - 1] == comp_prev:
        stint_start -= 1
    
    end = len(gen)
    for j in range(rem_idx+1, len(pits)):
        if pits[j] == 1:
            end = j
            break
    
    pits[rem_idx] = 0

    # Extender compuesto previo
    for j in range(rem_idx, end):
        gen[j] = comp_prev
    
    # Recomputar
    tyre_age = []
    last_age = 0
    for i in range(len(gen)):
        if pits[i] == 1:
            last_age = 0
            tyre_age.append(0)
        else:
            last_age += 1 if i > 0 else 0
            tyre_age.append(last_age)
    individual.TyreAge = tyre_age
    individual.NumPitStop = sum(pits)
    individual.Valid = validar_estrategia(individual)

def add_pit_mutation(individual):
    """Añade un pit stop en una vuelta aleatoria"""
    gen = individual[0]
    pits = individual.PitStop
    n = len(gen)
    
    if n <= 1:
        return
    
    j = random.randint(1, n-1)
    if pits[j] == 1:
        return
    
    pits[j] = 1
    new_comp = random.choice(["SOFT", "MEDIUM", "HARD"])
    gen[j] = new_comp

    # Buscar fin del stint
    end = n
    for k in range(j+1, n):
        if pits[k] == 1:
            end = k
            break
    
    for k in range(j+1, end):
        gen[k] = new_comp

    # Recomputar
    tyre_age = []
    last_age = 0
    for i in range(len(gen)):
        if pits[i] == 1:
            last_age = 0
            tyre_age.append(0)
        else:
            last_age += 1 if i > 0 else 0
            tyre_age.append(last_age)
    individual.TyreAge = tyre_age
    individual.NumPitStop = sum(pits)
    individual.Valid = validar_estrategia(individual)

# Registrar operadores
toolbox.register("select", seleccion)
toolbox.register("compound_mutation", compound_mutation)
toolbox.register("remove_pit_mutation", remove_pit_mutation)
toolbox.register("add_pit_mutation", add_pit_mutation)

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("INICIANDO OPTIMIZACIÓN DE ESTRATEGIA")
    print("="*60)
    print(f"Población: {POP_SIZE} | Generaciones: {NGEN}")
    print("="*60 + "\n")
    
    #Registrar el map de scoop para paralelización
    toolbox.register("map", futures.map)

    # Crear población inicial
    print("[1/3] Creando población inicial...")
    pop = toolbox.population(n=POP_SIZE)

    # Evaluar población inicial
    print("[2/3] Evaluando población inicial...")
    #for ind in pop:
    #    ind.fitness.values = toolbox.evaluate(ind)
    
    fitness_ini = map(toolbox.evaluate, pop) #Evaluar en paralelo
    for ind, fit in zip(pop, fitness_ini):
        ind.fitness.values = fit

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    k = POP_SIZE // 6 #Número de mejores individuos a seleccionar

    print("[3/3] Evolución en progreso...\n")
    
    # Evolución
    for gen in range(1, NGEN + 1):
        try:
            offspring = toolbox.select(pop, k)
            pop[:] = offspring

            #Evaluar individuos nuevos
            invalid_ind = [ind for ind in pop if not ind.fitness.valid]
            fitness_valid = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness_valid):
                ind.fitness.values = fit

            record = stats.compile(pop)
            
            # Obtener top 5 mejores individuos de esta generación
            # Ordenar población por fitness y obtener únicos por estrategia
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
            unique_ind = []
            seen_strategies = set()
            
            for ind in sorted_pop:
                # Crear un hash de la estrategia para verificar unicidad
                strategy_hash = tuple(ind[0])  # La estrategia es la lista de compuestos
                if strategy_hash not in seen_strategies:
                    unique_ind.append(ind)
                    seen_strategies.add(strategy_hash)
                if len(unique_ind) >= 5:
                    break
            
            top5_ind = unique_ind[:5]
            
            # Convertir tiempo a minutos:segundos
            min_time = record['min']
            min_minutes = int(min_time // 60)
            min_seconds = min_time % 60
            
            avg_time = record['avg']
            avg_minutes = int(avg_time // 60)
            avg_seconds = avg_time % 60
            
            # Mostrar cada generación con top 5
            print(f"Gen {gen:3d}/{NGEN}: min={min_minutes}:{min_seconds:05.2f} avg={avg_minutes}:{avg_seconds:05.2f}")
            for rank, ind in enumerate(top5_ind, 1):
                strategy_str = format_strategy_short(ind)
                total_time = ind.fitness.values[0]
                total_minutes = int(total_time // 60)
                total_seconds = total_time % 60
                # Mostrar con más precisión (3 decimales) para ver diferencias
                print(f"  #{rank}: {total_minutes}:{total_seconds:06.3f} ({total_time:.3f}s) | {strategy_str}")
            
            # Debugging opcional para la mejor estrategia de la primera generación
            if DEBUG_MODEL and gen == 1 and len(top5_ind) > 0:
                debug_predictions(top5_ind[0])
        except Exception as e:
            print(f"\n[!] Error en generación {gen}: {str(e)}")
            print(f"[!] Continuando con la población actual...")
            # No romper el loop, continuar con la siguiente generación
            continue
    
    print(f"\n[OK] Completadas {NGEN} generaciones\n")

    # Obtener la mejor estrategia
    best = tools.selBest(pop, 1)[0]
    total_time = best.fitness.values[0]
    total_minutes = int(total_time // 60)
    total_seconds = total_time % 60
    
    print("\n" + "="*60)
    print("OPTIMIZACIÓN COMPLETADA")
    print("="*60 + "\n")
    
    # Guardar estrategia
    output_file = f"best_strategy_{YEAR}_{GP}.txt"
    with open(output_file, 'w') as f:
        f.write(f"MEJOR ESTRATEGIA - {YEAR} {GP} ({metadata['circuit']})\n")
        f.write("="*60 + "\n\n")
        f.write(f"Tiempo total estimado: {total_minutes}:{total_seconds:05.2f}\n")
        f.write(f"Número de pit stops: {best.NumPitStop}\n\n")
        f.write("ESTRATEGIA POR STINTS:\n")
        f.write("-" * 60 + "\n")
        
        current_compound = best[0][0]
        stint_start = 0
        stint_number = 1
        
        for lap in range(1, CANT_VUELTAS + 1):
            if lap == CANT_VUELTAS or best[0][lap] != current_compound:
                stint_length = lap - stint_start
                f.write(f"Stint {stint_number}: Vueltas {stint_start+1}-{lap} ({stint_length} vueltas) - {current_compound}\n")
                
                if lap < CANT_VUELTAS:
                    current_compound = best[0][lap]
                    stint_start = lap
                    stint_number += 1
    
    print(f"[OK] Estrategia guardada en: {output_file}\n")

