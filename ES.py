import os
import numpy as np
import pandas as pd
import random
from deap import base, creator, tools
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

# Flag para activar/desactivar la cruza (cambiar a False para desactivar)
ENABLE_CROSSOVER = False

def set_crossover(enable):
    """
    Activa o desactiva la cruza (crossover) en el algoritmo evolutivo.
    
    Args:
        enable (bool): True para activar la cruza, False para desactivarla
    
    Ejemplo:
        set_crossover(False)  # Desactiva la cruza
        set_crossover(True)   # Activa la cruza
    """
    global ENABLE_CROSSOVER
    ENABLE_CROSSOVER = enable
    print(f"Cruza {'ACTIVADA' if ENABLE_CROSSOVER else 'DESACTIVADA'}")

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

def create_individual():
    """
    Crea un individuo (estrategia) aleatorio.
    El modelo aprenderá implícitamente que mantener un neumático por mucho tiempo
    resulta en tiempos de vuelta peores.
    
    genTyreCompound: lista de compuestos por vuelta usando enteros
    Mapeo: 0=SOFT, 1=MEDIUM, 2=HARD
    """
    genTyreCompound = []
    PitStop = []
    TyreAge = []
    NumPitStop = 0

    # Mapeo de enteros a compuestos
    compound = random.randint(0, 2)
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
            new_compound = random.choice([c for c in [0,1,2] if c != compound])
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
    """Formatea la estrategia en una línea corta mostrando cada stint
    
    Mapeo: 0=SOFT, 1=MEDIUM, 2=HARD
    """
    compound_map = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    
    stints = []
    current_compound = ind[0][0]
    stint_length = 1
    
    for lap in range(1, len(ind[0])):
        if ind[0][lap] != current_compound:
            # Nuevo stint
            if current_compound not in compound_map:
                raise ValueError(f"Compuesto inválido: {current_compound}. Debe ser 0, 1 o 2")
            compound_name = compound_map[current_compound]
            stints.append(f"{compound_name}({stint_length})")
            current_compound = ind[0][lap]
            stint_length = 1
        else:
            stint_length += 1
    
    # Agregar el último stint
    if current_compound not in compound_map:
        raise ValueError(f"Compuesto inválido: {current_compound}. Debe ser 0, 1 o 2")
    compound_name = compound_map[current_compound]
    stints.append(f"{compound_name}({stint_length})")
    
    # Formato: "HARD(37) → SOFT(16) → HARD(5)"
    strategy_str = " → ".join(stints)
    return strategy_str

def debug_predictions(individual, max_laps_to_show=10):
    """
    Función de debugging para mostrar predicciones detalladas de una estrategia
    
    Mapeo: 0=SOFT, 1=MEDIUM, 2=HARD
    """
    compound_map = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    
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
        compound_int = individual[0][lap]
        if compound_int not in compound_map:
            raise ValueError(f"Compuesto inválido: {compound_int} en vuelta {lap}. Debe ser 0, 1 o 2")
        compound = compound_map[compound_int]  # Convertir a string
        tyre_life = individual.TyreAge[lap]
        
        try:
            lap_time = predict_lap_time(
                lap, compound, tyre_life, fuel_load,
                reference_conditions, model, features
            )
            
            # Obtener features usadas
            features_dict = create_features_for_lap(lap, compound, tyre_life, fuel_load, reference_conditions)
            
            features_str = f"T²={features_dict.get('TyreLifeSquared', 0):4d} T³={features_dict.get('TyreLifeCubed', 0):5d} "
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
    
    Mapeo de compuestos: 0=SOFT, 1=MEDIUM, 2=HARD
    """
    # Mapeo de enteros a nombres de compuestos
    compound_map = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    
    # Validar primero
    individual.Valid = validar_estrategia(individual)
    
    if not individual.Valid:
        return (1e6,)  # Penalización enorme si no es válida
    
    total_time = 0.0
    
    for lap in range(CANT_VUELTAS):
        # Calcular carga de combustible (decrece linealmente)
        fuel_load = 1.0 - (lap / CANT_VUELTAS)
        
        # Obtener datos de la vuelta
        compound_int = individual[0][lap]
        if compound_int not in compound_map:
            raise ValueError(f"Compuesto inválido: {compound_int} en vuelta {lap}. Debe ser 0, 1 o 2")
        compound = compound_map[compound_int]  # Convertir a string
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

    hijos = []
    
    # Solo realizar cruza si está habilitada
    if ENABLE_CROSSOVER:
        for _ in range(1, len(mejoresK)//2):
            p1, p2 = random.sample(mejoresK, 2)
            hijo1 = toolbox.clone(p1)
            hijo2 = toolbox.clone(p2) 
            del hijo1.fitness.values
            del hijo2.fitness.values
            
            if hijo1.NumPitStop == 0 or hijo2.NumPitStop == 0:  # Cruzar solo si ambos tienen pit stops
                continue

            toolbox.mate(hijo1, hijo2)
            
            if hijo1.Valid:
                hijos.append(hijo1)
            if hijo2.Valid:    
                hijos.append(hijo2)

    new_pop = mejoresK + mut1 + mut2 + mut3 + mut123 + hijos

    num_random = POP_SIZE - len(new_pop)
    for _ in range(num_random):
        new_pop.append(toolbox.individual())

    return new_pop

def cruza(ind1, ind2):
    #Teniendo dos padres dividirlos en dos mitades en donde ocurrio un pitstop
    #Agarrar la primera mitad de un padre y la segunda del otro y formar una nueva estrategia
    #Si se superponen en vueltas elegir aleatoreamente q domine uno
    #Si faltan vueltas extender el ultimo stint del padre 1
    padre1 = toolbox.clone(ind1)
    padre2 = toolbox.clone(ind2)
    
    pit_indices_1 = [i for i,v in enumerate(padre1.PitStop) if v == 1]
    idx_1 = random.choice(pit_indices_1)

    pit_indices_2 = [i for i,v in enumerate(padre2.PitStop) if v == 1]
    idx_2 = random.choice(pit_indices_2)

    #print("Puntos de cruza elegidos: Padre1 en vuelta", idx_1, ", Padre2 en vuelta", idx_2)

    # Intercambiar stints
    gen1 = padre1[0]
    gen2 = padre2[0]

    mitad1_padre1 = gen1[:idx_1]
    mitad2_padre1 = gen1[idx_1:]
    
    mitad1_padre2 = gen2[:idx_2]
    mitad2_padre2 = gen2[idx_2:]
    
    #Hijo 1
    new_gen = None
    new_pit_stops = None
    if  len(mitad1_padre1) + len(mitad2_padre2) > CANT_VUELTAS:
        # Superposición, elegir aleatoriamente qué mitad domina
        extra = len(mitad1_padre1) + len(mitad2_padre2) - CANT_VUELTAS 
        if random.random() < 0.5:
            new_gen = mitad1_padre1[:len(mitad1_padre1)-extra] + mitad2_padre2 
            new_pit_stops = padre1.PitStop[:len(mitad1_padre1)-extra] + padre2.PitStop[len(mitad1_padre1)-extra:]
            new_pit_stops[len(mitad1_padre1)-extra] = 1 #Creo que no es necesario ya que deberia ser 1 desde el padre2
        else:
            new_gen = mitad1_padre1 + mitad2_padre2[extra:]
            new_pit_stops = padre1.PitStop[:len(mitad1_padre1)] + padre2.PitStop[len(mitad1_padre1):]
            new_pit_stops[len(mitad1_padre1)] = 1
           
    else:
        # No hay superposición, extender el último stint del padre1 si faltan vueltas
        i=0
        new_pit_stops = padre1.PitStop[:len(mitad1_padre1)]
        extra = CANT_VUELTAS - (len(mitad1_padre1)+len(mitad2_padre2))
        while i < extra:
            mitad1_padre1.append(mitad1_padre1[-1])
            new_pit_stops.append(0)
            i+=1
        new_gen = mitad1_padre1 + mitad2_padre2 
        new_pit_stops += padre2.PitStop[len(mitad1_padre1):]

    ind1[0] = new_gen
    ind1.PitStop = new_pit_stops

    # Recalcular TyreAge, NumPitStop del inviduo 1
    tyre_age = []
    last_age = 0
    for i in range(len(new_gen)):
        if new_pit_stops[i] == 1:
            last_age = 0
            tyre_age.append(0)
        else:
            last_age += 1 if i > 0 else 0
            tyre_age.append(last_age)
    ind1.TyreAge = tyre_age
    ind1.NumPitStop = sum(new_pit_stops)
    ind1.Valid = validar_estrategia(ind1)
    
    #Hijo 2
    new_gen = None
    new_pit_stops = None
    if  len(mitad1_padre2) + len(mitad2_padre1) > CANT_VUELTAS:
        # Superposición, elegir aleatoriamente qué mitad domina
        extra = len(mitad1_padre2) + len(mitad2_padre1) - CANT_VUELTAS
        if random.random() < 0.5:
            new_gen = mitad1_padre2[:len(mitad1_padre2)-extra] + mitad2_padre1 
            new_pit_stops = padre2.PitStop[:len(mitad1_padre2)-extra] + padre1.PitStop[len(mitad1_padre2)-extra:]
            new_pit_stops[len(mitad1_padre2)-extra] = 1 #Creo que no es necesario ya que deberia ser 1 desde el padre1
        else:
            new_gen = mitad1_padre2 + mitad2_padre1[extra:]
            new_pit_stops = padre2.PitStop[:len(mitad1_padre2)] + padre1.PitStop[len(mitad1_padre2):]
            new_pit_stops[len(mitad1_padre2)] = 1
    else:
        # No hay superposición, extender el último stint del padre1 si faltan vueltas
        i=0
        new_pit_stops = padre2.PitStop[:len(mitad1_padre2)]
        Extra = CANT_VUELTAS - (len(mitad1_padre2)+len(mitad2_padre1))
        while i < Extra:
            mitad1_padre2.append(mitad1_padre2[-1])
            new_pit_stops.append(0)
            i+=1
        new_gen = mitad1_padre2 + mitad2_padre1 
        new_pit_stops += padre1.PitStop[len(mitad1_padre2):]

    ind2[0] = new_gen
    ind2.PitStop = new_pit_stops
    # Recalcular TyreAge, NumPitStop del inviduo 1
    tyre_age = []
    last_age = 0
    for i in range(len(new_gen)):
        if new_pit_stops[i] == 1:
            last_age = 0
            tyre_age.append(0)
        else:
            last_age += 1 if i > 0 else 0
            tyre_age.append(last_age)
    ind2.TyreAge = tyre_age
    ind2.NumPitStop = sum(new_pit_stops)
    ind2.Valid = validar_estrategia(ind2)



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
    opciones = [0,1,2]
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
    new_comp = random.choice([0,1,2])
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
    print(f"Cruza: {'ACTIVADA' if ENABLE_CROSSOVER else 'DESACTIVADA'}")
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
    print(f"\nSeleccionando los mejores {k} individuos por generación.\n")

    print("[3/3] Evolución en progreso...\n")
    
    # Evolución
    for gen in range(1, NGEN + 1):
        try:
            offspring = toolbox.select(pop, k)
            print("Cantidad de individuos en la nueva población:", len(offspring),"(deberian ser",POP_SIZE,")")
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
    compound_map = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}
    
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
                if current_compound not in compound_map:
                    raise ValueError(f"Compuesto inválido en mejor estrategia: {current_compound}. Debe ser 0, 1 o 2")
                compound_name = compound_map[current_compound]
                f.write(f"Stint {stint_number}: Vueltas {stint_start+1}-{lap} ({stint_length} vueltas) - {compound_name}\n")
                
                if lap < CANT_VUELTAS:
                    current_compound = best[0][lap]
                    stint_start = lap
                    stint_number += 1
    
    print(f"[OK] Estrategia guardada en: {output_file}\n")

