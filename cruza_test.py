import random
from ES import CANT_VUELTAS, validar_estrategia, toolbox, creator

if __name__ == "__main__":
    ind1 = creator.Individual([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]) 
    ind1.PitStop =             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    ind2 = creator.Individual([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]])
    ind2.PitStop =             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    print("Padre 1: ", ind1[0])
    print("Pitstops: ", ind1.PitStop)
    print("Padre 2: ", ind2[0])
    print("Pitstops: ", ind2.PitStop)

    hijo1= toolbox.clone(ind1)
    hijo2= toolbox.clone(ind2)
    toolbox.mate(hijo1, hijo2)

    print("----- Después de la cruza -----")
    print("Hijo 1:  ", hijo1, len(hijo1[0]))
    print("Pitstops: ", hijo1.PitStop,len(hijo1.PitStop))
    print("TyreAge", hijo1.TyreAge)
    print("-----")
    print("Hijo 2:  ", hijo2,len(hijo2[0]))
    print("Pitstops: ", hijo2.PitStop, len(hijo2.PitStop))
    print("TyreAge", hijo2.TyreAge)

    print("----- Validación de mutaciones -----")
    print("Individuo original:  ", ind2, len(ind2[0]))
    print("Pitstops: ", ind2.PitStop,len(ind2.PitStop))

    mut = toolbox.clone(ind2)
    toolbox.compound_mutation(mut)
    print("Cambio de compuesto:  ", mut, len(mut[0]))
    print("Pitstops: ", mut.PitStop,len(mut.PitStop))
    print("TyreAge", mut.TyreAge)

    mut = toolbox.clone(ind2)
    toolbox.add_pit_mutation(mut)
    print("Agregar parada:  ", mut, len(mut[0]))
    print("Pitstops: ", mut.PitStop,len(mut.PitStop))
    print("TyreAge", mut.TyreAge)

    mut = toolbox.clone(ind2)
    toolbox.remove_pit_mutation(mut)
    print("Quitar parada:  ", mut, len(mut[0]))
    print("Pitstops: ", mut.PitStop,len(mut.PitStop))
    print("TyreAge", mut.TyreAge)
