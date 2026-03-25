# %% [markdown]
# ### Ejericicio Clustering
# 
# Suponer que representamos cada cluster de puntos de dos coordenadas numéricas mediante listas (o arrays).
# Definir una función que, dado un cluster y dado un punto adicional (con 2 coordenadas),  determine cuál punto del cluster ese el más cercano.

# %% [markdown]
# * Sabemos que un cluster es un grupo de puntos o puntos en un eje de coordenadas
# * Crear una función que reciba un punto (2, 7) y en base a un cluster de puntos, determinar que punto es el más cercano
# 
# #### Input
# $$[
#     (1.2, 2.3),
#     (4.5, 5.8),
#     (1.1, 2.1),
#     (9.0, 8.5),
#     (2.0, 3.1),
#     (4.8, 6.0),
#     (0.5, 0.8),
#     (8.7, 9.2),
#     (5.1, 5.5),
#     (1.5, 1.9)
# ]$$
# 
# $$(4.0, 5.0)$$
# 
# #### Output
# $$(4.5, 5.8)$$

# %%
import math


def dis_euclidiana(punto_a, punto_b):

    if punto_a is None or punto_b is None:
        print("La función debe recibir un punto A y B")
        return

    a_x = punto_a[0]
    a_y = punto_a[1]

    b_x = punto_b[0]
    b_y = punto_b[1]

    distancia = math.sqrt(((a_x - b_x) ** 2) + ((a_y - b_y) ** 2))

    return distancia

# %%
def obtener_punto_cercano_en_cluster(cluster, ref_punto):

    if cluster is None or ref_punto is None:
        print("La función debe recibir un cluster y un punto de referencia")
        return


    punto_min = cluster[0]
    for punto in cluster:
        if dis_euclidiana(punto, ref_punto) < dis_euclidiana(punto_min, ref_punto):
            punto_min = punto
    
    return punto_min

cluster = [
    (1.2, 2.3),
    (4.5, 5.8),
    (1.1, 2.1),
    (9.0, 8.5),
    (2.0, 3.1),
    (4.8, 6.0),
    (0.5, 0.8),
    (8.7, 9.2),
    (5.1, 5.5),
    (1.5, 1.9),
]

ref_punto = (5.3, 5.5)

punto_mas_cercano = obtener_punto_cercano_en_cluster(cluster, ref_punto)

punto_mas_cercano


