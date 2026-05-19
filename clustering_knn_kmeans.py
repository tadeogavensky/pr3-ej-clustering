import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent


def imprimir_titulo(titulo):
    print("\n" + "=" * 80)
    print(titulo)
    print("=" * 80)


def ejercicio_1():
    def dis_euclidiana(punto_a, punto_b):
        if punto_a is None or punto_b is None:
            raise ValueError("La funcion debe recibir un punto A y un punto B.")

        a_x = punto_a[0]
        a_y = punto_a[1]
        b_x = punto_b[0]
        b_y = punto_b[1]

        return math.sqrt(((a_x - b_x) ** 2) + ((a_y - b_y) ** 2))

    def obtener_punto_cercano_en_cluster(cluster, ref_punto):
        if cluster is None or ref_punto is None:
            raise ValueError("La funcion debe recibir un cluster y un punto de referencia.")

        punto_min = cluster[0]
        for punto in cluster:
            if dis_euclidiana(punto, ref_punto) < dis_euclidiana(punto_min, ref_punto):
                punto_min = punto

        return punto_min

    imprimir_titulo("EJERCICIO 1")
    print(
        "Enunciado: definir una funcion que, dado un cluster y un punto adicional, "
        "determine cual punto del cluster es el mas cercano."
    )

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
    ref_punto = (8.8, 9)
    punto_mas_cercano = obtener_punto_cercano_en_cluster(cluster, ref_punto)

    print(f"Cluster: {cluster}")
    print(f"Punto de referencia: {ref_punto}")
    print(f"Punto mas cercano: {punto_mas_cercano}")


def ejercicio_2():
    def calcular_dis_euclidiana(punto_a, punto_b):
        a_x = punto_a[0]
        a_y = punto_a[1]
        b_x = punto_b[0]
        b_y = punto_b[1]

        return math.sqrt(((a_x - b_x) ** 2) + ((a_y - b_y) ** 2))

    def calcular_centroide(cluster):
        prom_x = sum(punto[0] for punto in cluster) / len(cluster)
        prom_y = sum(punto[1] for punto in cluster) / len(cluster)
        return prom_x, prom_y

    def calcular_radio(cluster):
        centroide = calcular_centroide(cluster)
        dis_max = 0

        for x, y in cluster:
            distancia_actual = calcular_dis_euclidiana((x, y), centroide)
            if distancia_actual > dis_max:
                dis_max = distancia_actual

        return dis_max

    def calcular_diametro(cluster):
        dis_max = 0

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                distancia = calcular_dis_euclidiana(cluster[i], cluster[j])
                if distancia > dis_max:
                    dis_max = distancia

        return dis_max

    imprimir_titulo("EJERCICIO 2")
    print("Enunciado: calcular dado un cluster su radio y su diametro.")

    cluster = [
        (0.2, 9.8),
        (15.3, -4.2),
        (-7.5, 3.1),
        (22.0, 18.4),
        (-12.7, -9.6),
        (5.5, 25.3),
        (-18.9, 14.2),
        (30.1, -20.5),
        (8.8, -15.7),
        (-25.0, 5.0),
    ]

    centroide = calcular_centroide(cluster)
    radio = calcular_radio(cluster)
    diametro = calcular_diametro(cluster)

    print(f"Cluster: {cluster}")
    print(f"Centroide: {centroide}")
    print(f"Radio: {radio}")
    print(f"Diametro: {diametro}")


def ejercicio_3():
    def calcular_dis_euclidiana(punto_a, punto_b):
        return math.sqrt(
            ((punto_a[0] - punto_b[0]) ** 2) + ((punto_a[1] - punto_b[1]) ** 2)
        )

    def calc_diametro(cluster):
        max_distancia = 0

        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                distancia = calcular_dis_euclidiana(cluster[i], cluster[j])
                if distancia > max_distancia:
                    max_distancia = distancia

        return max_distancia

    def calc_varianza(diametros):
        promedio = sum(diametros) / len(diametros)
        suma_varianza = 0

        for diam in diametros:
            suma_varianza += (diam - promedio) ** 2

        return suma_varianza / len(diametros)

    def calc_kmeans(df, k_min, k_max):
        if df is None or k_min is None or k_max is None:
            raise ValueError("La funcion debe recibir un dataframe, un k minimo y un k maximo.")

        valores_k = list(range(k_min, k_max))
        mejor_k = valores_k[0]
        varianza_min = float("inf")

        for valor_k in valores_k:
            kmeans = KMeans(n_clusters=valor_k, random_state=42, n_init=10)
            kmeans.fit(df)

            clusters_labels = kmeans.labels_
            copia_df = df.copy()
            copia_df["cluster_id"] = clusters_labels

            diametros = []
            for cluster_id in set(clusters_labels):
                puntos = copia_df[copia_df["cluster_id"] == cluster_id][
                    ["latitude", "longitude"]
                ]
                diametro_cluster = calc_diametro(puntos.values.tolist())
                diametros.append(diametro_cluster)

            varianza_cluster = calc_varianza(diametros)

            if varianza_cluster < varianza_min:
                varianza_min = varianza_cluster
                mejor_k = valor_k

        return mejor_k

    imprimir_titulo("EJERCICIO 3")
    print(
        "Enunciado: calcular KMeans con distintos valores de k y determinar el mejor k "
        "segun la varianza de los diametros de los clusters."
    )

    df = pd.read_csv(BASE_DIR / "data.csv")
    df = df.drop(columns=["id"])
    mejor_k = calc_kmeans(df, k_min=2, k_max=10)

    print("DataFrame usado:")
    print(df)
    print(f"Mejor k: {mejor_k}")


def ejercicio_4():
    def dis_manhattan(x, y):
        v1 = np.array(x)
        v2 = np.array(y)
        return np.sum(np.abs(v1 - v2))

    def dis_euclidiana(x, y):
        v1 = np.array(x)
        v2 = np.array(y)
        return np.sqrt(np.sum((v1 - v2) ** 2))

    class KNN:
        def __init__(self, k):
            self.k = k

        def fit(self, X_train, Y_train):
            self.X_train = X_train
            self.Y_train = Y_train

        def predict_euclidian(self, single_X_test):
            distances = [dis_euclidiana(single_X_test, x) for x in self.X_train]
            k_labels = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.Y_train[i] for i in k_labels]
            resultado = Counter(k_nearest_labels).most_common(1)[0][0]
            return resultado.item() if hasattr(resultado, "item") else resultado

        def predict_euclidian_batch(self, X_test):
            return [self.predict_euclidian(x) for x in X_test]

        def predict_manhattan(self, single_X_test):
            distances = [dis_manhattan(single_X_test, x) for x in self.X_train]
            k_labels = np.argsort(distances)[: self.k]
            k_nearest_labels = [self.Y_train[i] for i in k_labels]
            resultado = Counter(k_nearest_labels).most_common(1)[0][0]
            return resultado.item() if hasattr(resultado, "item") else resultado

        def predict_manhattan_batch(self, X_test):
            return [self.predict_manhattan(x) for x in X_test]

    imprimir_titulo("EJERCICIO 4")
    print(
        "Enunciado: aplicarlo a tablas conocidas, modificarlo para que tome 5 vecinos "
        "con 25% de test y metrica Manhattan, y calcular la planta mas alejada "
        "de la primera en la tabla iris."
    )

    df = pd.read_csv(BASE_DIR / "clientes.csv")
    X = df[["edad", "gasto"]].values
    y = df["deudor"].values

    print("\nSubejercicio 1")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    knn = KNN(3)
    knn.fit(X_train, y_train)
    categoria = knn.predict_euclidian(X_test[0])
    categorias = knn.predict_euclidian_batch(X_test)
    print("Prediccion individual con distancia euclidiana:", categoria)
    print("Predicciones batch con distancia euclidiana:", categorias)

    print("\nSubejercicio 2")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    knn = KNN(5)
    knn.fit(X_train, y_train)
    categoria = knn.predict_manhattan(X_test[0])
    categorias = knn.predict_manhattan_batch(X_test)
    print("Prediccion individual con distancia Manhattan:", categoria)
    print("Predicciones batch con distancia Manhattan:", categorias)

    print("\nSubejercicio 3")
    data = load_iris()
    X_iris = data.data
    y_iris = data.target
    df_iris = pd.DataFrame(X_iris, columns=data.feature_names)
    df_iris["target"] = y_iris
    primera_planta = df_iris.iloc[0][:-1].values

    distances = [dis_euclidiana(primera_planta, x) for x in X_iris]
    indice_planta_mas_alejada = int(np.argmax(distances))
    max_distance = float(distances[indice_planta_mas_alejada])

    print(
        "La planta mas alejada de la primera es la:",
        indice_planta_mas_alejada,
        "con una distancia de:",
        max_distance,
    )


def main():
    ejercicio_1()
    ejercicio_2()
    ejercicio_3()
    ejercicio_4()


if __name__ == "__main__":
    main()
