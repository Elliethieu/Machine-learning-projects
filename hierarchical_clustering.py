import scipy.spatial
import scipy.cluster
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.pyplot as plt

def load_data(filepath):
    data = csv.DictReader(open(filepath))
    data_list = list(data)
    return data_list

def calc_features(row):
    Attack = row['Attack']
    Sp_Attack = row['Sp. Atk']
    Speed = row['Speed']
    Defense = row['Defense']
    Sp_Def = row['Sp. Def']
    HP = row['HP']
    features = np.array([Attack, Sp_Attack, Speed,Defense, Sp_Def, HP ], dtype = np.int64)
    return features


def hac(features):
    n = len(features)

    X = np.asarray(features)
    X_squared = np.square(X)

    # creating the distance matrix D_squared
    G = X @ X.transpose()
    A = np.sum(X_squared, axis=1)
    B = np.ones((n,))
    D_squared = np.outer(A, B) + np.outer(B, A) - 2 * G

    # creat initial clusters
    clusters = []
    for i in range(n):
        cluster_i_dict = {'id': i, 'cluster': [i]}
        clusters.append(cluster_i_dict)

    Z = np.ones((n - 1, 4))

    for index in range(n - 1):  # iterate by row in Z
        # create the D_squared_current matrix with only distances between the current clusters
        m = len(clusters)
        D_squared_current = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                D_squared_current[i, j] = D_squared[clusters[i]['id'], clusters[j]['id']]

        # find the clusters to link in the D_squared_current matrix
        min_distance_D_squared_current = None

        for i in range(m):
            for j in range(i + 1, m):
                value = D_squared_current[i, j]
                if min_distance_D_squared_current == None:
                    min_distance_D_squared_current = value
                    cluster_pairs_list = [[i, j]]
                else:
                    if value < min_distance_D_squared_current:
                        min_distance_D_squared_current = value
                        cluster_pairs_list.clear()
                        cluster_pairs_list.append([i, j])
                    if value == min_distance_D_squared_current:
                        cluster_pairs_list.append([i, j])
        # cluster_pairs_list
        # min_distance_D_squared_current

        chosen_cluster_pair = cluster_pairs_list[0]
        linkage_distance = np.sqrt(min_distance_D_squared_current)

        first_cluster = clusters[chosen_cluster_pair[0]]
        second_cluster = clusters[chosen_cluster_pair[1]]
        merged_cluster = {'id': n + index, 'cluster': first_cluster['cluster'] + second_cluster['cluster']}

        # add new info into the D_squared matrix
        k = len(D_squared[0])
        D_squared_new_column = np.zeros((k, 1))
        for h in range(k):
            D_squared_new_column[h, 0] = max(D_squared[h, first_cluster['id']], D_squared[h, second_cluster['id']])

        D_squared_new_row = D_squared_new_column.transpose()
        D_squared_new_row = np.append(D_squared_new_row, [0])

        D_squared_new = np.hstack((D_squared, np.atleast_2d(D_squared_new_column)))
        D_squared_new = np.vstack([D_squared_new, D_squared_new_row])
        D_squared = D_squared_new

        # update the clusters list
        clusters.append(merged_cluster)
        clusters.remove(first_cluster)
        clusters.remove(second_cluster)

        Z[index, 0] = int(first_cluster['id'])
        Z[index, 1] = int(second_cluster['id'])
        Z[index, 2] = linkage_distance
        Z[index, 3] = len(merged_cluster['cluster'])

    return Z


def imshow_hac(Z, names):
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    axe = scipy.cluster.hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)
    fig.tight_layout()

    plt.show()

