import numpy as np
import matplotlib.pyplot as plt
import utils
from scipy.spatial.distance import euclidean
from DBCV import DBCV


def extract_cluster(condensed_tree, test_data):
    condensed_tree.sort(order='child_size')
    sorted_tree = condensed_tree[::-1]
    c_start = condensed_tree['parent'].min()
    c_end = condensed_tree['parent'].max()
    clusters_size = c_end - c_start + 1
    n_samples = len(test_data)
    i_degree = np.zeros((clusters_size), dtype=int)
    o_degree = np.zeros((clusters_size), dtype=int)
    graph = np.zeros((clusters_size, clusters_size), dtype=int)
    # print (condensed_tree)
    for i in sorted_tree:
        if i['child_size'] == 1:
            break
        graph[i['parent'] - n_samples, i['child'] - n_samples] = 1
        o_degree[i['parent'] - n_samples] += 1
        i_degree[i['child'] - n_samples] += 1

    labels = np.zeros(n_samples, dtype=int)
    idx = 1
    parent = {}
    heights = {}
    cluster_labels = {}
    print ('Assigning clusters...')
    for i in range(clusters_size):
        for j in range(len(condensed_tree)):
            if condensed_tree[j]['parent'] == n_samples + i and condensed_tree[j]['child'] < n_samples:
                labels[condensed_tree[j]['child']] = idx
                cluster_labels[n_samples + i] = idx
        idx += 1
        clist = []
        if o_degree[i] > 0:
            for j in range(len(condensed_tree)):
                if condensed_tree[j]['child_size'] > 1 and condensed_tree[j]['parent'] == n_samples + i:
                    clist.append(condensed_tree[j]['child'])
            parent[n_samples + i] = clist

    print ('Calculating height...')
    utils.cal_height(parent, c_start, heights)

    prev_score = DBCV(test_data, labels, dist_function=euclidean)
    max_height = heights[c_start]

    print ('Extracting clusters...')
    for h in range(1, max_height):
        for key in parent.keys():
            if heights[key] == h:
                c_list = []
                temp_labels = labels.copy()
                new_label = max(cluster_labels.values()) + 1
                utils.get_children(parent, key, c_list, cluster_labels)
                # print ('Children of {}: {}'.format(key, c_list))
                # collect all children and label them
                for c in c_list:
                    temp_labels[temp_labels == cluster_labels[c]] = new_label
                if key in cluster_labels:
                    temp_labels[temp_labels == cluster_labels[key]] = new_label
                # print (temp_labels)
                try:
                    new_score = DBCV(test_data, temp_labels, dist_function=euclidean)
                except:
                    new_score = 0
                print ("New score: {}, Prev score: {}".format(new_score, prev_score))
                if new_score > prev_score:
                    prev_score = new_score
                    labels = temp_labels
                    for c in clist:
                        del cluster_labels[c]
                    cluster_labels[key] = new_label
    print ('Final score:{} and clusters: {}'.format(prev_score, labels))
    plt.scatter(test_data.T[0], test_data.T[1], c=labels)
    plt.show()
