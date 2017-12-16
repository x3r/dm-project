import numpy as np


def cal_height(children, root, heights):
    if root not in children:
        heights[root] = 0
        return heights[root]
    else:
        left = cal_height(children, children[root][0], heights)
        right = cal_height(children, children[root][1], heights)
        heights[root] = 1 + max(left, right)
        return heights[root]


def get_children(children, parent, list, labels):
    if parent not in children:
        return
    for c in children[parent]:
        if c in labels:
            list.append(c)
            get_children(children, c, list, labels)
        else:
            get_children(children, c, list, labels)


def assign_labels(tree, o_deg):
    n_samples = len(tree) - len(o_deg)
    labels = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        labels[i] = -1
    idx = 0
    for i in range(len(o_deg)):
        if o_deg[i] == 0:
            for j in range(len(tree)):
                if tree[j]['parent'] == 100 + i:
                    labels[tree[j]['child']] = idx
            idx += 1
    return labels
