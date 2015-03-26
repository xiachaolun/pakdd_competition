#!/usr/bin/env python
import networkx as nx
import math
import csv
import random as rand
import sys
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import SpectralClustering, AffinityPropagation


def main():

    G = nx.Graph()  #let's create the graph first
    G.add_nodes_from(range(10))

    G.add_edge(1, 2, weight=0.5)
    G.add_edge(1, 3, weight=0.5)
    G.add_edge(1, 0, weight=0.5)
    G.add_edge(1, 4, weight=0.5)
    G.add_edge(3, 2, weight=0.5)
    G.add_edge(4, 2, weight=0.5)
    G.add_edge(0, 2, weight=0.5)
    G.add_edge(3, 4, weight=0.5)
    G.add_edge(3, 0, weight=0.5)
    G.add_edge(4, 0, weight=0.5)

    G.add_edge(5, 6, weight=2.5)
    G.add_edge(5, 7, weight=1.5)
    G.add_edge(5, 8, weight=3.5)
    # G.add_edge(5, 9, weight=4.5)
    G.add_edge(1, 5, weight=4.5)
    nx.draw(G)
    plt.savefig("before.png")
    plt.close()

    aff_mat = np.zeros((10, 10))

    for e in G.edges():
        aff_mat[e[0]][e[1]] = 1.0
        aff_mat[e[1]][e[0]] = 1.0
    print aff_mat

    r = AffinityPropagation(affinity='precomputed')
    r.fit(aff_mat)

    label_mapping = {}
    for i in xrange(len(r.labels_)):
        label_mapping[r.labels_[i]] = label_mapping.get(r.labels_[i], [])
        label_mapping[r.labels_[i]].append(i)

    print r.labels_

    labels = label_mapping.keys()
    for i in xrange(len(labels)):
        for j in xrange(i+1, len(labels)):
            for n1 in label_mapping[i]:
                for n2 in label_mapping[j]:
                    G.add_edge(n1, n2)
                    G.remove_edge(n1, n2)
                    # G.remove_edge(n2, n1)

    # nx.draw(G)
    # plt.savefig("after.png")
    # plt.close()

    print nx.connected_components(G)


if __name__ == "__main__":
    main()