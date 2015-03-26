import networkx as nx
import json
import matplotlib.pyplot as plt
import numpy as np

import copy, math, os, sys
import calendar, datetime

import threading

from pprint import pprint

from scipy.spatial import distance

from multiprocessing.pool import ThreadPool

from utility.mongodb_interface import MongoDBInterface

def getCurrentStampUTC():
    cur_utc_timestamp = calendar.timegm(datetime.datetime.utcnow().utctimetuple())
    return int(cur_utc_timestamp)

def _loadData(file):
    products_seq = []
    with open(file) as f:
        for line in f.readlines():
            d = line.strip().split(',')
            products_str = d[3]
            ids = products_str.split(';')
            products = []
            for product in ids:
                ids = product.split('/')[0:4]
                products.append((ids[0], ids[1], ids[2], ids[3]))
            products_seq.append(products)
    return products_seq

def _loadIndex(level):
    mapping = {1: 'B',
               2: 'C',
               3: 'D'}
    with open('data/%s_product_index.json' % mapping[level]) as f:
        reversed_product_index = json.loads(f.readlines()[0])

    product_index = {}
    for k, v in reversed_product_index.items():
        product_index[v] = k
    return product_index, reversed_product_index

def _analysis(level=3):
    # users never view any product more than once
    all_product_view = {}
    train_products = _loadData('data/trainingData.csv')
    # d_set = set()
    train_freq = {}
    for d_seq in train_products:
        for product in d_seq:
            train_freq[product[level]] = train_freq.get(product[level], 0) + 1

    test_products = _loadData('data/testData.csv')
    test_freq = {}
    for d_seq in test_products:
        for product in d_seq:
            test_freq[product[level]] = test_freq.get(product[level], 0) + 1

    for prod, freq in train_freq.items():
        all_product_view[prod] = all_product_view.get(prod, {})
        all_product_view[prod]['train'] = freq
        # all_product_view[prod]['']

    for prod, freq in test_freq.items():
        all_product_view[prod] = all_product_view.get(prod, {})
        all_product_view[prod]['test'] = freq

    for prod, freq in sorted(all_product_view.items(), key=lambda x: x[1].get('train', 0) + x[1].get('test', 0)):
        print prod, freq

    # generate weak rule

    # missing_set = set()
    # train_set = set(train_freq.keys())
    # for d_seq in test_products:
    #     for product in d_seq:
    #         if product[level] not in train_set:
    #             missing_set.add(product[level])
    # print missing_set
    # print len(missing_set)
    # print len(test_products)

def _connectedAnalysis(level=3):
    train_products = _loadData('data/trainingData.csv')

    G = nx.Graph()
    index = {}
    idx = 0
    for product_record in train_products:
        for product in product_record:
            name = product[level]
            if index.get(name, -1) == -1:
                index[name] = idx
                idx += 1
                # G.add_node(idx)
            G.add_node(name)

    for product_record in train_products:
        # if len(product_record) == 1:
        #     continue
        for i in xrange(len(product_record)):
            # for j in xrange(i+1, len(product_record)):
            for j in xrange(i+1, min(i+3, len(product_record))):
                G.add_edge(product_record[i][level],product_record[j][level])

    test_products = _loadData('data/testData.csv')
    for product_record in test_products:
        for product in product_record:
            name = product[level]
            if index.get(name, -1) == -1:
                index[name] = idx
                idx += 1
                # G.add_node(idx)
            G.add_node(name)

    for product_record in test_products:
        # if len(product_record) == 1:
        #     continue
        for i in xrange(len(product_record)):
            # for j in xrange(i+1, len(product_record)):
            for j in xrange(i+1, min(i+3, len(product_record))):
                G.add_edge(product_record[i][level],product_record[j][level])

    print nx.number_connected_components(G)
    for cc in nx.connected_components(G):
        print len(cc)

def offlineRandomWalk(N, start_node, W, transfer_to):
    c = 0.05
    res = randomWalkWithRestart(N, start_node, W, transfer_to, c)
    sparse = []
    for i in xrange(N):
        if res[i] > 1e-10:
            sparse.append((i, res[i]))

    mi = MongoDBInterface()
    mi.setDB('pakdd_comp')
    mi.setCollection('D_affinity_matrix')
    mi.saveDocument({'node_id': start_node, 'affinity_vector_sparse': sparse})

def randomWalkWithRestart(N, start_node, W, transfer_to, c=0.05):
    v = np.zeros(N)
    v[start_node] = 1.0

    k = 20
    while k > 0:
        updated_v = np.zeros(N)
        for j in xrange(N):
            if v[j] == 0:
                continue
            for i in transfer_to[j]:
                assert W[i][j] > 0 and W[i][j] <= 1
                updated_v[i] += W[i][j] * v[j] * (1-c)

        updated_v[start_node] += c
        l2_dis = distance.euclidean(v, updated_v)
        v = copy.deepcopy(updated_v)
        tot = sum(v)
        assert math.fabs(tot - c) < 1e-6 or math.fabs(tot - 1) < 1e-6
        k -= 1

        if l2_dis < 1e-4:
            break
    return v


def randomWalkWithRestartSparse(N, start_node, W, transfer_to, c=0.05):
    v = {start_node: 1.0}

    k = 20
    while k > 0:
        updated_v = {}
        if c > 0:
            updated_v[start_node] = c
        for j in v.keys():
            for i in transfer_to[j]:
                assert W[i][j] > 0 and W[i][j] <= 1
                updated_v[i] = updated_v.get(i, 0) + W[i][j] * v[j] * (1-c)

        key_set = set()
        for k in v.keys() + updated_v.keys():
            key_set.add(k)
        l2_dis = 0
        for key in key_set:
            l2_dis += (v.get(key, 0) - updated_v.get(key, 0))**2
        l2_dis = math.sqrt(l2_dis)
        tot = sum([v for k, v in updated_v.items()])
        v = copy.deepcopy(updated_v)
        assert math.fabs(tot - c) < 1e-6 or math.fabs(tot - 1) < 1e-6
        k -= 1

        if l2_dis < 1e-4:
            break
    return v

def preprocess(level=3, n_step=2):
    # users never view any product more than once
    all_records = _loadData('data/trainingData.csv') + _loadData('data/testData.csv')
    product_index, reversed_product_index = _loadIndex(level)
    G = nx.Graph()
    N = len(product_index)

    W = [{} for i in xrange(N)] # sparse representation
    transfer_to = [set() for i in xrange(N)]

    for user_record in all_records:
        if len(user_record) == 1:
            continue
        for product in user_record:
            pid = reversed_product_index[product[level]]
            G.add_node(pid)

        for i in xrange(len(user_record)):
            pid_1 = reversed_product_index[user_record[i][level]]
            for j in xrange(i+1, min(i+n_step, len(user_record))):
                pid_2 = reversed_product_index[user_record[j][level]]
                # G.add_edge(pid_1, pid_2)
                W[pid_2][pid_1] = W[pid_2].get(pid_1, 0) + 1.0
                transfer_to[pid_1].add(pid_2)

                W[pid_1][pid_2] = W[pid_1].get(pid_2, 0) + 1.0
                transfer_to[pid_2].add(pid_1)

    for i in xrange(N):
        transfer_to[i] = sorted(list(transfer_to[i]))

    # normalize to probability
    normalization_factors = [0 for i in xrange(N)]

    for j in xrange(N):
        for k, v in W[j].items():
            normalization_factors[k] += v

    for j in xrange(N):
        for k in W[j].keys():
            W[j][k] /= normalization_factors[k]

    # pprint(W)
    # pool = ThreadPool(processes=30)

    # st = getCurrentStampUTC()
    for start_node in xrange(N):
        # print transfer_to[start_node]
        # pool.apply_async(offlineRandomWalk, [N, start_node, W, transfer_to])
        offlineRandomWalk(N, start_node, W, transfer_to)
        # print start_node
    # pool.close()
    # pool.join()

    # nx.draw(G)
    # plt.savefig("test.png")
    # plt.close()

    # print 'Done'
    #
    # for i in xrange(N):
    #     normalization = sum(W[i])
    #     # if normalization > 0:
    #     #     for j in xrange(N):
    #     #         W[j][i] /= normalization
    # print sum(W[:,2])

    # for cc in nx.connected_components(G):
    #     print len(cc)



if __name__ == '__main__':
    # analysis(1)
    # _connectedAnalysis()
    preprocess(3)
