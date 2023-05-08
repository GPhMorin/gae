import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp

from tqdm import tqdm


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_balsac():
    with open('../../../data/parents.asc', 'r') as infile:
        lines = infile.readlines()[1:]
        data = {}

        for line in lines:
            ind, father, mother, _ = line.strip().split('\t')
            father = father if father != '0' else None
            mother = mother if mother != '0' else None
            data[ind] = (father, mother)

    def get_parents(ind):
        return data.get(ind, (None, None))

    G = nx.Graph()
    for ind in data.keys():
        G.add_node(ind)

    edges = []

    for ind in tqdm(data.keys()):
        father, mother = get_parents(ind)
        if father:
            edges.append((mother, father))
        if mother:
            edges.append((ind, mother))

    G.add_edges_from(edges)

    print(list(G.nodes())[:10])

    adj = nx.to_scipy_sparse_array(G)
    features = sp.csr_matrix((len(data.keys()), 1), dtype=float).tolil()
    return adj, features



def load_data(dataset):
    if dataset == 'balsac':
        adj, features = load_balsac()
        return adj, features

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
