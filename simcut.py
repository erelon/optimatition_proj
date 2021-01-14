import networkx
import numpy
import scipy.sparse
import scipy
# import yappi

#   This software is an implementation of the invention in US Patent 8929363
#   "Method and System for Image Segmentation".
#   The software computes an approximation to the minimum s-t cut using the
#   Simulation s-t cut algorithm.
#   The software applies to the partitioning of undirected graphs provided
#   two seed nodes "s" and "t".
from scipy.signal import convolve2d


class Circuit(networkx.Graph):
    def __init__(self,G=None, **attr):
        if G is None:
            G =super().__init__(**attr)
        self.G = G

    def matrix(self):
        edges = networkx.get_edge_attributes(self.G, 'capacity')
        size = len(self.G.nodes())
        rows = numpy.array(list(edges.keys()))[:, 0]
        columns = numpy.array(list(edges.keys()))[:, 1]
        weights = numpy.array(list(edges.values()))
        matrix = scipy.sparse.coo_matrix((weights, (rows, columns)), shape=(size, size)).tocsr()
        return matrix + matrix.transpose()

    def flow(self):
        vector = numpy.zeros(len(self.G.nodes()))
        nodes = networkx.get_node_attributes(self.G, 'flow')
        for (node, flow) in nodes.items():
            if (flow == 's'):
                vector[node] = 1
            if (flow == 't'):
                vector[node] = -1
        return vector

    def s(self):
        nodes = networkx.get_node_attributes(self.G, 'flow')
        for (node, flow) in nodes.items():
            if (flow == 's'):
                s = node
        return s

    def t(self):
        nodes = networkx.get_node_attributes(self.G, 'flow')
        for (node, flow) in nodes.items():
            if (flow == 't'):
                t = node
        return t


class Nonlinear:
    def __init__(self, weights):
        self.weights = weights

    def set_voltage(self, voltage):
        self.voltage = voltage

    def linearize(self):
        nodes = self.weights.shape[0]
        entries = self.weights.nnz
        # data = self.weights.tocoo().data
        rows = self.weights.tocoo().row
        columns = self.weights.tocoo().col
        ones = numpy.ones(entries)
        # negative_ones = -1.0 * numpy.ones(entries)
        positive = scipy.sparse.coo_matrix((ones, (range(0, entries), rows)), shape=(entries, nodes)).tocsr()
        negative = scipy.sparse.coo_matrix((ones, (range(0, entries), columns)), shape=(entries, nodes)).tocsr()
        subtract = positive - negative
        difference = subtract * self.voltage
        C = numpy.divide(self.weights.data, ones + numpy.absolute(difference))
        matrix = scipy.sparse.coo_matrix((C, (rows, columns)), shape=(nodes, nodes)).tocsr()
        return matrix


def sim_cut(graph: networkx.Graph, s, t):
    # yappi.set_clock_type("cpu")
    # yappi.start()
    G = Circuit(graph)

    g_len = len(graph) - 3
    G.G = networkx.relabel_nodes(G.G, {s: g_len + 1, t: g_len + 2},copy=False)

    s, t = g_len + 1, g_len + 2
    G.G.nodes[s]['flow'] = 's'
    G.G.nodes[t]['flow'] = 't'
    x = numpy.zeros(len(G.G.nodes()))
    ones = numpy.ones(len(G.G.nodes()))
    W = G.matrix()
    # rowsum = W * ones
    # D = scipy.sparse.diags(rowsum, 0)
    # F = D - W
    f = G.flow()
    s = G.s()
    t = G.t()
    N = Nonlinear(W)
    for iteration in range(0, 20):
        N.set_voltage(x)
        L = N.linearize()
        rowsum = L * ones
        D = scipy.sparse.diags(rowsum, 0)
        A = D - L
        A.data[A.indptr[s]:A.indptr[s + 1]] = 0
        A.data[A.indptr[t]:A.indptr[t + 1]] = 0
        A = A + scipy.sparse.diags(numpy.abs(f), 0)
        x = scipy.sparse.linalg.spsolve(A, 1000000.0 * f)
        # cut = F * segmentation
        # cut = numpy.multiply(cut, segmentation)
        # flow = numpy.sum(cut)
        # print(0.25 * flow)
    segmentation = numpy.sign(x).astype(int)[:-2]

    # yappi.get_func_stats().print_all()
    # yappi.stop()
    return segmentation

