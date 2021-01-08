"""
https://github.com/Abdallah-Elshamy/Karger-Minimum-cut-/blob/master/kargerMinCut.py

Created on Mon Jul 30 16:10:40 2018
@author: Abdallah-Elshamy
"""
from collections import defaultdict
from random import choice
from copy import deepcopy


def contract(graph):
    u = choice(list(graph.keys()))
    v = choice(graph[u])
    new_key = u + "-" + v
    graph[new_key] = graph[u] + graph[v]
    del graph[u]
    del graph[v]
    for key in graph.keys():
        copy = graph[key][:]
        if new_key == key:
            for item in copy:
                if item == u or item == v:
                    graph[key].remove(item)
        else:
            for item in copy:
                if item == u or item == v:
                    graph[key].remove(item)
                    graph[key].append(new_key)


def kargerMinCut(org_graph, s, t):
    graph = defaultdict(list)
    edges = set()

    for i, v in enumerate(org_graph, 1):
        for j, u in enumerate(v, 1):
            if u != 0 and frozenset([i, j]) not in edges:
                edges.add(frozenset([i, j]))
                graph[i].append({j: u})

    n = len(graph)
    minimum = n * (n - 1) // 2
    for i in range(n):
        copy = deepcopy(graph)
        while len(copy) > 2:
            contract(copy)
            minimum = min(minimum, len(list(copy.values())[0]))
    return minimum
