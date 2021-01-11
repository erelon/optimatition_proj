from __future__ import division

import datetime
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import cv2
import networkx as nx
import numpy as np
import os
import argparse
from math import exp, pow

SIGMA = 20
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)
SOURCE, SINK = -2, -1

SF = 10
LOADSEEDS = False


def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def plantSeed(image, pathname):
    def drawLines(x, y, pixelType):
        code = BKGCODE
        if pixelType == OBJ:
            code = OBJCODE

        cv2.circle(seeds, (x // SF, y // SF), 10 // SF, code, -1)

    seeds = np.zeros(image.shape, dtype="uint8")

    f = open("seeds.txt", "r")
    size = image.shape[0]
    if size > 100: size = "none"
    line_str = pathname + "_" + str(size) + "\n"
    for x in f:
        if x == line_str:
            x = f.readline()
            while x != "#\n":
                pair = x.split()
                drawLines(int(pair[0]), int(pair[1]), OBJ)
                x = f.readline()
            x = f.readline()

            while x != "#\n":
                pair = x.split()
                drawLines(int(pair[0]), int(pair[1]), BKG)
                x = f.readline()
            break

    f.close()

    return seeds, None



def buildGraph(image, pathname):
    V = image.size + 2
    if (V > 1e5):
        raise MemoryError
    graph = nx.Graph()
    K, graph = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image, pathname)
    makeTLinks(graph, seeds, K)
    return graph, seededImage


BOUNDRAY_PENALTY_CONSTANT = -2 * pow(SIGMA, 2)


# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(pow(int(ip) - int(iq), 2) / BOUNDRAY_PENALTY_CONSTANT)
    return bp


def makeNLinks(graph, image):
    K = -float("inf")
    try:
        r, c, _ = image.shape
    except ValueError:
        r, c = image.shape

    zero_len_vec = np.expand_dims(np.zeros(image.shape[1]), axis=0)
    Dimage = np.vstack((image, zero_len_vec))[1:, :]

    zero_high_vec = np.expand_dims(np.zeros(image.shape[0]), axis=1)
    Rimage = np.hstack((image, zero_high_vec))[:, 1:]

    Dimage = 100 * np.exp(((image.astype("int16") - Dimage.astype("int16")) ** 2) / BOUNDRAY_PENALTY_CONSTANT)[:-1, :]
    Rimage = 100 * np.exp(((image.astype("int16") - Rimage.astype("int16")) ** 2) / BOUNDRAY_PENALTY_CONSTANT)[:, :-1]

    for i in range(r - 1):
        x = np.arange(c) + (i * c)
        s = Dimage[i, :]
        y = np.arange(c) + ((i + 1) * c)
        a = np.vstack((x, y, s)).T
        graph.add_weighted_edges_from(a, weight="capacity")

    for i in range(c - 1):
        x = np.arange(r) * (r) + i
        s = Rimage[:, i]
        y = np.arange(r) * (r) + (i + 1)
        a = np.vstack((x, y, s)).T
        graph.add_weighted_edges_from(a, weight="capacity")

    K = max(K, Dimage.max())
    K = max(K, Rimage.max())

    return K, graph


def org_makeNLinks(graph, image):
    K = -float("inf")
    try:
        r, c, _ = image.shape
    except ValueError:
        r, c = image.shape
    for i in range(r):
        edges = []
        for j in range(c):
            x = i * c + j
            if i + 1 < r:  # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                edges.append((x, y, {"capacity": bp}))
                K = max(K, bp)
            if j + 1 < c:  # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                edges.append((x, y, {"capacity": bp}))
                K = max(K, bp)
        graph.add_edges_from(edges)
    return K, graph


def makeTLinks(graph, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                graph.add_edge(SOURCE, x, capacity=K)
            elif seeds[i][j] == BKGCODE:
                graph.add_edge(SINK, x, capacity=K)


def displayCut(image, cuts):
    def colorPixel(i, j):
        image[int(i)][int(j)] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image


def imageSegmentation(imagefile, size=None, flag=False, algo=None, show=False):
    print(algo.__name__)
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    if size != (None, None):
        image = cv2.resize(image, size)
    else:
        q, w = image.shape
        m = min(q, w)
        image = cv2.resize(image, (m, m))

    while True:
        try:
            graph, seededImage = buildGraph(image, pathname)
            print(f"The resolution will be: {image.shape[0]}x{image.shape[1]}")
            break
        except MemoryError:
            size = (int(image.shape[0] * 0.75), int(image.shape[1] * 0.75))
            image = cv2.resize(image, size)
            continue

    global SOURCE, SINK

    if algo.__name__ == "shortest_augmenting_path" and flag is False:
        start_time = time.process_time()
        cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo, two_phase=False)
        end_time = time.process_time()
    elif algo.__name__ == "shortest_augmenting_path":
        start_time = time.process_time()
        cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo, two_phase=True)
        end_time = time.process_time()
    else:
        start_time = time.process_time()
        cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo)
        end_time = time.process_time()

    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    cuts = sorted(cutset)

    print("cuts:")
    print(cuts)
    print("Time to calculate:")
    print(end_time - start_time)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    if show:
        show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)
    return end_time - start_time, len(graph)

    parser = argparse.ArgumentParser()
    # parser.add_argument("imagefile")
    parser.add_argument("--size", "-s",
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()


if __name__ == "__main__":
    import networkx.algorithms.flow as algos

    total_times_for_algorithems = defaultdict(list)
    all_im_sizes_for_algorithem = defaultdict(list)
    sizes = [30, 100, None]
    for size in sizes:
        flag = False
        for algo in [algos.boykov_kolmogorov, algos.preflow_push, algos.shortest_augmenting_path
            , algos.shortest_augmenting_path,algos.dinitz, algos.edmonds_karp]:
            # algos.network_simplex ???
            run_data = {}
            for img_path in ["cat_easy.jpg", "cat_a.jpg", "cat_yoy.jpg", "cat_medium.jpg"]:
                run_time, im_size = imageSegmentation(img_path, (size, size), algo=algo, flag=flag)
                run_data[img_path.replace(".jpg", "")] = {"run_time": run_time, "im_size": im_size}
            total_times = [d["run_time"] for d in list(run_data.values())]

            if algo.__name__ == "shortest_augmenting_path" and flag is False:
                total_times_for_algorithems[algo.__name__ + " one phase"].append(sum(total_times))
                all_im_sizes_for_algorithem[algo.__name__ + " one phase"].append(
                    np.mean([d["im_size"] for d in list(run_data.values())]))
                flag = True
            elif algo.__name__ == "shortest_augmenting_path":
                total_times_for_algorithems[algo.__name__ + " two phase"].append(sum(total_times))
                all_im_sizes_for_algorithem[algo.__name__ + " two phase"].append(
                    np.mean([d["im_size"] for d in list(run_data.values())]))
            else:
                total_times_for_algorithems[algo.__name__].append(sum(total_times))
                all_im_sizes_for_algorithem[algo.__name__].append(
                    np.mean([d["im_size"] for d in list(run_data.values())]))

    for kt in total_times_for_algorithems:
        plt.plot(all_im_sizes_for_algorithem[kt], total_times_for_algorithems[kt], label=kt)

    plt.xlabel("Sizes")
    plt.ylabel("Time in seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Times per size for all algorithems")
    plt.show()
