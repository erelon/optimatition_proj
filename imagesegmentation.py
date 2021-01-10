from __future__ import division

import datetime

import cv2
import networkx as nx
import numpy as np
import os
import sys
import argparse
from math import exp, pow
from augmentingPath import augmentingPath
from pushRelabel import pushRelabel
from boykovKolmogorov import boykovKolmogorov
from kargerMinCut import kargerMinCut

# np.set_printoptions(threshold=np.inf)
graphCutAlgo = {"ap": augmentingPath,
                "pr": pushRelabel,
                "bk": boykovKolmogorov,
                "ka": kargerMinCut}
SIGMA = 30
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)
SOURCE, SINK = -2, -1

SF = 10
LOADSEEDS = False

cat_easy_points = {OBJ: [(134, 76), (179, 101), (116, 238), (190, 202)],
                   BKG: [(38, 41), (46, 245), (264, 58), (154, 12)]}
cat_medium_points = {OBJ: [(122, 170), (71, 222), (196, 240), (173, 99), (63, 175), (131, 60), (145, 226)],
                     BKG: [(267, 22), (140, 25), (15, 60), (27, 280), (145, 285), (278, 250), (281, 139)]}
cat_yoy_points = {OBJ: [(149, 37), (89, 101), (86, 153), (93, 235), (184, 274), (216, 192), (192, 131), (174, 78)],
                  BKG: [(182, 7), (235, 59), (285, 171), (261, 281), (82, 284), (33, 208), (40, 135), (58, 40)]}
cat_a_points = {OBJ: [(69, 119), (96, 183), (134, 221), (158, 97), (150, 154)],
                BKG: [(29, 256), (29, 38), (193, 33), (264, 138), (262, 181), (243, 277)]}


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
    arr = eval(pathname + "_points");
    for key, value in arr.items():
        for pair in value:
            drawLines(pair[0], pair[1], key)

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
                # graph[x][source] = K
                # graph[SOURCE][x] = K
                graph.add_edge(SOURCE, x, capacity=K)
            elif seeds[i][j] == BKGCODE:
                # graph[x][SINK] = K
                graph.add_edge(SINK, x, capacity=K)
                # graph[sink][x] = K
            # else:
            #     graph[x][source] = LAMBDA * regionalPenalty(image[i][j], BKG)
            #     graph[x][sink]   = LAMBDA * regionalPenalty(image[i][j], OBJ)


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


def imageSegmentation(imagefile, size=(30, 30), algo="ff"):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, size)
    while True:
        try:
            graph, seededImage = buildGraph(image, pathname)
            print(f"The resolution will be: {image.shape[0]}x{image.shape[1]}")
            break
        except MemoryError:
            size = (int(image.shape[0] * 0.75), int(image.shape[1] * 0.75))
            image = cv2.resize(image, size)
            continue

    # nx.classes.set_edge_attributes(graph, nx.classes.get_edge_attributes(graph, "weight"), name="capacity")
    global SOURCE, SINK

    start_time = datetime.datetime.now()

    cuts = []
    cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK)
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    cuts = sorted(cutset)

    end_time = datetime.datetime.now()
    print("cuts:")
    print(cuts)
    print("Time to calculate:")
    print(end_time - start_time)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    show_image(image)
    savename = pathname + "cut.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)


def parseArgs():
    def algorithm(string):
        if string in graphCutAlgo:
            return string
        raise argparse.ArgumentTypeError(
            "Algorithm should be one of the following:", graphCutAlgo.keys())

    parser = argparse.ArgumentParser()
    # parser.add_argument("imagefile")
    parser.add_argument("--size", "-s",
                        default=30, type=int,
                        help="Defaults to 30x30")
    parser.add_argument("--algo", "-a", default="ap", type=algorithm)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    for img_path in ["cat_easy.jpg", "cat_a.jpg", "cat_yoy.jpg", "cat_medium.jpg"]:
        imageSegmentation(img_path, (args.size, args.size), args.algo)
