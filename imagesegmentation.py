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


# Large when ip - iq < sigma, and small otherwise
def boundaryPenalty(ip, iq):
    bp = 100 * exp(- pow(int(ip) - int(iq), 2) / (2 * pow(SIGMA, 2)))
    return bp


def buildGraph(image, pathname):
    V = image.size + 2
    graph = np.zeros((V, V), dtype='int32')
    K = makeNLinks(graph, image)
    seeds, seededImage = plantSeed(image, pathname)
    makeTLinks(graph, seeds, K)
    return graph, seededImage


def makeNLinks(graph, image):
    K = -float("inf")
    r, c = image.shape
    for i in range(r):
        for j in range(c):
            x = i * c + j
            if i + 1 < r:  # pixel below
                y = (i + 1) * c + j
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
            if j + 1 < c:  # pixel to the right
                y = i * c + j + 1
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
    return K


def makeTLinks(graph, seeds, K):
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                # graph[x][source] = K
                graph[SOURCE][x] = K
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K
                # graph[sink][x] = K
            # else:
            #     graph[x][source] = LAMBDA * regionalPenalty(image[i][j], BKG)
            #     graph[x][sink]   = LAMBDA * regionalPenalty(image[i][j], OBJ)


def displayCut(image, cuts):
    def colorPixel(i, j):
        image[i][j] = CUTCOLOR

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
    graph, seededImage = buildGraph(image, pathname)

    g = nx.convert_matrix.from_numpy_array(graph)
    nx.classes.set_edge_attributes(g, nx.classes.get_edge_attributes(g, "weight"), name="capacity")
    global SOURCE,SINK
    SOURCE, SINK = -2, -1

    SOURCE += len(graph)
    SINK += len(graph)

    start_time = datetime.datetime.now()

    cuts = []
    cut_value, partition = nx.minimum_cut(g, SOURCE, SINK)
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, g[n]) for n in reachable):
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
