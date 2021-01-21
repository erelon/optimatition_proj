from __future__ import division

import datetime
import pickle
import time
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
import cv2
import networkx as nx
import numpy as np
import os
import argparse
from math import exp, pow

import scipy

from simcut import sim_cut

SIGMA = 20
# LAMBDA = 1
OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"

CUTCOLOR = (0, 0, 255)
SOURCE, SINK = -2, -1

# plant seeds option
MANUAL = False
# plant seeds only ones for every image size and type
MANUAL_FIRST = False

SF = 10
LOADSEEDS = False


def show_image(image):
    windowname = "Segmentation"
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow(windowname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plantSeed_manual(image, m):
    first = MANUAL_FIRST
    if first:
        f = open("seeds1.txt", 'a')
        shape_str = 'none' if  image.shape[0] > 100 else str(image.shape[0])
        f.write(m.split('.')[0]+"_"+shape_str+"\n")


    def drawLines(x, y, pixelType):
        if first:
            f.write(str(x)+' '+str(y)+"\n")
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def onMouse(event, x, y, flags, pixelType):

        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drawLines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paintSeeds(pixelType):
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowname, 500, 500)
        cv2.setMouseCallback(windowname, onMouse, pixelType)
        while (1):
            cv2.imshow(windowname, image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if first:
            f.write("#\n")
        cv2.destroyAllWindows()

    seeds = np.zeros(image.shape, dtype="uint8")
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)

    radius = 10
    thickness = -1  # fill the whole circle
    global drawing
    drawing = False

    paintSeeds(OBJ)
    paintSeeds(BKG)
    if first:
        f.close()
    return seeds, image


def plantSeed(image, pathname):
    seeds_file = "seeds.txt"
    if MANUAL:
        seeds_file = "seeds1.txt"

    def drawLines(x, y, pixelType):
        code = BKGCODE
        if pixelType == OBJ:
            code = OBJCODE

        cv2.circle(seeds, (x // SF, y // SF), 10 // SF, code, -1)

    seeds = np.zeros(image.shape, dtype="uint8")

    f = open(seeds_file, "r")
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
#


def buildGraph(image, pathname, sigma=30):
    V = image.size + 2
    if (V > 1e5):
        raise MemoryError
    graph = nx.Graph()
    K, graph = makeNLinks(graph, image, sigma=sigma)
    if MANUAL_FIRST and MANUAL:
        seeds, seededImage = plantSeed_manual(image, pathname)
    else:
        seeds, seededImage = plantSeed(image, pathname)

    makeTLinks(graph, seeds, K)

    # # Fast way to save the seeds on the image
    # kk = []
    # for x in seeds.flatten():
    #     if x == 0:
    #         kk.append((0, 0, 0, 0))
    #     if x == 1:
    #         kk.append((255, 0, 0, 150))
    #     if x == 2:
    #         kk.append((0, 255, 0, 150))
    # k1 = np.array(kk)
    # plt.imshow(image, cmap="gray")
    # plt.imshow(k1.reshape(image.shape[0], image.shape[0], 4))
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(f"seeded_cats/{pathname}_{image.shape[0]}.png", bbox_inches='tight', pad_inches=0)
    return graph, seededImage


def makeNLinks(graph, image, sigma=30):
    BOUNDRAY_PENALTY_CONSTANT = -2 * pow(sigma, 2)

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


# Large when ip - iq < sigma, and small otherwise
# def boundaryPenalty(ip, iq):
#     bp = 100 * exp(pow(int(ip) - int(iq), 2) / BOUNDRAY_PENALTY_CONSTANT)
#     return bp

# def org_makeNLinks(graph, image):
#     K = -float("inf")
#     try:
#         r, c, _ = image.shape
#     except ValueError:
#         r, c = image.shape
#     for i in range(r):
#         edges = []
#         for j in range(c):
#             x = i * c + j
#             if i + 1 < r:  # pixel below
#                 y = (i + 1) * c + j
#                 bp = boundaryPenalty(image[i][j], image[i + 1][j])
#                 edges.append((x, y, {"capacity": bp}))
#                 K = max(K, bp)
#             if j + 1 < c:  # pixel to the right
#                 y = i * c + j + 1
#                 bp = boundaryPenalty(image[i][j], image[i][j + 1])
#                 edges.append((x, y, {"capacity": bp}))
#                 K = max(K, bp)
#         graph.add_edges_from(edges)
#     return K, graph


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


def create_graph_from_img(imagefile, size=None):
    pathname = os.path.splitext(imagefile)[0]
    image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    if size != (None, None):
        image = cv2.resize(image, size)
    else:
        q, w = image.shape
        m = min(q, w)
        image = cv2.resize(image, (m, m))

    sigma = SIGMA
    while True:
        try:
            graph, seededImage = buildGraph(image, pathname, sigma=sigma)
            print(f"The resolution will be: {image.shape[0]}x{image.shape[1]}")
            size = (int(image.shape[0]), int(image.shape[1]))
            break
        except MemoryError:
            size = (int(image.shape[0] * 0.75), int(image.shape[1] * 0.75))
            image = cv2.resize(image, size)
            continue
    return graph, image, size, pathname


def imageSegmentation(graph, image, size, pathname, flag=False, algo=None, algo_name=None, show=False):
    print(algo_name)

    sigma = SIGMA
    global SOURCE, SINK
    found_good_sigma = False
    while not found_good_sigma:
        if algo_name == "shortest_augmenting_path" and flag is False:
            start_time = time.process_time()
            cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo,
                                                  two_phase=False)
            end_time = time.process_time()

        elif algo_name == "shortest_augmenting_path":
            start_time = time.process_time()
            cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo,
                                                  two_phase=True)
            end_time = time.process_time()
        elif "sim_cut" in algo_name:
            start_time = time.time()
            segmentation = algo(graph, SOURCE, SINK)
            end_time = time.time()
            print(end_time - start_time)
            # end of calculation - make the data look like the other algorithems
            segmentation = (segmentation.reshape(size))
            imax = (scipy.ndimage.maximum_filter(segmentation, size=3) != segmentation)
            # keep only pixels of original image at borders
            edges = np.where(imax, 1, -1).reshape(size[0] ** 2)
            cuts = []
            for node in np.where(edges > 0)[0]:
                if node < (len(graph) - 2):
                    cuts.append((node, node))
        else:
            start_time = time.process_time()
            cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo)
            end_time = time.process_time()

        try:
            reachable, non_reachable = partition
            cutset_clean = set()
            for u, nbrs in ((n, graph[n]) for n in reachable):
                cutset_clean.update((u, v) for v in nbrs if v in non_reachable)
            if len(cutset_clean) == 0:
                sigma += 10
                print(f"trying another sigma. sigma={sigma}")
                if sigma > 100:
                    found_good_sigma = True
                    cuts = sorted(cutset_clean)
                    print("Can't find a good sigma")
                graph, seededImage = buildGraph(image, pathname, sigma=sigma)
            else:
                found_good_sigma = True
                cuts = sorted(cutset_clean)
        except:
            break
    print("cuts:")
    print(cuts)
    print("Time to calculate:")
    print(end_time - start_time)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    if show:
        show_image(image)
    os.makedirs(f"{pathname} results", exist_ok=True)
    savename = f"{pathname} results/ {algo_name}_{size}_result.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)
    return end_time - start_time, len(graph)


if __name__ == "__main__":
    import networkx.algorithms.flow as algos


    def dd():
        return defaultdict(dict)


    all_runs_data = defaultdict(dd)
    try:
        with open("pickled_data", "rb") as f:
            all_runs_data = pickle.load(f)
            f.close()
    except:
        pass
    if len(sys.argv) > 1:
        MANUAL = True
    target_sizes = [30, 100, None]
    for size in target_sizes:
        flag = False
        MANUAL_FIRST = True
        for algo in [algos.preflow_push,
                     sim_cut]:  # sim_cut, algos.boykov_kolmogorov, algos.preflow_push,algos.shortest_augmenting_path,algos.shortest_augmenting_path]:
            # , algos.dinitz, algos.edmonds_karp,
            # ]:

            if algo.__name__ == "shortest_augmenting_path" and flag is False:
                algo_name = algo.__name__ + " one phase"
            elif algo.__name__ == "shortest_augmenting_path" and flag is True:
                algo_name = algo.__name__ + " two phase"
            elif algo.__name__ == "sim_cut":
                algo_name = "sim_cut_20_iter_better_implement"
            else:
                algo_name = algo.__name__

            run_data = {}
            for img_path in ["cat_easy.jpg", "cat_a.jpg", "cat_yoy.jpg", "cat_medium.jpg"]:
                graph, image, size_, pathname = create_graph_from_img(img_path, size=(size, size))
                try:
                    all_runs_data[algo_name][len(graph)][img_path.replace(".jpg", "")]
                    # This data was collected allready
                    continue
                except:
                    pass
                run_time, im_size = imageSegmentation(graph, image, size_, pathname, algo=algo, algo_name=algo_name,
                                                      flag=flag, show=False)
                all_runs_data[algo_name][len(graph)][img_path.replace(".jpg", "")] = run_time

            if algo.__name__ == "shortest_augmenting_path" and flag is False:
                flag = True

            with open("pickled_data", "wb") as f:
                pickle.dump(all_runs_data, f)
                f.close()
            MANUAL_FIRST = False

    index_of_free_size = target_sizes.index(None)
    for algo_name, D_1 in zip(all_runs_data.keys(), all_runs_data.values()):
        all_avg_times_of_all_sizes = []
        sizes = []
        for size, D_2 in zip(D_1.keys(), D_1.values()):
            all_imgs_run_time = []
            sizes.append(size)
            for img, run_time in zip(D_2.keys(), D_2.values()):
                all_imgs_run_time.append(run_time)
            avg_time_for_all_imgs = np.mean(all_imgs_run_time)
            all_avg_times_of_all_sizes.append(avg_time_for_all_imgs)
        all_avg_times_of_all_sizes = np.append(np.array(all_avg_times_of_all_sizes[:index_of_free_size]),
                                               np.mean(all_avg_times_of_all_sizes[index_of_free_size:]))
        sizes = np.append(np.array(sizes[:index_of_free_size]), (np.mean(sizes[index_of_free_size:])))
        plt.plot(sizes, all_avg_times_of_all_sizes, label=algo_name)
    plt.xlabel("Sizes")
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.ylabel("Time in seconds")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Times per size for all algorithems.svg")
    plt.show()
