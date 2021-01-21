from __future__ import division

import argparse

import networkx.algorithms.flow as algos
import matplotlib.pyplot as plt
from scipy import ndimage
import networkx as nx
import numpy as np
import pickle
import scipy
import time
import cv2
import sys
import os

from collections import defaultdict
from simcut import sim_cut

OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)
OBJCODE, BKGCODE = 1, 2
OBJ, BKG = "OBJ", "BKG"
CUTCOLOR = (0, 0, 255)
SOURCE, SINK = -2, -1
SF = 10

# plant seeds option
MANUAL = False
# plant seeds only ones for every image size and type
MANUAL_FIRST = False
LOADSEEDS = False


def show_image(image: np.ndarray):
    """
    Shows an image using cv2 library
    :param image: The image to show.
    :return: Nothing
    """
    cv2.namedWindow("", cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.imshow("", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plant_seed_manual(image: np.ndarray, m):
    """
    Show a window prompt to choose the seeds manually
    :param image: The image to add the seeds on
    :param m:
    :return:
    """
    first = MANUAL_FIRST
    if first:
        # Create the seed file for the graph building
        f = open("seeds1.txt", 'a')
        shape_str = 'none' if image.shape[0] > 100 else str(image.shape[0])
        f.write(m.split('.')[0] + "_" + shape_str + "\n")

    def draw_lines(x, y, pixelType):
        """
        Draw a pixel on the image
        :param x: The x coordinate of the seed
        :param y: The y coordinate of the seed
        :param pixelType: The color of the pixel
        :return: None
        """
        if first:
            f.write(str(x) + ' ' + str(y) + "\n")
        if pixelType == OBJ:
            color, code = OBJCOLOR, OBJCODE
        else:
            color, code = BKGCOLOR, BKGCODE
        cv2.circle(image, (x, y), radius, color, thickness)
        cv2.circle(seeds, (x // SF, y // SF), radius // SF, code, thickness)

    def on_mouse(event, x, y, flags, pixelType):
        """
        Follow mouse click
        :param event: The event handler
        :param x: The x coordinate of the mouse
        :param y: The y coordinate of the mouse
        :param flags:
        :param pixelType: The pixel color
        :return: None
        """
        global drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            draw_lines(x, y, pixelType)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            draw_lines(x, y, pixelType)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    def paint_seeds(pixelType):
        """
        Paint the seeds on the image
        :param pixelType:
        :return:
        """
        print("Planting", pixelType, "seeds")
        global drawing
        drawing = False
        windowname = "Plant " + pixelType + " seeds"
        cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowname, 500, 500)
        cv2.setMouseCallback(windowname, on_mouse, pixelType)
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

    paint_seeds(OBJ)
    paint_seeds(BKG)
    if first:
        f.close()
    return seeds, image


def plant_seed(image: np.ndarray, pathname: str):
    """
    Opens image and seed, and adds the seed onto the image
    :param image: The image to load
    :param pathname: The name tof the image
    :return: The seeds
    """
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


def build_graph(image: np.ndarray, pathname: str, sigma: int = 30, save_seeded_image: bool = False):
    """
    Build the graph from the image
    :param image: The image to build the graph from
    :param pathname: The name of the image
    :param sigma: The sigma to use for calculating the capacity value
    :param save_seeded_image: Should the image with the seed be saved
    :return: The graph and the seeded image
    """
    V = image.size + 2
    if (V > 1e5):
        raise MemoryError
    graph = nx.Graph()
    K, graph = make_N_links(graph, image, sigma=sigma)
    if MANUAL_FIRST and MANUAL:
        seeds, seededImage = plant_seed_manual(image, pathname)
    else:
        seeds, seededImage = plant_seed(image, pathname.split("/")[-1])

    make_T_links(graph, seeds, K)

    if save_seeded_image:
        # Fast way to save the seeds on the image
        pixels = []
        for x in seeds.flatten():
            if x == 0:
                pixels.append((0, 0, 0, 0))
            if x == 1:
                pixels.append((255, 0, 0, 150))
            if x == 2:
                pixels.append((0, 255, 0, 150))
        np_pixels = np.array(pixels)
        plt.imshow(image, cmap="gray")
        plt.imshow(np_pixels.reshape(image.shape[0], image.shape[0], 4))
        plt.axis("off")
        plt.tight_layout()
        os.makedirs("seeded_cats", exist_ok=True)
        plt.savefig(f"seeded_cats/{pathname}_{image.shape[0]}.png", bbox_inches='tight', pad_inches=0)
    return graph, seededImage


def make_N_links(graph, image, sigma=30):
    """
    Create the N links of the graph from the image.
    :param graph: The graph to add the capacity on
    :param image: The image of reference
    :param sigma: The sigma to use in the calculation
    :return: The K value for the T links and the graph
    """
    boundray_penalty_constant = -2 * pow(sigma, 2)

    K = -float("inf")
    r, c = image.shape

    # Create a mask for all pixels under the original pixels
    zero_len_vec = np.expand_dims(np.zeros(image.shape[1]), axis=0)
    Dimage = np.vstack((image, zero_len_vec))[1:, :]

    # Create a mask for all pixels to the right of the original pixels
    zero_high_vec = np.expand_dims(np.zeros(image.shape[0]), axis=1)
    Rimage = np.hstack((image, zero_high_vec))[:, 1:]

    # Calculate the links between the rows
    Dimage = 100 * np.exp(((image.astype("int32") - Dimage.astype("int32")) ** 2) / boundray_penalty_constant)[:-1, :]
    # Calculate the links between the cols
    Rimage = 100 * np.exp(((image.astype("int32") - Rimage.astype("int32")) ** 2) / boundray_penalty_constant)[:, :-1]

    # Create the nodes of the graph and define the capacity of the rows
    for i in range(r - 1):
        x = np.arange(c) + (i * c)
        s = Dimage[i, :]
        y = np.arange(c) + ((i + 1) * c)
        a = np.vstack((x, y, s)).T
        graph.add_weighted_edges_from(a, weight="capacity")
    # Define the capacity of the cols
    for i in range(c - 1):
        x = np.arange(r) * (r) + i
        s = Rimage[:, i]
        y = np.arange(r) * (r) + (i + 1)
        a = np.vstack((x, y, s)).T
        graph.add_weighted_edges_from(a, weight="capacity")
    # Define K to be the maximum capacity in the graph
    K = max([c["capacity"] for s, t, c in graph.edges(data=True)])

    return K, graph


def make_T_links(graph, seeds, K):
    """
    Create the T links of the graph. works inplace.
    :param graph: The graph to add the T links on
    :param seeds: The seeds to connect
    :param K: The capacity of the T links
    :return: None
    """
    r, c = seeds.shape

    for i in range(r):
        for j in range(c):
            x = i * c + j
            if seeds[i][j] == OBJCODE:
                graph.add_edge(SOURCE, x, capacity=K)
            elif seeds[i][j] == BKGCODE:
                graph.add_edge(SINK, x, capacity=K)


def display_cut(image, cuts):
    """
    Draw the cut on the image
    :param image:The image to draw on
    :param cuts: The cuts to draw
    :return: The image with the cuts
    """

    def colorPixel(i, j):
        image[int(i)][int(j)] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image


def create_graph_from_img(imagefile: str, sigma: int, size=None):
    """
    Create graph from the image
    :param imagefile: The path of the image
    :param size: A size for resizing the image
    :param sigma: The initial sigma to calculate the capacity with
    :return: The graph created, the image loaded ,the size chosen and the name of the image
    """
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
            graph, seededImage = build_graph(image, pathname, sigma=sigma)
            print(f"The resolution will be: {image.shape[0]}x{image.shape[1]}")
            size = (int(image.shape[0]), int(image.shape[1]))
            break
        except MemoryError:
            size = (int(image.shape[0] * 0.75), int(image.shape[1] * 0.75))
            image = cv2.resize(image, size)
            continue
    return graph, image, size, pathname


def image_segmentation(graph, image, size, pathname, flag=False, algo=None, algo_name=None, show=False,
                       sigma: int = 30):
    """
    The main segmentation workflow
    :param graph: The graph to work with
    :param image: The image to segment
    :param size: A size to resize
    :param pathname: The name of the image
    :param algo: The algorithm to use in the segmentation
    :param algo_name: The name of the algorithm
    :param sigma: The sigma to calculate the capacity with
    :param flag: A flag for the shortest augmenting path algorithm. if false- use one phase, else- two phases
    :param show: True to show the segmentation after the calculation
    :return: The time of the calculation and the size of the image
    """
    print(algo_name)

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
            # end of calculation - make the data look like the other algorithms
            segmentation = (segmentation.reshape(size))
            imax = (scipy.ndimage.maximum_filter(segmentation, size=3) != segmentation)
            # keep only pixels of original image at borders
            edges = np.where(imax, 1, -1).reshape(size[0] ** 2)
            cuts = []
            for node in np.where(edges > 0)[0]:
                if node < (len(graph) - 2):
                    cuts.append((node, node))
        else:
            # Normal case
            start_time = time.process_time()
            cut_value, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity", flow_func=algo)
            end_time = time.process_time()

        try:
            reachable, non_reachable = partition  # This line will fail if we used sim cut
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
                graph, seededImage = build_graph(image, pathname, sigma=sigma)
            else:
                found_good_sigma = True
                cuts = sorted(cutset_clean)
        except:
            # sim cut was used
            break
    print("cuts:")
    print(cuts)
    print("Time to calculate:")
    print(end_time - start_time)
    image = display_cut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=SF, fy=SF)
    if show:
        show_image(image)
    os.makedirs(f"{pathname} results", exist_ok=True)
    savename = f"{pathname} results/ {algo_name}_{size}_result.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)
    return end_time - start_time, len(graph)


def parseArgs(algo_options_dict, default_images):
    parser = argparse.ArgumentParser()
    parser.add_argument(*["-p", "--paths", "--imagefile"], type=str, nargs="*", default=default_images,
                        help="Path of images. could be one or more. defualt is an example of 4 cats.")
    parser.add_argument("--size", "-s",
                        default="30", type=str,
                        help="The resize value. Defaults to 30x30. None for maximum resolution")
    parser.add_argument("--algos", "-a", default=["bk"], nargs='*',
                        choices=["all", *algo_options_dict.keys()],
                        help="The algorithms to use in this run. one or more. default is boyokov kolmagorov")
    parser.add_argument("--sigma", default=30, type=int,
                        help="The sigma to use in the capacity calculations. defualt is 30.")

    return parser.parse_args()


if __name__ == "__main__":
    algo_options_dict = {"pp": algos.preflow_push, "sc": sim_cut, "bk": algos.boykov_kolmogorov,
                         "sap": algos.shortest_augmenting_path, "dinitz": algos.dinitz, "d": algos.dinitz,
                         "ek": algos.edmonds_karp}
    default_images = ["example cats/cat_easy.jpg", "example cats/cat_a.jpg", "example cats/cat_yoy.jpg",
                      "example cats/cat_medium.jpg"]
    args = parseArgs(algo_options_dict, default_images)
    sigma = args.sigma
    images = args.paths
    manual = (images != default_images)
    try:
        size = int(args.size)
    except:
        size = None
    if "all" in args.algos:
        algorithms = algo_options_dict.values()
    else:
        algorithms = [algo_options_dict[i] for i in args.algos]


    def dd():
        return defaultdict(dict)


    all_runs_data = defaultdict(dd)
    try:
        if not manual:
            with open("pickled_data", "rb") as f:
                all_runs_data = pickle.load(f)
                f.close()
    except:
        pass

    if len(sys.argv) > 1:
        MANUAL = manual

    if manual:
        target_sizes = [size]
    else:
        target_sizes = [30, 100, None]

    for size in target_sizes:
        flag = False
        MANUAL_FIRST = True
        for algo in algorithms:
            # Save the algorithm nam for saving
            if algo.__name__ == "shortest_augmenting_path" and flag is False:
                algo_name = algo.__name__ + " one phase"
            elif algo.__name__ == "shortest_augmenting_path" and flag is True:
                algo_name = algo.__name__ + " two phase"
            elif algo.__name__ == "sim_cut":
                algo_name = "sim_cut_20_iter_better_implement"
            else:
                algo_name = algo.__name__

            run_data = {}
            for img_path in images:
                graph, image, size_, pathname = create_graph_from_img(img_path, sigma=sigma, size=(size, size))
                try:
                    all_runs_data[algo_name][len(graph)][
                        img_path.split("/")[-1].replace(".jpg", "").replace(".png", "")]

                    # This data was collected already - skip it
                    continue
                except:
                    pass
                run_time, im_size = image_segmentation(graph, image, size_, pathname, algo=algo, algo_name=algo_name,
                                                       flag=flag, show=False, sigma=sigma)
                # Save the runtime data
                all_runs_data[algo_name][len(graph)][img_path.replace(".jpg", "")] = run_time

            if algo.__name__ == "shortest_augmenting_path" and flag is False:
                # Flip the flag so that the two phase version will happen next time
                flag = True

            # Save the runtime data
            with open("pickled_data", "wb") as f:
                pickle.dump(all_runs_data, f)
                f.close()
            MANUAL_FIRST = False

    # Show runtime plot:
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
