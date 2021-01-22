import os
import pickle
import re
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    def dd():
        return defaultdict(dict)


    with open("pickled_data", "rb") as f:
        all_runs_data = pickle.load(f)
        f.close()

    # Show the cats
    fig, axs = plt.subplots(1, 4)
    i = 0
    for imagefile in ["cat_easy.jpg", "cat_a.jpg", "cat_yoy.jpg", "cat_medium.jpg"]:
        image = cv2.cvtColor(cv2.imread(f"example cats/{imagefile}"), cv2.COLOR_BGR2RGB)
        axs[i].imshow(image, aspect='auto')
        i += 1
    plt.tight_layout()
    plt.show()

    # Show seeds on all resulitions
    fig, axs = plt.subplots(3, 4)
    i, j = 0, 0
    for imagefile in sorted(os.listdir("seeded_cats")):
        image = cv2.cvtColor(cv2.imread("seeded_cats/" + imagefile), cv2.COLOR_BGR2RGB)
        axs[i][j].imshow(image)
        axs[i][j].axis("off")
        i += 1
        j += (i) // 3
        i = (i) % 3
    plt.tight_layout()
    plt.show()


    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)


    # Show all results
    for cat in ["cat_yoy results","cat_a results","cat_easy results","cat_medium results"]:
        fig, axs = plt.subplots(3, 10, figsize=(20, 10))
        i, j = 0, 0
        for imagefile in sorted_alphanumeric(os.listdir(cat)):
            image = cv2.cvtColor(cv2.imread(cat+"/" + imagefile), cv2.COLOR_BGR2RGB)
            axs[i][j].imshow(image)
            axs[i][j].set_yticklabels([])
            axs[i][j].set_xticklabels([])
            axs[i][j].set_yticks([])
            axs[i][j].set_xticks([])

            i += 1
            j += (i) // 3
            i = (i) % 3

        cols = ["boyokov", "diniz", "edmondes", "pp", "sap 1", "sap 2", "BSC 20", "SC 1", "BSC 1", "SC 2"]
        rows = ['30X30', '100X100', 'Max size']
        for ax, row in zip(axs[:, 0], rows):
            ax.set_ylabel(row, rotation=90)
        for ax, col in zip(axs[0], cols):
            ax.set_title(col, fontsize=10)

        plt.tight_layout()
        plt.show()
