import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import mplcursors


# Absolute Path
IMG = "puan-chan.jpg"


class ColorPalleteExtractor:
    def __init__(self, img):
        self.img = img
        self.img = cv.imread(self.img)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
        self.img = cv.resize(self.img, (0, 0), fx=0.5, fy=0.5)
        self.img = self.img.reshape(self.img.shape[0] * self.img.shape[1], 3)

    def extract_colors(self):
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(self.img)
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        return colors, labels

    def plot_colors(self, colors):
        fig, axs = plt.subplots(1, len(colors), figsize=(8, 2))

        for sp in range(len(colors)):
            color_array = np.zeros((110, 110, 3), dtype="uint8")
            color_block = np.zeros((106, 106, 3), dtype="uint8")
            color_block[:, :, :] = colors[sp]
            color_array[2:-2, 2:-2, :] = color_block
            axs[sp].imshow(color_array)
            axs[sp].axis("off")

        cursor = mplcursors.cursor(hover=True)

        @cursor.connect("add")
        def on_add(sel):
            try:
                sel.annotation.set_text(f"RGB: {colors[sel.index]}")
                sel.annotation.get_bbox_patch().set(fc="white")
                sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white")
            except IndexError:
                sel.annotation.set_text("")
                sel.annotation.get_bbox_patch().set(fc="white")
                sel.annotation.arrow_patch.set(arrowstyle="simple", fc="white")

        plt.show()


if __name__ == "__main__":
    extractor = ColorPalleteExtractor(IMG)
    colors, labels = extractor.extract_colors()
    extractor.plot_colors(colors)
