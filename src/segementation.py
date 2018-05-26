import numpy as np
from scipy import misc, ndimage
from sklearn.cluster import KMeans
import tensorflow as tf
import matplotlib.pyplot as plt

class Segementation(object):
    """
    segementation of the ROI (region of interest) by
    unsupervised kmeans algorithm  
    author: yw3025
    """

    def __init__(self, image):
        self.image = image
        self.shape = 128

    # yw3025
    def read_image(self, path = "trees.png"):
        self.image = ndimage.imread(path, mode="RGB")
    # yw3025
    def fit_kmeans(self, n_clusters):
        """
        fit kmeans
        :param n_clusters:
        :return:
        """
        kme = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        kme.fit(self.rgb)
        return kme

    # yw3025
    def predict_kmeans(self, kme):
        """
        predict which center an image point belong to
        :param test: test data
        :param shape: the shape of the image (x,y)
        :return:
        """
        return kme.predict(self.rgb).reshape((self.shape, self.shape))

    # fill in rgb values with
    # yw3025
    def fill_rgbs(self, kme, result):
        """
        :param projected: prejected image (x,y,r,g,b)
        :param kme: kme trained model
        :param result: predicted cluster center(stored in result), for every x, y, one center number
                0 - num_cluster
        :return:
        """
        projected = np.empty_like(self.image)
        for i, c in enumerate("RGB"):
            projected[:, :, i] = kme.cluster_centers_[:, i].take(result)
        self.image = projected
        # for i in range(projected.shape[0]):
        #     for j in range(projected.shape[1]):
        #         for k in range(3):  # RGB
        #             cluster_center = kme.cluster_centers_[result[i][j], :]
        #             projected[i][j][k] = cluster_center[k]
    # yw3025
    def show_image(self):
        plt.imshow(self.image)
        plt.show()

    # yw3025
    def get_result(self, image):
        """
        get segementation result for discriminator
        :param image:
        :return:
        """
        self.image = image.reshape(self.shape, self.shape, 3)
        # shape = self.image.shape[0:2]
        self.rgb = self.image.reshape((-1, 3)).astype(np.float64)
        kme = self.fit_kmeans(5)
        result = self.predict_kmeans(kme)
        self.fill_rgbs(kme, result)

        sky_center = (5, 64)
        rail_center = (125, 64)
        # left_tree_corner = (100, 18)
        # right_tree_corner = (100, 110)
        mask_set = set()
        mask_set.add(result[sky_center])
        # mask_set.add(result[rail_center])
        # mask_set.add(result[left_tree_corner])
        # mask_set.add(result[right_tree_corner])
        for row in range(self.shape):
            for col in range(self.shape):
                if result[row, col] in mask_set:
                    self.image[row, col, :] = [2,2,2]
                else:
                    self.image[row, col, :] = [1, 1, 1]

        # self.show_image()
        return self.image.reshape(1,self.shape,self.shape,3)
    # yw3025
    def get_result_railway(self, image):
        """
        get segementation result for generator
        :param image:
        :return:
        """
        self.image = image.reshape(self.shape, self.shape, 3)
        self.rgb = self.image.reshape((-1, 3)).astype(np.float64)
        kme = self.fit_kmeans(5)
        result = self.predict_kmeans(kme)
        self.fill_rgbs(kme, result)

        rail_center = (125, 64)
        mask_set = set()
        mask_set.add(result[rail_center])

        for row in range(self.shape):
            for col in range(self.shape):
                if result[row, col] in mask_set:
                    self.image[row, col, :] = [1,1,1]
                else:
                    self.image[row, col, :] = [0, 0, 0]

        # self.show_image()
        return self.image.reshape(1,self.shape,self.shape,3)
    # yw3025
    def get_result_test(self, image):
        """
        for test use
        :param image:
        :return:
        """
        # self.image = image.reshape(256, 256, 3)
        self.rgb = self.image.reshape((-1, 3)).astype(np.float64)
        kme = self.fit_kmeans(5)
        result = self.predict_kmeans(kme)
        self.fill_rgbs(kme, result)

        rail_center = (125, 64)
        mask_set = set()
        mask_set.add(result[rail_center])

        for row in range(self.shape):
            for col in range(self.shape):
                if result[row, col] in mask_set:
                    self.image[row, col, :] = [1,1,1]
                else:
                    self.image[row, col, :] = [255, 0, 0]

        self.show_image()
        # return self.image.reshape(1,256,256,3)
#


# yw3025
if __name__ == "__main__":
    # read image and preprocessing
    # test
    segementation = Segementation(None)
    segementation.read_image("0008.jpg")
    segementation.get_result_test(None)






