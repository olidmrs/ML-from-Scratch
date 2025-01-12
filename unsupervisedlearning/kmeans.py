import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

class KMeans():
    def __init__(self, data : np.ndarray, k : int):
        self.data = data
        self.k = k
        self.centroids = np.zeros((0,self.data.shape[1]))
        self.labels = np.zeros(self.data.shape[0], dtype=int)
        self.new_centroids = np.zeros((0, self.data.shape[1]))
            
    def algorithm(self):
        """
        This function is used for the kmeans algorithm
        """

        # setting up basis for the algorithm
        self.feature_normalization()
        self.initializing_centroids()
        keep_going = True
        distances = np.zeros((self.data.shape[0], self.centroids.shape[0]))
        count = 0
        # iterating until no modifications are brought on the centroids
        while keep_going:
            count += 1
            # updating our distance matrix with current distance of every centroid to every datapoint
            for i in range(self.data.shape[0]):
                for j in range(self.centroids.shape[0]):
                    distances[i,j] = np.linalg.norm(self.data[i] - self.centroids[j])
            
            # updating labels matrix that assigns a centroid to all of our points
            for i in range(self.labels.shape[0]):
                self.labels[i] = np.argmin(distances[i])
            
            # initializing centroids matrix
            self.new_centroids = np.zeros_like(self.centroids)

            # updating our centroid matrix based on the mean of all our labelled data
            for i in range(self.centroids.shape[0]):
                temp = self.data[self.labels == i]
                if len(temp) != 0:
                    self.new_centroids[i] = np.mean(temp, axis = 0)
            
            # plotting graphs if dimension is 2d or 3d
            if self.data.shape[1] == 2:
                self.plot_process2d(count)
            elif self.data.shape[1] == 3:
                self.plot_process3d(count)

            # condition for algorithm to stop iterating
            if np.all(self.new_centroids == self.centroids):
                keep_going = False
            else: 
                self.centroids = self.new_centroids

    def feature_normalization(self):
        """
        This function normalizes our data using min-max normalization
        """
        for col in range(self.data.shape[1]):
            minimum = np.min(self.data[:,col])
            maximum = np.max(self.data[:,col])
            for index, value in enumerate(self.data[:,col]):
                if maximum - minimum == 0:
                    self.data[index, col] = 0
                else:
                    self.data[index, col] = (value - minimum)/(maximum - minimum)
        
    def initializing_centroids(self):
        """
        This function randomizes the centroids
        """
        random.seed(42)
        for _ in range(self.k):
            row = np.array([random.random() for _ in range(self.data.shape[1])])
            self.centroids = np.vstack([self.centroids, row])
            
    def plot_process2d(self, iteration : int):
        """
        This function is used to plot a 2d scatter plot
        """
        x_data = self.data[:,0]
        y_data = self.data[:,1]
        x_centroids = self.centroids[:,0]
        y_centroids = self.centroids[:,1]
        fig = plt.figure()
        ax = fig.add_subplot()

        colors = mcolors.TABLEAU_COLORS
        color_map = {}
        for index, colors in enumerate(colors.values()):
            color_map[index] = colors
        
        label_colors = [color_map[label] for label in self.labels]
        centroid_colors = [color_map[index] for index in range(self.centroids.shape[0])]
        
        ax.scatter(x_data, y_data, c = label_colors, s = 10)
        ax.scatter(x_centroids, y_centroids, c = centroid_colors, edgecolors = 'black', s = 45)
        plt.title(f'Iteration number: {iteration}')
        plt.show()


    def plot_process3d(self, iteration : int):
        """
        This function is used to plot a 3d scatter plot

        Args:
            data (np.ndarray): datapoints to plot on graph
            centroids (np.ndarray): centroids to plot on graph
        """
        x_data = self.data[:,0]
        y_data = self.data[:,1]
        z_data = self.data[:,2]
        x_centroids = self.centroids[:,0]
        y_centroids = self.centroids[:,1]
        z_centroids = self.centroids[:,2]

        colors = mcolors.TABLEAU_COLORS
        color_map = {}
        for index, colors in enumerate(colors.values()):
            color_map[index] = colors

        label_colors = [color_map[label] for label in self.labels]
        centroid_colors = [color_map[index] for index in range(self.centroids.shape[0])]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_data, y_data, z_data, c=label_colors, label = 'datapoints', s = 10)
        ax.scatter(x_centroids, y_centroids, z_centroids, c=centroid_colors, edgecolors = 'black', label = 'centroids', s = 45)
        ax.azim = 20
        ax.elev = 15
        plt.title(f'Iteration number: {iteration}')
        plt.show()
