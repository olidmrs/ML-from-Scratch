import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

class DBSCAN():
    def __init__(self, data : np.ndarray, eps : float, minPts : int):
        self.data = data
        self.eps = eps
        self.minPts = minPts
        self.label = np.zeros((self.data.shape[0], 1))
    
    def count_border_points(self, point : np.ndarray):
        """
        Function to count number of neighbours in eps and who they are

        Args:
            point (np.ndarray): vector representing a point

        Returns:
            int: returns number of neighbouring points
            np.ndarray: returns array of neighbours
        """
        distances = np.linalg.norm(self.data - point, axis = 1)
        return (distances <= self.eps).sum(), (distances <= self.eps).nonzero()[0]
        
    def develop_neighbours(self, point : np.ndarray, cluster : int):
        """
        Recursive function to develop on neighbours and assign them to cluster

        Args:
            point (np.ndarray): vector representing a point
            cluster (int): cluster index to assign to neighbours
        """        
        distances = np.linalg.norm(self.data - point, axis = 1)
        neighbours = (distances <= self.eps).nonzero()[0]

        for neighbour in neighbours:
            # 0 being points not visited yet
            if self.label[neighbour] == 0: 
                self.label[neighbour] == cluster
                self.develop_neighbours(self.data[neighbour], cluster)
            # -1 being points that are noise
            elif self.label[neighbour] == -1: 
                self.label[neighbour] = cluster
        
    def algorithm(self):
        """
        Function to perform the algorithm of DBSCAN until all points are in a cluster or noise
        """        
        cluster = 1
        for point_index in range(self.data.shape[0]):

            if self.label[point_index] == 0:
                point = self.data[point_index]
                nb_border, neighbours = self.count_border_points(point)

                # Checking condition to create a cluster
                if nb_border >= self.minPts:
                    self.label[point_index] = cluster
                    self.label[neighbours] = cluster
                    for neighbour in neighbours:
                        if self.label[neighbour] == 0:
                            self.develop_neighbours(self.data[neighbour], cluster)
                    cluster += 1
                else:
                    self.label[point_index] = -1
    
    def plot_clusters(self):

        """
        Function to plot our data now labelled in 2d or 3d based on dimension of features

        Raises:
            Exception: Raised if dimension is not valid for plotting or not supported by function
        """
        colors = mcolors.CSS4_COLORS
        color_map = {}
        for index, colors in enumerate(colors.values()):
            color_map[index] = colors
        color_map[-1] = 'black' 

        label_colors = [color_map[label] for label in self.label.reshape(-1)]
        if self.data.shape[1] == 2:
            x_data = self.data[:,0]
            y_data = self.data[:,1]
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_facecolor('grey')
            ax.scatter(x_data, y_data, c = label_colors, s = 10)
            plt.show()
            
        elif self.data.shape[1] == 3:
            x_data = self.data[:,0]
            y_data = self.data[:,1]
            z_data = self.data[:,2]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('grey')
            ax.scatter(x_data, y_data, z_data, c=label_colors, label = 'datapoints', s = 10)
            ax.azim = 20
            ax.elev = 15
            plt.show()
        elif self.data.shape[1] == 1:
            raise Exception(f'input feautures of dimension {self.data.shape[1]} which is not supported by this method')
        else:
            raise Exception(f'input feautures is of dimension {self.data.shape[1]} which is not representable graphically')


