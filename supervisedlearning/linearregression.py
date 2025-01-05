import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinRegression():
    def __init__(self, trainset: pd.DataFrame, testset: pd.DataFrame, target: str, learning_rate : int):
        self.trainset = trainset
        self.testset = testset
        self.target = target
        self.learning_rate = learning_rate

        self.theta = np.array([0 for _ in range(trainset.shape[1])])

        self.training_features = self.formatting_features(training = True)
        self.training_target = self.formatting_target(training = True)

        self.testing_features = self.formatting_features(training = False)
        self.testing_target = self.formatting_target(training = False)

        self.cost_history = []
        self.gradient_descent_iterations = 0

    
    def gradientdescent(self):
        """
        This function is used to compute the gradient descent algorithm 

        Raises:
            ValueError: If learning rate is too high, Gradient descent diverges 
        """
        for _ in range(1000):
            # making prediction
            y_pred = np.dot(self.training_features, self.theta)

            # computing derivative of cost function
            dtheta = (1/self.training_features.shape[0]) * np.dot(self.training_features.T, y_pred - self.training_target)
            
            # modifying theta vector based on error
            self.theta = self.theta - (self.learning_rate * dtheta)

            # computing cost and storing it in cost_history
            cost = (1/(2*self.trainset.shape[0])) * np.sum(np.square(np.matmul(self.training_features, self.theta) - self.training_target))
            self.cost_history.append(cost)
            self.gradient_descent_iterations += 1

            # if learning rate is too high we get nan values in our thetas and raise an error
            if np.isnan(self.theta).any():
                raise ValueError(f'Learning rate: {self.learning_rate} is too high. Gradient descent is diverging')

    
    def formatting_features(self, training : bool) -> np.ndarray:
        """
        This function is used to format the feature data

        Args:
            training (bool): True to format training data. False to format target data

        Returns:
            np.ndarray: returns formatted data
        """
        if training:
            temp = self.trainset.drop(columns=self.target)
            temp = temp.to_numpy()
            features = np.c_[np.ones(temp.shape[0]), temp]
            return features
        else: 
            temp = self.testset.drop(columns=self.target)
            temp = temp.to_numpy()
            features = np.c_[np.ones(temp.shape[0]), temp]
            return features

    def formatting_target(self, training : bool) -> np.ndarray:
        """
        This function is used to format the target data

        Args:
            training (bool): True to format training data. False to format target data

        Returns:
            np.ndarray: returns formatted data
        """
        if training: 
            target = self.trainset[self.target].to_numpy()
            return target
        else:
            target = self.testset[self.target].to_numpy()
            return target
    
    def RMSE_calculation(self) -> float:
        """
        This function calculates the RMSE

        Returns:
            float: returns RMSE
        """
        cost = np.sqrt(1/self.trainset.shape[0]*np.sum(np.square(np.dot(self.testing_features, self.theta) - self.testing_target)))
        return cost
    
    def make_prediction(self, features : np.ndarray) -> float:
        """
        This function is used to make a prediction based on new features

        Args:
            features (np.ndarray): array of features used to make prediction

        Returns:
            float: returns predicted value
        """
        return np.dot(features, self.theta)

    def plot_error_function(self):
        """
        This function is used to plot the error function
        """
        plt.plot(range(self.gradient_descent_iterations), self.cost_history, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function during gradient descent')
        plt.show()