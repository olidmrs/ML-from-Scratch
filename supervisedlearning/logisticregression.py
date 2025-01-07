import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogRegression():
    def __init__(
            self,
            train_x : np.ndarray,
            train_y : np.ndarray,
            test_x : np.ndarray,
            test_y : np.ndarray,
            learning_rate : float,
            iterations : int
            ):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.learning_rate = learning_rate

        self.theta = np.zeros((self.train_x.shape[1]))
        self.b = 0
        self.cost_history = []
        self.gradient_descent_iterations = 0
        self.iterations = iterations

    def sigmoid_function(self, x):
        x = np.clip(x, -709, 709)
        return 1/(1+ np.exp(-x))
    
    def gradientdescent(self):
        """
        This function is used to compute the gradient descent algorithm 

        Raises:
            ValueError: If learning rate is too high, Gradient descent diverges 
        """
        for _ in range(self.iterations):
            
            # making prediction
            y_pred = self.sigmoid_function(np.dot(self.train_x, self.theta) + self.b)
            
            y_pred = np.clip(y_pred, 0.0001, 0.9999)
            
            # computing derivative of cost function
            dtheta = 1/self.train_x.shape[0] * np.dot(y_pred - self.train_y, self.train_x)
            db = 1/self.train_x.shape[0]*np.sum(y_pred - self.train_y)
            
            # modifying theta vector based on error
            self.theta = self.theta - (self.learning_rate * dtheta)
            self.b = self.b - (self.learning_rate * db)

            # computing cost and storing it in cost_history
            cost = -(1/self.train_x.shape[0])*np.sum(self.train_y * np.log(y_pred) + (1-self.train_y) * np.log(1-y_pred))
            self.cost_history.append(cost)
            self.gradient_descent_iterations += 1

            # if learning rate is too high we get nan values in our thetas and raise an error
            if np.isnan(self.theta).any():
                raise ValueError(f'Learning rate: {self.learning_rate} is too high. Gradient descent is diverging')
            
    def make_prediction(self, features : np.ndarray, threshold : float) -> np.ndarray:
        """
        This function is used to make a prediction based on new features

        Args:
            features (np.ndarray): array of features used to make prediction

        Returns:
            float: returns predicted value
        """
        predictions = self.sigmoid_function(np.dot(features, self.theta) + self.b)
        for index,value in enumerate(predictions):
            if value >= threshold:
                predictions[index] = 1
            else:
                predictions[index] = 0
        return predictions

    
    def plot_error_function(self):
        """
        This function is used to plot the error function
        """
        plt.plot(range(self.gradient_descent_iterations), self.cost_history, label='Cost')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost function during gradient descent')
        plt.show()
    
    def confusion_matrix(self, threshold : float):
        T_true = 0
        T_false = 0
        F_true = 0
        F_false = 0
        prediction = self.make_prediction(self.test_x, threshold)

        for index in range(prediction.shape[0]):
            if (prediction[index] == 1) and (self.test_y[index] == 1):
                T_true += 1
            elif (prediction[index] == 1) and (self.test_y[index] != 1):
                F_true += 1
            elif (prediction[index] == 0) and (self.test_y[index] == 0):
                T_false += 1
            else:
                F_false += 1
        return T_true, F_true, F_false, T_false
        
    
    def ROC(self):
        tpr_list = []
        fpr_list = []

        for i in np.linspace(0,1,111):
            tp, fp, fn, tn = self.confusion_matrix(threshold = i)
            tpr_list.append((tp)/(tp + fn) if (tp + fn) > 0 else 0)
            fpr_list.append((fp)/(fp + tn) if (fp + tn) > 0 else 0)
            
        plt.plot(fpr_list, tpr_list)
        plt.plot([0,1],[0,1], 'r--')
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()



