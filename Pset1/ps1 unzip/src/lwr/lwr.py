import matplotlib.pyplot as plt
import numpy as np
import util


def main(tau, train_path, eval_path):
    """Problem: Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train,y_train)

    mse = np.mean(((np.array(clf.y_pred) - np.array(clf.y_hat))**2))
    print("MSE: ", mse)
    
    plt.plot(x_train[0::,1],y_train,'bo')
    plt.plot(clf.x_hat[0::,1],clf.y_pred,'r+')
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression():
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x_hat = None
        self.y_hat = None
        self.theta = None
        self.y_pred = []
        valid_path = './valid.csv'
        self.x_hat, self.y_hat = util.load_dataset(valid_path, add_intercept=True)
        for z in self.x_hat:
            w = []
            for elem in x:
                w.append(np.exp((np.linalg.norm(elem - z)) * (-1 / (2 * self.tau ** 2))))
                
            w = np.diag(w)
            if self.tau<4e-2:
                inv_xTWx = np.linalg.inv(x.T @ w @ x + 1e-3*np.eye(x.shape[1]))
            else:
                inv_xTWx = np.linalg.inv(x.T @ w @ x)
                                         
            self.theta = inv_xTWx @ x.T @ w @ y.T
            self.y_pred.append(self.theta @ z.T)
        return self
        
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (np.dot(x, theta))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau=5e-1,
         train_path='./train.csv',
         eval_path='./valid.csv')
