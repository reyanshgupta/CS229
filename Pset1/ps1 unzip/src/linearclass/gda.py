import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=False)
    plot_path = 'gda.jpg'
    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)
    util.plot(x_valid, y_valid, clf.theta, plot_path)
    np.savetxt(save_path, clf.predict(x_valid))
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
       # *** START CODE HERE ***
        x0, x1 = np.split(x, [np.sum(y == 0)])
        phi = np.mean(y)
        mu_0= np.mean(x0,axis=0)
        mu_1=np.mean(x1,axis=0)
        mean_y = np.where(y.reshape(-1, 1), mu_1, mu_0)
        sigma = (x - mean_y).T @ (x-mean_y) / len(x)
        
        self.theta= np.empty(x.shape[1]+1)
        sigma_inverse=np.linalg.inv(sigma)
        mu_difference=(mu_1 - mu_0).squeeze()
        mu_sum = (mu_1+mu_0).squeeze()
        self.theta[1:] = np.dot(mu_difference, sigma_inverse)
        self.theta[0] = np.log(phi / (1 - phi)) - 0.5 * np.dot(np.dot(mu_difference, sigma_inverse), mu_sum)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        prediction = self.theta[0] + np.dot(self.theta[1:], x.T)
        return (prediction > 0).astype(int)
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
