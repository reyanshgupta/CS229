import matplotlib.pyplot as plt
import numpy as np
import util

from lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem: Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    
    
    # *** START CODE HERE ***
    
    mse = {}
    for tau in range(len(tau_values)):
        clf = LocallyWeightedLinearRegression(tau_values[tau])
        x_train, y_train = util.load_dataset(train_path, add_intercept=True)
        clf.fit(x_train, y_train)

        plt.figure(tau)
        plt.plot(x_train[:, 1], y_train, 'bo')
        plt.plot(clf.x_hat[:, 1], clf.y_pred, 'r+')
        plt.title(f'Train vs predicted data for tau = {tau_values[tau]}')

        mseval = ((clf.y_hat - clf.y_pred) ** 2).mean()
        print("MSE: ", mseval, "for tau: ", tau_values[tau])
        mse[mseval] = tau_values[tau]

    tau = mse[min(list(mse.keys()))]
    clf = LocallyWeightedLinearRegression(tau)
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    clf.fit(x_train, y_train)
    plt.figure(len(tau_values) + 1)
    plt.plot(x_train[:, 1], y_train, 'bo')
    plt.plot(clf.x_hat[:, 1], clf.y_pred, 'r+')
    plt.title(f'Train vs predicted data for tau = {tau} (with maximum MSE)')
    mse_tot = ((clf.y_hat - clf.y_pred) ** 2).mean()
    print("MSE =", mse_tot, "for train split with tau =", tau)

    
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(tau_values=[3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1],
         train_path='./train.csv',
         valid_path='./valid.csv',
         test_path='./test.csv',
         pred_path='./pred.txt')
