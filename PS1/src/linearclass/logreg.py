import numpy as np
import util
import os

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_test, y_test = util.load_dataset(valid_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    plot_path = os.getcwd()+'/'+save_path[:-4] + '.jpg'
    util.plot(x_test, y_test, clf.theta, plot_path)
    prediction = clf.predict(x_test)
    np.savetxt(save_path, prediction)


    # Train a logistic regression classifier

    # Plot decision boundary on top of validation set set

    # Use np.savetxt to save predictions on eval set to save_path

    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
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
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        self.theta = np.zeros(x.shape[1])
        while True: 
            y_pred = self.predict(x)
            gradient = ((y_pred - y) * x.T).mean(axis=1)
            hessian = ((y_pred * (1 - y_pred)) * x.T) @ x / x.shape[1]
            update = np.linalg.inv(hessian) @ gradient
            self.theta = self.theta - update 
            if np.abs(update).sum() < self.eps: 
                return self

        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-(self.theta @ x.T)))
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
