import numpy as np
import util
import os

def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    # *** START CODE HERE ***
    x_test, y_test = util.load_dataset(valid_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)
    plot_path = os.getcwd()+'/'+save_path[:-4] + '.jpg'
    util.plot(x_test, y_test, clf.theta, plot_path)
    np.savetxt(save_path, clf.predict(x_test))

    # Train a GDA classifier

    # Plot decision boundary on validation set

    # Use np.savetxt to save outputs from validation set to save_path

    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        phi = np.mean(y)
        mu0 = (x[y == 0]).mean(axis=0, keepdims=True)
        mu1 = (x[y == 1]).mean(axis=0, keepdims=True)
        mean = np.where(y.reshape(-1, 1), mu1, mu0)
        sigma = (x - mean).T @ (x - mean) / x.shape[0]

        self.theta = np.empty (x.shape[1] + 1)
        mu_dif = (mu1 - mu0).squeeze()
        sigma_inv = np.linalg.inv(sigma)
        self.theta[1:] = mu_dif @ sigma_inv 
        self.theta[0] = (np.log(phi / (1 - phi)) - mu_dif @ sigma_inv @mu_dif / 2)

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return ((self.theta[0] + self.theta[1:]@x.T) > 0).astype('int')

        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
