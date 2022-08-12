from __future__ import division, print_function
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def init_centroids(num_clusters, image):
    """
    Initialize a `num_clusters` x image_shape[-1] nparray to RGB
    values of randomly chosen pixels of`image`

    Parameters
    ----------
    num_clusters : int
        Number of centroids/clusters
    image : nparray
        (H, W, C) image represented as an nparray

    Returns
    -------
    centroids_init : nparray
        Randomly initialized centroids
    """

    # *** START YOUR CODE ***
    H, W, C = np.shape(image)
    rand_idx = np.random.randint(H * W, size=num_clusters)
    centroids_init = image.reshape(-1, C)[rand_idx]
    return centroids_init
    # raise NotImplementedError('init_centroids function not implemented')

    # *** END YOUR CODE ***


def update_centroids(centroids, image, max_iter=30, print_every=10):
    """
    Carry out k-means centroid update step `max_iter` times

    Parameters
    ----------
    centroids : nparray
        The centroids stored as an nparray
    image : nparray
        (H, W, C) image represented as an nparray
    max_iter : int
        Number of iterations to run
    print_every : int
        Frequency of status update

    Returns
    ------
    new_centroids : nparray
        Updated centroids
    """

    # *** START YOUR CODE ***
    num_clusters = np.shape(centroids)[0]
    H, W, C = np.shape(image)
    image_2D = image.reshape(-1, C) 
    converged = False 
    dist = np.empty([num_clusters, H * W])
    
    for it in range(max_iter):
        # single iteration 
        #step 1: assign training data to cluster 
        for j in range(num_clusters):
            dist[j] = np.sum ((image_2D - centroids[j]) **2, axis=1)
        clu_idx = np.argmin(dist, axis=0).reshape(-1, 1) 
        #step 2: moving cluster to the mean of the points 
        centroids_new = np.empty([num_clusters, C])
        for j in range(num_clusters):
            indicator = (clu_idx == j)
            #print ('indicator shape is:', np.shape(indicator))
            centroids_new[j] = np.sum(indicator * image_2D, axis=0) / np.sum(indicator, axis=0)
        #print loss 
        if (it + 1) % print_every == 0: 
            # TBD: dont really understand the subtracted part, kinda understand after writing the code out 
            # for || x(i) - miu(j){c(i)==j}||_2, so sum over RBG 
            loss = (image_2D - centroids_new[clu_idx.squeeze()]) ** 2 
            loss = np.sum(loss)
            print ('loss: %.2f' %loss)
        #check convergence 
        if np.array_equal(centroids_new, centroids):
            converged = True 
    
        centroids = centroids_new 
    
    if converged: 
        print ('Model converges in %d iterations' %(it+1))
    else: 
        print ('Model not converge in %d iterations' %(it+1))
    
    return centroids_new

    # raise NotImplementedError('update_centroids function not implemented')
    
    # *** END YOUR CODE ***

    


def update_image(image, centroids):
    """
    Update RGB values of pixels in `image` by finding
    the closest among the `centroids`

    Parameters
    ----------
    image : nparray
        (H, W, C) image represented as an nparray
    centroids : int
        The centroids stored as an nparray

    Returns
    -------
    image : nparray
        Updated image
    """

    # *** START YOUR CODE ***
    num_clusters = np.shape(centroids)[0]
    H, W, C = np.shape(image)
    image_2D = image.reshape(-1, C) 
    dist = np.empty([num_clusters, H*W])

    for j in range(num_clusters):
        dist[j]= np.sum((image_2D - centroids[j]) ** 2, axis=1)
    clu_idx = np.argmin(dist, axis=0)
    image = centroids[clu_idx].reshape(H, W, C)
    image = image.astype(int)
    return image
    # raise NotImplementedError('update_image function not implemented')

    # *** END YOUR CODE ***

   


def main(args):

    # Setup
    max_iter = args.max_iter
    print_every = args.print_every
    image_path_small = args.small_path
    image_path_large = args.large_path
    num_clusters = args.num_clusters
    figure_idx = 0

    # Load small image
    image = np.copy(mpimg.imread(image_path_small))
    print('[INFO] Loaded small image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original small image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_small.png')
    plt.savefig(savepath, transparent=True, format='png', bbox_inches='tight')

    # Initialize centroids
    print('[INFO] Centroids initialized')
    centroids_init = init_centroids(num_clusters, image)

    # Update centroids
    print(25 * '=')
    print('Updating centroids ...')
    print(25 * '=')
    centroids = update_centroids(centroids_init, image, max_iter, print_every)

    # Load large image
    image = np.copy(mpimg.imread(image_path_large))
    image.setflags(write=1)
    print('[INFO] Loaded large image with shape: {}'.format(np.shape(image)))
    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image)
    plt.title('Original large image')
    plt.axis('off')
    savepath = os.path.join('.', 'orig_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    # Update large image with centroids calculated on small image
    print(25 * '=')
    print('Updating large image ...')
    print(25 * '=')
    image_clustered = update_image(image, centroids)

    plt.figure(figure_idx)
    figure_idx += 1
    plt.imshow(image_clustered)
    plt.title('Updated large image')
    plt.axis('off')
    savepath = os.path.join('.', 'updated_large.png')
    plt.savefig(fname=savepath, transparent=True, format='png', bbox_inches='tight')

    print('\nCOMPLETE')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--small_path', default='./peppers-small.tiff',
                        help='Path to small image')
    parser.add_argument('--large_path', default='./peppers-large.tiff',
                        help='Path to large image')
    parser.add_argument('--max_iter', type=int, default=150,
                        help='Maximum number of iterations')
    parser.add_argument('--num_clusters', type=int, default=16,
                        help='Number of centroids/clusters')
    parser.add_argument('--print_every', type=int, default=10,
                        help='Iteration print frequency')
    args = parser.parse_args()
    main(args)
