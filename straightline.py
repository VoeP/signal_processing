import numpy as np
from skimage import feature
import sys
import matplotlib
from scipy import ndimage
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'nearest'
import matplotlib.pyplot as plt



def fit_line(points):

    p1x = points[0, 0]
    p1y = points[0, 1]

    p2y = points[1, 0]
    p2x = points[1, 1]

    if p2y == p1y:
        m = sys.float_info.epsilon
        c = p1y

    elif p2x == p1x:
        m = sys.float_info.max
        c = p1y

    else:
        m = (p2y - p1y) / (p2x - p1x)
        c = p1y - m*p1x

    return m, c


def point_to_line_dist(m, c, x, y):

    m_new = 1/m
    c_new = y - m_new * x
    sx = (c - c_new) / (m_new - m)
    sy = m* sx + c
    dist = np.sqrt((sx - x)**2 + (sy - y)**2)

    return dist


def edge_map(img, sigma = 3):

    edges = feature.canny(img, sigma)

    return edges

def ransac(img, savelocation):
    # image is a 2D array
    image = img #ndimage.gaussian_filter(img, 2)
    edges = edge_map(image)

    #plt.imshow(edges)
    #plt.title('edge map')
    #plt.show()

    edge_pts = np.array(np.nonzero(edges), dtype=float).T
    edge_pts_xy = edge_pts[:, ::-1]

    ransac_iterations = 500
    ransac_threshold = 2
    n_samples = 2

    ratio = 0

    # perform RANSAC iterations
    for it in range(ransac_iterations):

        # this shows progress
        sys.stdout.write('\r')
        sys.stdout.write('iteration {}/{}'.format(it + 1, ransac_iterations))
        sys.stdout.flush()

        all_indices = np.arange(edge_pts.shape[0])
        np.random.shuffle(all_indices)

        indices_1 = all_indices[:n_samples]
        indices_2 = all_indices[n_samples:]

        maybe_points = edge_pts_xy[indices_1, :]
        test_points = edge_pts_xy[indices_2, :]

        # find a line model for these points
        m, c = fit_line(maybe_points)

        x_list = []
        y_list = []
        num = 0

        # find distance to the model for all testing points
        for ind in range(test_points.shape[0]):

            x0 = test_points[ind, 0]
            y0 = test_points[ind, 1]

            # distance from point to the model
            dist = point_to_line_dist(m, c, x0, y0)

            # check whether it's an inlier or not
            if dist < ransac_threshold:
                num += 1

        # in case a new model is better - cache it
        if num / float(n_samples) > ratio:
            ratio = num / float(n_samples)
            model_m = m
            model_c = c

    x = np.arange(image.shape[1])
    y = model_m * x + model_c

    if m != 0 or c != 0:
        plt.plot(x, y, 'r')

    plt.imshow(image)
    #plt.show()
    plt.savefig(savelocation)

    return model_m, model_c

def calculate_dist(line_param, edge):
    distances = np.zeros(len(edge))

    for row in range(150, len(edge)-150):
        pixel = edge[row]
        y = line_param[0] * row + line_param[1]
        if (y - pixel) > 0:  # only take positive offset
            distances[row] = y - pixel
    dist = np.sum(distances)
    return dist
