from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    x_centered = x - np.mean(x, axis=0)
    return x_centered

#x = load_and_center_dataset('YaleB_32x32.npy')


def get_covariance(dataset):
    y= dataset
    S = np.dot(np.transpose(y), y) / (len(y)-1)
    return S

#S = get_covariance(x)


def get_eig(S, m):
    n= len(S)
    w, v = eigh(S, subset_by_index=[n-m, n-1])
    w_2 = np.flip(w)
    v_2 = np.fliplr(v)
    return w_2, v_2

#Lambda, U = get_eig(S, 2)

def get_eig_prop(S, prop):
    trace= np.trace(S)
    w, v = eigh(S, subset_by_value=[prop*trace, np.inf])
    w_2 = np.flip(w)
    v_2 = np.fliplr(v)
    return w_2, v_2

#Lambda_2, U_2 = get_eig_prop(S, 0.07)

def project_image(image, U):
    m = len(U[0])
    sum = 0
    for j in range(0,m):
        u_j = U[:][:,j]
        alpha_j = np.dot(np.transpose(u_j), image)
        sum += alpha_j * u_j
    return sum

#projection = project_image(x[0], U)

from mpl_toolkits.axes_grid1 import make_axes_locatable

def display_image(orig, proj):
    reshape_orig = np.reshape(orig, (32,32)).transpose()
    reshape_projection = np.reshape(proj, (32,32)).transpose()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Original')
    #im1 = ax1.imshow(reshape_orig, aspect='equal')

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im1 = ax1.imshow(reshape_orig, aspect='equal')
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    ax2.set_title('Projection')
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    im2 = ax2.imshow(reshape_projection, aspect='equal')
    fig.colorbar(im1, cax=cax2, orientation='vertical')

    plt.show()