import numpy as np

# TODO(L): Determine good variance (might depend on radius)
def get_gauss_kernel(radius, var=None):
    if var is None:
        var = _get_default_var(radius)
    size = 2*radius + 1

    dist_to_center = np.arange(-radius, radius+1) # (size,)
    dist_x, dist_y = np.meshgrid(dist_to_center, dist_to_center) # (size, size)

    K = np.exp(-0.5 * (dist_x**2 + dist_y**2)/var)
    return K

def get_gauss_dijkstra_kernel(dist, var=None):
    # dist, np.array, (size, size)
    # assert np.all(dist > 0), dist
    if var is None:
        radius = int(dist.shape[0]//2)
        var = _get_default_var(radius)
    K = np.exp(-0.5 * dist**2 / var)
    return K

def _get_default_var(radius):
    return 2*radius*radius