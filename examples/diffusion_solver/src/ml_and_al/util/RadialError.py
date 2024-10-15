import cv2
import numpy as np


def get_sources_centers(image):
    image = np.array(255 * image, dtype=np.uint8)
    ret, binary = cv2.threshold(image, 1, 255, 0)

    # https://stackoverflow.com/a/55806272
    major = cv2.__version__.split('.')[0]
    if major == '3':
        ret, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    centers = []

    for c in contours:
        # moment of inertia of the countour
        M = cv2.moments(c)

        # x, y
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]

        # cv2.circle(binary, (int(center_x), int(center_y)), 20, (255,255,255))

        centers.append((round(center_x), round(center_y)))
    # cv2.imshow('im', binary)
    return centers


def create_circular_mask(center, radius=1, h=512, w=512):
    # https://newbedev.com/how-can-i-create-a-circular-mask-for-a-numpy-array

    Y, X = np.ogrid[:h, :w]

    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def create_ring_mask(center, radius=1, dr=5, h=512, w=512):
    inner = create_circular_mask(center, radius=radius, h=h, w=w)

    outer = create_circular_mask(center, radius=radius + dr, h=h, w=w)

    mask = outer * np.where(inner, False, True)

    return mask


def create_nan_circular_mask(center, radius=1, h=512, w=512):
    mask = create_circular_mask(center, radius=radius, h=h, w=w)

    return np.where(mask, 1.0, float('nan'))


# @torch.no_grad()
def per_center_MAE(src_img, target_img, pred_im, radius_list,
                   h=512, w=512):
    centers = get_sources_centers(src_img)

    error_dict = {c: [] for c in centers}

    for c in centers:
        for r in radius_list:
            mask = create_nan_circular_mask(c, r, h=h, w=w)
            error = np.nanmean(mask * np.abs(target_img - pred_im))
            error_dict[c].append([r, error])
    return error_dict


def radial_differences(target, predicted, center, r=50):
    """
    Computes the difference of target - predicted in the x and y axis. The difference is computed from center_x - r
    to center_x + r (same for the y direction). If center +- r is outside the image it computes the difference to the 
    edge.

    Parameters
    ----------
    target : numpy.array 
        Target image of the neural net. Must be already detached, in cpu mode, converted to numpy and be 2D.
    predicted : numpy.array 
        Predicted image by the neural net. Must be already detached, in cpu mode, converted to numpy and be 2D.
    center : tupple / list
        (x, y) coordinates of the center.
    r : int, optional
        +- distance from the center the difference will be computed. The default is 50.

    Returns
    -------
    diff_x : numpy.array
        The difference of target-predicted in the x direction
    xlims : tupple
        The used min and max x in the difference calculation
    diff_y : numpy.array
        The difference of target-predicted in the y direction
    ylims : tupple
        The used min and max y in the difference calculation

    """
    c = center
    big_x = min(c[0] + r, target.shape[0])
    small_x = max(c[0] - r, 0)

    big_y = min(c[1] + r, target.shape[1])
    small_y = max(c[1] - r, 0)

    diff_x = target[small_x:big_x, c[1]] - predicted[small_x:big_x, c[1]]

    diff_y = target[c[0], small_y:big_y] - predicted[c[0], small_y:big_y]

    xlims = (small_x, big_x)

    ylims = (small_y, big_y)

    return diff_x, xlims, diff_y, ylims


def get_radial_differences(src_img, target_img, pred_img):
    centers = get_sources_centers(src_img[0, :, :].detach().cpu().numpy())

    error_dict = {c: {'difference in x direction': None, 'xlims': None,
                      'difference in y direction': None, 'ylims': None} for c in centers}

    for c in centers:
        diff_x, xlims, diff_y, ylims = radial_differences(target_img[0, :, :].detach().cpu().numpy(),
                                                          pred_img[0, :, :].detach().cpu().numpy(),
                                                          c)
        error_dict[c]['difference in x direction'] = diff_x
        error_dict[c]['xlims'] = xlims

        error_dict[c]['difference in y direction'] = diff_y

    return error_dict


if __name__ == '__main__':
    from loaders import generateDatasets
    import torch
    import matplotlib.pyplot as plt

    PATH = r'D:\Google Drive IU\phdStuff\AI-project-with-javier\diffusion project'
    device = torch.device("cpu")

    with torch.no_grad():
        _, testloader = generateDatasets(PATH, datasetName='sample-set',
                                         batch_size=1,
                                         num_workers=1,
                                         std_tr=0.0, s=512).getDataLoaders()
        b = next(iter(testloader))
        x = b[0].to(device)
        target = b[1].to(device)
        # print(x.shape)

        centers = get_sources_centers(x[0, 0, :, :].detach().cpu().numpy())

        c = centers[0]

        predicted = np.random.uniform(size=x[0, 0, :, :].shape)

        diff_x, xlims, diff_y, ylims = radial_differences(target[0, 0, :, :].detach().cpu().numpy(), predicted, c)

        xdist = np.arange(xlims[0], xlims[1])
        ydist = np.arange(ylims[0], ylims[1])
        plt.figure()
        plt.plot(xdist, diff_x)
        plt.figure()
        plt.plot(ydist, diff_y)
