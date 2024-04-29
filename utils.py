import numpy as np
import matplotlib.pyplot as plt
import cv2


def rgb2grayscale(image):
    """
    Args:
        image (np.array): RGB image of shape [H, W, 3] (np.array of np.uint8 type)
    Returns:
        image (np.array): grayscale image of shape [H, W]
    """
    image = np.dot(image, [0.299, 0.587, 0.114]).astype(dtype=np.uint8)
    
    return np.clip(image, 0, 255)


def grayscale2rgb(image):
    """
    Args:
        image (np.array): grayscale image of shape [H, W]
    Returns:
        image (np.array): RGB image of shape [H, W, 3]
    """
    return np.stack((image,)*3, axis=-1)


def read_image(image_path):
    """
    Args:
        image_path (str): path to the image
    Returns:
        image (np.array): image of shape [H, W] (grayscale image)
    """
    image = plt.imread(image_path)

    if len(image.shape) == 3:
        image = rgb2grayscale(image)

    return image


def show_image(image, axis=None, as_gray=False):
    if axis is None:
        axis = plt
    if as_gray:
        axis.imshow(image, 'gray')
    else:
        axis.imshow(image)

    axis.axis('off')


def show_contours(image, contours, axis=None, contour_color=(255, 0, 0)):
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()

    # Ensure consistent data type for contours (list of np.int32 arrays)
    contours_int32 = [contour.reshape(-1, 2).astype(np.int32) for contour in contours]

    image_with_contours = cv2.drawContours(image_rgb, contours_int32, -1, contour_color, 2)
    show_image(image_with_contours, axis=axis, as_gray=False)


def show_corners(image, corners, axis=None, color='r'):
    show_image(image, axis=axis, as_gray=True)

    if axis is None:
        axis = plt

    axis.scatter(x=[point[0] for point in corners],
                 y=[point[1] for point in corners],
                 c=color, marker="x")

    axis.plot([point[0] for point in corners] + [corners[0][0]],
              [point[1] for point in corners] + [corners[0][1]],
              c=color)


def show_sudoku_cells(sudoku_cells):
    num_cells = sudoku_cells.shape[0]
    figure, axes = plt.subplots(nrows=num_cells, ncols=num_cells, figsize=(num_cells, num_cells))

    for i in range(num_cells):
        for j in range(num_cells):
            if np.min(sudoku_cells[i][j]) == np.max(sudoku_cells[i][j]):
                show_image(np.ones((*sudoku_cells[i][j].shape, 3)), axis=axes[i][j])
            else:
                show_image(sudoku_cells[i][j], axis=axes[i][j], as_gray=True)
