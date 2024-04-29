import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from utils import read_image, show_image


# BEGIN YOUR IMPORTS

# END YOUR IMPORTS


def find_edges(image):
    """
    Args:
        image (np.array): (grayscale) image of shape [H, W]
    Returns:
        edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE

    # Edge detection using Canny algorithm, which identifies changes in intensity that represent edges in the image
    edges = cv2.Canny(image, threshold1=100, threshold2=200)  # YOUR CODE
    # Contour retrieval
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # END YOUR CODE

    return edges


def highlight_edges(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlited_edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR
    # Kernel of shape 3x3 to apply transformation such as dilation in the next step
    kernel = np.ones((3, 3), np.uint8)
    # Dilates the edges using the aforementioned Kernel
    highlited_edges = cv2.dilate(edges, kernel, iterations=1)  # YOUR CODE

    # END YOUR CODE

    return highlited_edges


def find_contours(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours (List[np.array]): list of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    """
    # Find contours using OpenCV findContours function
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure consistent data type for contours (list of np.int32 arrays)
    contours = [contour.astype(np.int32) for contour in contours]

    return contours

def get_max_contour(contours):
    """
    Args:
        contours (List[np.array]): list of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    Returns:
        max_contour (np.array): an array of points (vertices) of the contour with the maximum area of shape [N, 1, 2]
    """
    # Ensure contours is not empty

    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour

def order_corners(corners):
    """
    Args:
        corners (np.array): an array of corner points (corners) of shape [4, 2]
    Returns:
        ordered_corners (np.array): an array of corner points in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    # squeezed corners in order to make sure that all entries are of same dimension
    corners = np.squeeze(corners)

    # calculates the centroid by finding the mean in the first axis
    centroid = np.mean(corners, axis=0)
    # calculates the angle between each corner point and centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])

    # corners are sorted in counterclockwise order
    sorted_indices = np.argsort(angles)
    ordered_corners = corners[sorted_indices]

    return ordered_corners

def find_corners(contour, accuracy=0.1):
    """
    Args:
        contour (np.array): an array of points (vertices) of the contour of shape [N, 1, 2]
        accuracy (float): how accurate the contour approximation should be
    Returns:
        ordered_corners (np.array): an array of corner points (corners) of quadrilateral approximation of contour of shape [4, 2]
                                    in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    # The Douglas-Peucker is used to approximate the contours by reducing number of vertices
    epsilon = accuracy * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Reshape all corners to a 2d shape array
    corners = approx.reshape(-1, 2)

    # If there are no four corners, then create a default square
    if len(corners) != 4:
        corners = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

    ordered_corners = order_corners(corners)

    return ordered_corners

def rescale_image(image, scale=0.5):
    """
    Args:
        image (np.array): input image
        scale (float): scale factor
    Returns:
        rescaled_image (np.array): 8-bit (with range [0, 255]) rescaled image
    """
    image = cv2.convertScaleAbs(image)

    rescaled_image = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    return rescaled_image

def gaussian_blur(image, sigma):
    """
    Args:
        image (np.array): input image
        sigma (float): standard deviation for Gaussian kernel
    Returns:
        blurred_image (np.array): 8-bit (with range [0, 255]) blurred image
    """
    # Setting image intensity in the range (0, 255]
    image = cv2.convertScaleAbs(image)
    # Resize image while keeping the quality using bilinear interpolation
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    return blurred_image


def distance(point1, point2):
    """
    Args:
        point1 (np.array): n-dimensional vector
        point2 (np.array): n-dimensional vector
    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    # BEGIN YOUR CODE

    # Calculate Euclidian distance
    distance = np.linalg.norm(point1 - point2)

    # END YOUR CODE

    return distance


def warp_image(image, ordered_corners):
    """
    Args:
        image (np.array): input image
        ordered_corners (np.array): corners in order [top left, top right, bottom right, bottom left]
    Returns:
        warped_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # 4 source points
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # BEGIN YOUR CODE

    # The side length of the Sudoku grid based on distances between corners
    side = max(distance(top_left, top_right), distance(top_left, bottom_left))

    # Define the 4 target points
    destination_points = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.float32)

    # The perspective transformation matrix computation
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners.astype('float32'), destination_points)

    # Warp the image using the perspective transformation matrix
    warped_image = cv2.warpPerspective(image, transform_matrix, (int(side), int(side)))

    # END YOUR CODE

    assert warped_image.shape[0] == warped_image.shape[1], "height and width of the warped image must be equal"
    return warped_image


def frontalize_image(sudoku_image, pipeline):
    """
    Args:
        sudoku_image (np.array): input Sudoku image
        pipeline (Pipeline): Pipeline instance
    Returns:
        frontalized_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # BEGIN YOUR CODE
    processed_image, ordered_corners = pipeline(sudoku_image)
    frontalized_image = warp_image(processed_image, ordered_corners)
    # END YOUR CODE

    return frontalized_image


def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)

    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])

        sudoku_image = read_image(image_path=image_path)
        frontalized_image = frontalize_image(sudoku_image, pipeline)

        show_image(frontalized_image, axis=axis, as_gray=True)
