import os

from tqdm import tqdm

from skimage.io import imsave

from frontalization import frontalize_image
from pipeline import get_test_pipeline

from utils import read_image
from recognition import TEMPLATES_PATH

# BEGIN YOUR IMPORTS
from recognition import get_sudoku_cells

# END YOUR IMPORTS

IMAGES_PATH = os.path.join(".", "sudoku_puzzles", "train")

# BEGIN YOUR CODE

""""I've used coordinates of images 0-5 to create templates, enhancing the recognition of Sudoku grids in the results."""

CELL_COORDINATES = {
    "image_0.jpg": {
        '1': (6, 4),
        '2': (2, 8),
        '3': (2, 1),
        '4': (1, 4),
        '5': (4, 0),
        '6': (4, 4),
        '7': (2, 4),
        '8': (1, 8),
        '9': (2, 0)
    },
    "image_1.jpg": {
        '1': (1, 3),
        '2': (0, 1),
        '3': (1, 1),
        '4': (2, 0),
        '5': (3, 0),
        '6': (1, 8),
        '7': (1, 6),
        '8': (1, 0),
        '9': (4, 3)
    },
    "image_2.jpg": {
        '1': (0, 7),
        '2': (3, 0),
        '3': (3, 3),
        '4': (6, 0),
        '5': (3, 2),
        '6': (2, 5),
        '7': (0, 3),
        '8': (0, 4),
        '9': (7, 0)
    },
    "image_3.jpg": {
        '1': (0, 0),
        '2': (2, 6),
        '3': (0, 1),
        '4': (0, 7),
        '5': (0, 2),
        '6': (1, 1),
        '7': (2, 0),
        '8': (4, 3),
        '9': (8, 1)
    },
    "image_4.jpg": {
        '1': (3, 0),
        '2': (1, 0),
        '3': (2, 1),
        '4': (3, 2),
        '5': (2, 0),
        '6': (0, 2),
        '7': (1, 2),
        '8': (6, 0),
        '9': (4, 0)
    },
    "image_5.jpg": {
        '1': (3, 2),
        '2': (2, 3),
        '3': (8, 3),
        '4': (0, 0),
        '5': (1, 1),
        '6': (5, 0),
        '7': (0, 5),
        '8': (8, 0),
        '9': (3, 3)
    }}
"""The coordinates below SHOULD NOT be used for creating templates!"""
# "image_6.jpg": {
#     '1': (0, 0),
#     '2': (2, 1),
#     '3': (1, 3),
#     '4': (2, 2),
#     '5': (3, 3),
#     '6': (0, 8),
#     # '7': (, ),
#     '8': (5, 1),
#     '9': (8, 0)
# },
# "image_7.jpg": {
#     '1': (0, 0),
#     '2': (1, 8),
#     '3': (0, 8),
#     '4': (1, 0),
#     '5': (4, 1),
#     '6': (1, 4),
#     '7': (0, 7),
#     '8': (0, 1),
#     '9': (8, 0)
# },
# "image_8.jpg": {
#     '1': (6, 0),
#     '2': (2, 0),
#     '3': (0, 5),
#     '4': (0, 0),
#     '5': (0, 3),
#     '6': (0, 2),
#     '7': (8, 5),
#     '8': (8, 0),
#     '9': (8, 8)
# },
# "image_9.jpg": {
#     '1': (2, 4),
#     '2': (6, 8),
#     '3': (0, 3),
#     '4': (5, 0),
#     '5': (6, 0),
#     '6': (4, 2),
#     '7': (1, 1),
#     '8': (6, 4),
#     '9': (7, 1)
# }
#


def main():
    os.makedirs(TEMPLATES_PATH, exist_ok=True)

    pipeline = get_test_pipeline()

    for file_name, coordinates_dict in CELL_COORDINATES.items():
        image_path = os.path.join(IMAGES_PATH, file_name)
        sudoku_image = read_image(image_path=image_path)

        # BEGIN YOUR CODE

        frontalized_image = frontalize_image(sudoku_image, pipeline)
        sudoku_cells = get_sudoku_cells(frontalized_image)

        # END YOUR CODE

        for digit, coordinates in tqdm(coordinates_dict.items(), desc=file_name):
            print("TEMPLATES PATH IS", TEMPLATES_PATH)
            digit_templates_path = os.path.join(TEMPLATES_PATH, digit)
            os.makedirs(digit_templates_path, exist_ok=True)

            digit_template_path = os.path.join(digit_templates_path, f"{os.path.splitext(file_name)[0]}_{digit}.jpg")
            imsave(digit_template_path, sudoku_cells[coordinates])


if __name__ == "__main__":
    main()
