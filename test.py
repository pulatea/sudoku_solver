import os
import argparse
import numpy as np

from skimage.io import imsave

from pipeline import get_test_pipeline

from utils import read_image
from recognition import load_templates

from sudoku_solver import matrix_to_puzzle, solve_sudoku

# BEGIN YOUR IMPORTS
from frontalization import frontalize_image
from recognition import get_sudoku_cells
from recognition import recognize_digits
import cv2
# END YOUR IMPORTS

FRONTALIZED_IMAGES_PATH = os.path.join(".", "frontalized_images")

CEND = '\33[0m'
CBOLD = '\33[1m'

CRED = '\33[31m'
CGREEN = '\33[32m'
CYELLOW = '\33[33m'


def get_recognition_error(sudoku_matrix, gt_sudoku_matrix):
    return np.sum(sudoku_matrix != gt_sudoku_matrix)


def get_recognition_error_str(recognition_error):
    if recognition_error == 0:
        return CBOLD + CGREEN + f"{recognition_error}" + CEND
    elif recognition_error <= 3:
        return CBOLD + CYELLOW + f"{recognition_error}" + CEND
    else:
        return CBOLD + CRED + f"{recognition_error}" + CEND


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--path', type=str, required=True, help="path to the folder with images and Sudoku matrix files")
    args = parser.parse_args()

    image_paths = [os.path.join(args.path, file_name) for file_name in sorted(os.listdir(args.path))
                   if 'jpg' in os.path.splitext(file_name)[1]]
    sudoku_matrix_paths = [os.path.join(args.path, file_name) for file_name in sorted(os.listdir(args.path))
                           if 'npy' in os.path.splitext(file_name)[1]]
    
    pipeline = get_test_pipeline()
    recognition_errors = []

    # print(args.path)

    for image_path, sudoku_matrix_path in zip(image_paths, sudoku_matrix_paths):
        print("-"*20)
        print(f"For Sudoku in the image {image_path}")
        sudoku_image = read_image(image_path=image_path)

        # BEGIN YOUR CODE

        frontalized_image = frontalize_image(sudoku_image, pipeline)# YOUR CODE
        
        # END YOUR CODE

        os.makedirs(FRONTALIZED_IMAGES_PATH, exist_ok=True)
        frontalized_image_path = os.path.join(FRONTALIZED_IMAGES_PATH, f"frontalized_{os.path.split(image_path)[1]}")
        print("Frontalized image path", frontalized_image_path)
        imsave(frontalized_image_path, frontalized_image)
        print(f"You can find the frontalized image at {frontalized_image_path}")
        print("-"*20)

        templates_dict = load_templates(TEMPLATES_PATH="./templates")
        # BEGIN YOUR CODE

        sudoku_cells = get_sudoku_cells(frontalized_image)# YOUR CODE
        sudoku_matrix = recognize_digits(sudoku_cells, templates_dict)# YOUR CODE

        # END YOUR CODE

        gt_sudoku_matrix = np.load(sudoku_matrix_path)

        print(f"Your Sudoku matrix is")
        print(matrix_to_puzzle(sudoku_matrix))
        print()
        print("Ground truth Sudoku matrix is")
        print(matrix_to_puzzle(gt_sudoku_matrix))
        print()

        recognition_error = get_recognition_error(sudoku_matrix, gt_sudoku_matrix)
        recognition_errors.append(recognition_error)
        print(f"There are {get_recognition_error_str(recognition_error)} cells with recognition error")
        print("-"*20)

        if recognition_error == 0:
            sudoku_matrix_solved = solve_sudoku(sudoku_matrix)
            print("The solved Sudoku puzzle is")
            print(matrix_to_puzzle(sudoku_matrix_solved))
        else:
            print("The recognized Sudoku matrix contains errors and cannot be solved")
        print()

    print(f"Successfully recognized {sum(np.array(recognition_errors) <= 3)} out of {len(recognition_errors)} Sudoku grids")


if __name__ == "__main__":
    main()
