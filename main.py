import os
from typing import List, Tuple
import tkinter as tk
from tkinter import filedialog
import cv2
import face_recognition
import numpy as np

def select_file(window_title: str = "Select a file", file_types: List[Tuple[str, str]] = None) -> str:
    """
    Opens a file selection dialog to allow the user to choose a file.

    Args:
        window_title (str): The title of the file selection dialog window. Defaults to "Select a file".
        file_types (List[Tuple[str, str]]): List of tuples defining file type filters (e.g., [("Images", "*.jpg")]).
                                           Defaults to image file types.

    Returns:
        str: The full path of the selected file as a string. If the user cancels, an empty string is returned.
    """
    if file_types is None:
        file_types = [("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")]

    # Create a root window (it won't be displayed)
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.title(window_title)

    # Open a file dialog and store the selected file path
    current_dir = os.getcwd()
    file_path = filedialog.askopenfilename(
        title=window_title,
        initialdir=current_dir,
        filetypes=file_types
    )

    return file_path

def extract_face_landmarks(image_path: str) -> Tuple[np.ndarray, List[dict], List[Tuple[int, int, int, int]]]:
    """
    Detect face landmarks from the image and return the points.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Tuple[np.ndarray, List[dict], List[Tuple[int, int, int, int]]]:
            - The image as a NumPy array.
            - List of face landmarks (dict of feature points).
            - List of face locations (top, right, bottom, left) for each detected face.
    """
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_landmarks = face_recognition.face_landmarks(rgb_image, face_locations)
    return image, face_landmarks, face_locations

def select_face(image: np.ndarray, face_landmarks: List[dict], face_locations: List[Tuple[int, int, int, int]]) -> Tuple[dict, Tuple[int, int, int, int]]:
    """
    Allow the user to select a face to swap.

    Args:
        image (np.ndarray): The image in which faces were detected.
        face_landmarks (List[dict]): List of detected face landmarks.
        face_locations (List[Tuple[int, int, int, int]]): List of face locations.

    Returns:
        Tuple[dict, Tuple[int, int, int, int]]: The selected face's landmarks and location.
    """
    for idx, location in enumerate(face_locations):
        top, right, bottom, left = location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, f"Face {idx}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Select Face", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    face_idx = int(input("Enter the face number to swap: "))
    return face_landmarks[face_idx], face_locations[face_idx]

def warp_face(source_landmarks: dict, target_landmarks: dict) -> np.ndarray:
    """
    Warp the source face to match the target face's landmarks.

    Args:
        source_landmarks (dict): Landmarks of the source face.
        target_landmarks (dict): Landmarks of the target face.

    Returns:
        np.ndarray: The warped source face image.
    """
    # Extract specific key points from landmarks (e.g., nose, eyes)
    source_points = np.array(
        [source_landmarks[key][0] for key in ["nose_tip", "left_eye", "right_eye"]],
        dtype=np.float32
    )
    target_points = np.array(
        [target_landmarks[key][0] for key in ["nose_tip", "left_eye", "right_eye"]],
        dtype=np.float32
    )

    # Compute an affine transformation matrix
    warp_matrix = cv2.getAffineTransform(source_points, target_points)
    return warp_matrix




def swap_faces(
    original_image: np.ndarray, source_image: np.ndarray,
    target_face_landmarks: dict, target_face_location: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Overlay the swapped face onto the original image.

    Args:
        original_image (np.ndarray): The original image containing the target face.
        source_image (np.ndarray): The source image with the face to swap.
        target_face_landmarks (dict): Landmarks of the target face.
        target_face_location (Tuple[int, int, int, int]): Location of the target face.

    Returns:
        np.ndarray: The image with the swapped face.
    """
    top, right, bottom, left = target_face_location
    resized_face = cv2.resize(source_image, (right - left, bottom - top))
    original_image[top:bottom, left:right] = resized_face
    return original_image

def main() -> None:
    """
    Main function to perform face swapping.
    """
    print("Select the original image with multiple faces.")
    original_image_path = select_file("Select Original Image")

    print("Select the source face image.")
    source_image_path = select_file("Select Source Face Image")

    # Extract landmarks and locations for original and source images
    original_image, original_landmarks_list, original_locations = extract_face_landmarks(original_image_path)
    source_image, source_landmarks_list, _ = extract_face_landmarks(source_image_path)

    if not original_landmarks_list or not source_landmarks_list:
        print("No faces detected in one of the images. Exiting.")
        return

    # Select a face to swap in the original image
    target_landmarks, target_location = select_face(
        original_image.copy(), original_landmarks_list, original_locations
    )

    # Select the first face in the source image
    source_landmarks = source_landmarks_list[0]

    # Warp and swap the face
    warp_matrix = warp_face(source_landmarks, target_landmarks)
    source_face = source_image  # Use the entire source image (can crop specific face if needed)
    h, w = original_image.shape[:2]

    # Apply warp matrix to the source image
    warped_source_face = cv2.warpAffine(source_face, warp_matrix, (w, h))

    # Overlay the swapped face
    result = swap_faces(original_image.copy(), warped_source_face, target_landmarks, target_location)

    # Show and save the result
    cv2.imshow("Swapped Image", result)
    output_path = "swapped_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"Swapped image saved to {output_path}.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
