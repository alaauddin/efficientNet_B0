import cv2
import numpy as np
import os

def preprocess_paper_detection(image_path, output_dir_canny, output_dir_dilated):
    """
    Preprocess an image and save Canny and Dilated edge outputs separately.

    Parameters:
        image_path (str): Path to the input image.
        output_dir_canny (str): Directory to save Canny edge images.
        output_dir_dilated (str): Directory to save Dilated edge images.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Warning: Image not found or can't be loaded: {image_path}")
        return

    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)

    # 3. Canny Edge Detection
    edges = cv2.Canny(clahe_image, threshold1=30, threshold2=100)

    # 4. Morphological Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Prepare filenames
    filename = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename)[0]

    # Ensure output directories exist
    os.makedirs(output_dir_canny, exist_ok=True)
    os.makedirs(output_dir_dilated, exist_ok=True)

    # Save images
    cv2.imwrite(os.path.join(output_dir_canny, f"{filename_no_ext}_3_canny_edges.jpg"), edges)
    cv2.imwrite(os.path.join(output_dir_dilated, f"{filename_no_ext}_4_dilated_edges.jpg"), dilated_edges)

    print(f"[✓] Processed {filename}")

def batch_process():
    base_dataset_path = "C://Users//Owner//Desktop//preprocessing//dataset"
    base_output_path = "preprocessing"

    classes = ["no_paper", "with_paper"]

    for cls in classes:
        input_dir = os.path.join(base_dataset_path, cls)
        output_canny_dir = os.path.join(base_output_path, cls, "3_canny_edges")
        output_dilated_dir = os.path.join(base_output_path, cls, "4_dilated_edges")

        if not os.path.exists(input_dir):
            print(f"⚠️ Input directory does not exist: {input_dir}")
            continue

        print(f"Processing class '{cls}' images from {input_dir}")

        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                preprocess_paper_detection(img_path, output_canny_dir, output_dilated_dir)

if __name__ == "__main__":
    batch_process()
