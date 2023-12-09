import os
import cv2
from image_processing import func

# Paths
source_path = './data'    # Source directory where original data is stored
output_path = 'data2'     # Destination directory for preprocessed data

# Create the output directory
os.makedirs(output_path, exist_ok=True)

# Process each class directory in the source data directory
for class_name in os.listdir(source_path):
    class_dir = os.path.join(source_path, class_name)
    if not os.path.isdir(class_dir):
        continue

    # Create a corresponding class directory in the output path
    output_class_dir = os.path.join(output_path, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    # Process and save each image in the class directory
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)

        # Read and process the image
        img = cv2.imread(file_path, 0)  # Assuming grayscale read
        if img is None:
            print(f"Failed to read image: {file_path}")
            continue

        try:

            processed_image = func(file_path)  # Apply your processing function


        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            continue

        # Save the processed image to the output class directory
        output_file_path = os.path.join(output_class_dir, file_name)
        cv2.imwrite(output_file_path, processed_image)
        print(f"Saved processed image to: {output_file_path}")

print("Data processing complete. Preprocessed data saved in:", output_path)
