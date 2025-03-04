import os
import numpy as np
from pathlib import Path
import micasense.imageset as imageset
import micasense.imageutils as imageutils
from micasense.capture import Capture
from skimage.transform import ProjectiveTransform
import matplotlib.pyplot as plt
import datetime

# Set paths
imagePath = Path("./data/50ml_15mheight/000")  # Change this to your folder containing image sets
outputPath = imagePath / 'processed_bands'
os.makedirs(outputPath, exist_ok=True)

# Camera properties and settings
overwrite = False  # Set to True to overwrite existing outputs
generateThumbnails = False  # Set to True to generate RGB thumbnails
panSharpen = True  # Enable pansharpening for compatible cameras
warp_matrices_SIFT = None  # Initialize warp matrices
img_type = "reflectance"  # Set to 'radiance' if irradiance data is unavailable

# Function to generate and save individual bands
def process_and_save_bands(capture, output_dir, img_type, panchroCam=False):
    print(f"Processing capture: {capture.uuid}")
    try:
        # Get aligned or pansharpened stack
        if panchroCam:
            stack, _ = capture.radiometric_pan_sharpened_aligned_capture(warp_matrices=warp_matrices_SIFT, img_type=img_type)
        else:
            stack = capture.create_aligned_capture(img_type=img_type)

        # Iterate over each band and save individually
        for i, band_name in enumerate(capture.band_names):
            band = stack[:, :, i]
            band_min, band_max = np.percentile(band.flatten(), (0.5, 99.5))
            band_normalized = imageutils.normalize(band, band_min, band_max)

            band_output_path = os.path.join(output_dir, f"{capture.uuid}_band_{band_name}.tif")
            plt.imsave(band_output_path, band_normalized, cmap="gray")
            print(f"Saved: {band_output_path}")

        if generateThumbnails:
            thumbnail_path = os.path.join(output_dir, f"{capture.uuid}_thumbnail.jpg")
            capture.save_capture_as_rgb(thumbnail_path)
            print(f"Saved thumbnail: {thumbnail_path}")

    except Exception as e:
        print(f"Error processing capture {capture.uuid}: {e}")

# Load the imageset
print("Loading images...")
imgset = imageset.ImageSet.from_directory(imagePath)
print(f"Loaded {len(imgset.captures)} captures.")

# Check camera model for panchromatic support
if len(imgset.captures) > 0:
    first_capture = imgset.captures[0]
    cam_model = first_capture.camera_model
    panchroCam = cam_model in ['RedEdge-P', 'Altum-PT']
    print(f"Camera model: {cam_model}, Panchromatic: {panchroCam}")

# Load or generate warp matrices
warp_matrices_filename = f"{first_capture.camera_serial}_warp_matrices.npy"
if os.path.exists(warp_matrices_filename):
    print(f"Loading warp matrices from {warp_matrices_filename}")
    warp_matrices_SIFT = np.load(warp_matrices_filename, allow_pickle=True)
    if panchroCam:
        warp_matrices_SIFT = [ProjectiveTransform(matrix=matrix) for matrix in warp_matrices_SIFT]
else:
    print("Warp matrices not found. They will be generated during processing.")

# Process each capture
start_time = datetime.datetime.now()
for capture in imgset.captures:
    output_dir = os.path.join(outputPath, capture.uuid)
    os.makedirs(output_dir, exist_ok=True)
    if not os.listdir(output_dir) or overwrite:
        process_and_save_bands(capture, output_dir, img_type, panchroCam)
    else:
        print(f"Skipping {capture.uuid}, already processed.")

end_time = datetime.datetime.now()
print(f"Processing completed in {end_time - start_time}")


# python batch_processing_script.py --imagepath ./data/20_15/raw_images --outputpath ./data/20_15/output_images --panelpath ./data/20_15/panel_images