#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm


def crop_images(input_dir, output_dir, top_ratio=0.25, bottom_ratio=0.0):
    """
    Crop out (remove) the top and/or bottom portions of all images in the input directory 
    and save to output directory.
    
    Args:
        input_dir (Path): Directory containing images to crop
        output_dir (Path): Directory to save cropped images
        top_ratio (float): Portion of image to remove from the top (e.g., 0.25 = remove top quarter)
        bottom_ratio (float): Portion of image to remove from the bottom (e.g., 0.25 = remove bottom quarter)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    
    # Process each image
    for img_path in tqdm(image_files, desc=f"Processing {input_dir.name}"):
        # Determine the relative path to maintain directory structure
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        
        # Create output subdirectory if needed
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        
        # Get dimensions
        height, width = img.shape[:2]
        
        # Calculate height of the portions to remove
        top_crop_height = int(height * top_ratio)
        bottom_crop_height = int(height * bottom_ratio)
        
        # Crop the image (removing both top and bottom portions if specified)
        if bottom_crop_height > 0:
            cropped_img = img[top_crop_height:(height - bottom_crop_height), :]
        else:
            cropped_img = img[top_crop_height:, :]
        
        # Save the cropped image
        cv2.imwrite(str(out_path), cropped_img)


def process_cuhk03_dataset(input_root, output_root, top_ratio, bottom_ratio):
    """
    Process the Cuhk03 dataset, handling both 'detected' and 'labeled' subdatasets.
    
    Args:
        input_root (Path): Root directory of Cuhk03 dataset
        output_root (Path): Root directory for saving cropped images
        top_ratio (float): Portion of image to remove from the top
        bottom_ratio (float): Portion of image to remove from the bottom
    """
    # Check for labeled and detected subdirectories
    for subset in ['labeled', 'detected']:
        subset_dir = input_root / subset
        if not subset_dir.exists():
            print(f"Warning: {subset} directory not found in {input_root}")
            continue
        
        # Process foreground_query directory
        query_dir = subset_dir / 'foreground_query'
        if query_dir.exists():
            out_query_dir = output_root / subset / 'foreground_query'
            crop_images(query_dir, out_query_dir, top_ratio, bottom_ratio)
        
        # Process foregroundImages_cropped directory
        gallery_dir = subset_dir / 'foregroundImages_cropped'
        if gallery_dir.exists():
            out_gallery_dir = output_root / subset / 'foregroundImages_cropped'
            crop_images(gallery_dir, out_gallery_dir, top_ratio, bottom_ratio)


def main():
    parser = argparse.ArgumentParser(description='Crop out the top and/or bottom portions of images in Cuhk03 dataset')
    parser.add_argument('--input', type=str, required=True, 
                        help='Root directory of the Cuhk03 dataset (containing labeled and detected subdirectories)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory to save cropped images')
    parser.add_argument('--top-ratio', type=float, default=0.4,
                        help='Portion of image to remove from the top (default: 0.4 for top 40%)')
    parser.add_argument('--bottom-ratio', type=float, default=0.575,
                        help='Portion of image to remove from the bottom (default: 0.575 for bottom 57.5%)')
    
    args = parser.parse_args()
    
    input_root = Path(args.input)
    output_root = Path(args.output)
    
    if not input_root.exists():
        print(f"Error: Input directory {input_root} does not exist")
        return
    
    # Process the dataset
    process_cuhk03_dataset(input_root, output_root, args.top_ratio, args.bottom_ratio)
    print(f"Processing complete. Cropped images saved to {output_root}")


if __name__ == "__main__":
    main()
