import os
import json
import requests
import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from tqdm import tqdm

def load_model(model_id="IDEA-Research/grounding-dino-tiny", device="cuda" if torch.cuda.is_available() else "cpu"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    return processor, model

def detect_objects_batch(images, image_paths, processor, model, device, box_threshold=0.4, text_threshold=0.3):
    """
    Process a batch of images to detect objects
    
    Args:
        images: List of PIL Images
        image_paths: List of corresponding image paths
        processor: The model processor
        model: The detection model
        device: The device to use (cuda/cpu)
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text
        
    Returns:
        Dictionary mapping file paths to detection results
    """
    batch_results = {}
    
    if not images:
        return batch_results
    
    # Prepare text prompt for all images in batch
    text_labels = [["an airplane", "a bear", "a bird", "a boat", "a car", "a cat", "a cow", "a deer", "a dog", "a duck", "an earless seal", "an elephant", "a fish", "a flying disc", "a fox", "a frog", "a giant panda", "a giraffe", "a horse", "a leopard", "a lizard", "a monkey", "a motorbike", "a mouse", "a parrot", "a person", "a rabbit", "a shark", "a skateboard", "a snake", "a snowboard", "a squirrel", "a surfboard", "a tennis racket", "a tiger", "a train", "a truck", "a turtle", "a whale", "a zebra"] for _ in range(len(images))]
    
    # Process batch
    inputs = processor(images=images, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process each image result
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img.size[::-1] for img in images]
    )
    
    # Extract results for each image in batch
    for i, (result, img_path) in enumerate(zip(results, image_paths)):
        detections = []
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            detection = {
                "label": labels,
                "confidence": round(score.item(), 3),
                "box": box
            }
            detections.append(detection)
        
        batch_results[str(img_path)] = detections
    
    return batch_results

def load_dataset_split(json_path):
    """
    Load a dataset split from a JSON file
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Set of image filenames to process
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'images' not in data:
        raise ValueError("JSON file must contain an 'images' key with a list of image objects")
    
    # Extract filenames from the images list
    filenames = set()
    for image_obj in data['images']:
        if 'file_name' in image_obj:
            filenames.add(image_obj['file_name'])
    
    print(f"Loaded {len(filenames)} image filenames from dataset split")
    return filenames

def process_directory(root_dir, processor, model, device, box_threshold=0.4, text_threshold=0.3, 
                      batch_size=4, dataset_split=None):
    """
    Process all images in a directory structure and detect objects using batched processing
    
    Args:
        root_dir: Path to the root directory containing subdirectories of images
        processor: The model processor
        model: The detection model
        device: The device to use (cuda/cpu)
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text
        batch_size: Number of images to process in a single batch
        dataset_split: Optional set of filenames to process (if provided, only process these files)
        
    Returns:
        Dictionary mapping file paths to detection results
    """
    results = {}
    root_path = Path(root_dir)
    
    # Get all image files recursively
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = []
    
    # If we have dataset_split, collect files that match the split
    if dataset_split:
        for file_path in dataset_split:
            # Split into dir and filename parts (format: "dir/file_name")
            parts = file_path.split('/')
            if len(parts) == 2:
                subdir, filename = parts
                # Check if the file exists in the root_dir/subdir path
                full_path = root_path / subdir / filename
                if full_path.is_file() and full_path.suffix.lower() in image_extensions:
                    image_files.append(full_path)
        
        print(f"Found {len(image_files)} images matching the dataset split (out of {len(dataset_split)} in split)")
    
    # If no dataset_split or no matching files found, collect all image files
    else:
        for path in root_path.rglob('*'):
            if path.is_file() and path.suffix.lower() in image_extensions:
                image_files.append(path)
        
        print(f"Found {len(image_files)} images to process")
    
    # Process images in batches
    batch_images = []
    batch_paths = []
    
    for i, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            image = Image.open(img_path).convert("RGB")
            rel_path = img_path.relative_to(root_path)
            
            batch_images.append(image)
            batch_paths.append(rel_path)
            
            # Process batch when it reaches the batch size or at the end
            if len(batch_images) >= batch_size or i == len(image_files) - 1:
                batch_results = detect_objects_batch(
                    batch_images,
                    batch_paths,
                    processor,
                    model,
                    device,
                    box_threshold,
                    text_threshold
                )
                
                # Update overall results and provide summary
                results.update(batch_results)
                
                for path, detections in batch_results.items():
                    print(f"Processed {path}: Found {len(detections)} objects")
                
                # Reset batch
                batch_images = []
                batch_paths = []
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Detect objects in directories of images")
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing images")
    parser.add_argument("--box_threshold", type=float, default=0.2, help="Confidence threshold for bounding boxes")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="Confidence threshold for text")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of images to process in a batch")
    parser.add_argument("--dataset_split", type=str, default=None, help="JSON file with dataset split information")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file to save results")
    
    args = parser.parse_args()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor, model = load_model(device=device)
    
    # Load dataset split if provided
    dataset_filenames = None
    if args.dataset_split:
        dataset_filenames = load_dataset_split(args.dataset_split)
    
    # Process directory
    results = process_directory(
        args.root_dir,
        processor,
        model,
        device,
        args.box_threshold,
        args.text_threshold,
        args.batch_size,
        dataset_filenames
    )
    
    # Save results if output specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()