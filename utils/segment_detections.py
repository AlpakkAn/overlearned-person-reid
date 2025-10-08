import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from collections import defaultdict

from transformers import AutoProcessor, AutoModelForMaskGeneration

class BoundingBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    @property
    def xyxy(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]

class DetectionResult:
    def __init__(self, label, confidence, box, mask=None):
        self.score = confidence
        self.label = label
        self.box = box
        self.mask = mask

    @classmethod
    def from_dict(cls, detection_dict):
        box = detection_dict['box']
        # Handle different box formats from detect.py
        if isinstance(box, list) and len(box) == 4:
            # [x1, y1, x2, y2] format
            bbox = BoundingBox(xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
        elif isinstance(box, dict):
            # Dict format with xmin, ymin, etc.
            bbox = BoundingBox(xmin=box.get('xmin', 0), 
                              ymin=box.get('ymin', 0), 
                              xmax=box.get('xmax', 0), 
                              ymax=box.get('ymax', 0))
        else:
            raise ValueError(f"Unsupported box format: {box}")
            
        return cls(label=detection_dict.get('label', ''),
                  confidence=detection_dict.get('confidence', 0.0),
                  box=bbox)

def get_boxes(results):
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return [boxes]

def refine_masks(masks, polygon_refinement=False):
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def mask_to_polygon(mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:  # Check if contours is empty
        return []  # Return empty list if no contours found

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract the vertices of the contour
    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon, image_shape):
    """
    Convert a polygon to a segmentation mask.
    """
    if not polygon:  # Check if polygon is empty
        return np.zeros(image_shape, dtype=np.uint8)  # Return empty mask
        
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def encode_binary_mask(mask):
    """Encode binary mask as a list of polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return []
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert to list of [x,y] coordinates
    polygon = largest_contour.reshape(-1, 2).tolist()
    
    return polygon

def load_detection_results(input_file):
    """Load detection results from JSON file"""
    with open(input_file, 'r') as f:
        return json.load(f)

def process_batch(batch_data, segmenter, processor, device, polygon_refinement=True):
    """
    Process a batch of image metadata and load images on demand
    
    Args:
        batch_data: List of tuples (image_path, full_image_path, results)
        segmenter: SAM model
        processor: SAM processor
        device: Device to run model on
        polygon_refinement: Whether to refine masks
        
    Returns:
        Dictionary with segmentation results for this batch
    """
    batch_results = {}
    
    for image_path, full_image_path, results in batch_data:
        try:
            # Load image on demand
            image = Image.open(full_image_path).convert("RGB")
            
            image_results = []
            
            # Process each detection for this image
            for result in results:
                # Get bounding box
                boxes = get_boxes(results=[result])
                
                # Generate mask using SAM
                inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)
                outputs = segmenter(**inputs)
                
                masks = processor.post_process_masks(
                    masks=outputs.pred_masks,
                    original_sizes=inputs.original_sizes,
                    reshaped_input_sizes=inputs.reshaped_input_sizes
                )[0]
                
                # Refine masks
                refined_masks = refine_masks(masks, polygon_refinement=polygon_refinement)
                
                if refined_masks:
                    mask = refined_masks[0]
                    
                    # Convert mask to polygon format for compact storage
                    polygon = encode_binary_mask(mask)
                    
                    # Store result with mask as polygon
                    image_results.append({
                        "label": result.label,
                        "confidence": result.score,
                        "box": result.box.xyxy,
                        "mask_polygon": polygon
                    })
                else:
                    # Store result without mask
                    image_results.append({
                        "label": result.label,
                        "confidence": result.score,
                        "box": result.box.xyxy,
                        "mask_polygon": []
                    })
            
            batch_results[str(image_path)] = image_results
            
            # Explicitly delete image to free memory
            del image
            
        except Exception as e:
            print(f"Error processing {full_image_path}: {e}")
    
    return batch_results

def segment_detections_batched(detection_results, image_root_dir, segmenter, processor, device, 
                              batch_size=4, polygon_refinement=True):
    """
    Generate segmentation masks for all detections using batch processing with lazy loading
    
    Args:
        detection_results: Dictionary of detection results from detect.py
        image_root_dir: Root directory for image paths
        segmenter: The SAM model
        processor: The SAM processor
        device: Device to run model on
        batch_size: Number of images to process in each batch
        polygon_refinement: Whether to refine masks with polygon extraction
        
    Returns:
        Dictionary with segmentation results
    """
    segmentation_results = {}
    root_path = Path(image_root_dir)
    
    # Prepare metadata for batching (don't load images yet)
    valid_items = []
    
    print("Preparing image metadata...")
    for image_path, detections in tqdm(detection_results.items(), desc="Validating images"):
        if not detections:  # Skip images with no detections
            continue
            
        # Construct absolute path to image
        full_image_path = root_path / image_path
        
        if not full_image_path.exists():
            print(f"Warning: Image not found: {full_image_path}")
            continue
        
        try:
            # Convert detections to DetectionResult objects
            results = [DetectionResult.from_dict(det) for det in detections]
            
            # Only store metadata, not the actual image
            valid_items.append((image_path, full_image_path, results))
        except Exception as e:
            print(f"Error preparing metadata for {full_image_path}: {e}")
    
    print(f"Found {len(valid_items)} valid images to process")
    
    # Process in batches
    total_batches = (len(valid_items) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(valid_items), batch_size), desc="Processing batches", total=total_batches):
        # Get current batch metadata
        batch = valid_items[i:i+batch_size]
        
        try:
            # Process batch (images will be loaded on demand)
            batch_results = process_batch(batch, segmenter, processor, device, polygon_refinement)
            
            # Update overall results
            segmentation_results.update(batch_results)
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}/{total_batches}: {e}")
    
    return segmentation_results

def main():
    parser = argparse.ArgumentParser(description="Generate segmentation masks for object detections")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to detection results JSON file from detect.py")
    parser.add_argument("--image-root", type=str, required=True,
                        help="Root directory containing images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON file to save segmentation results")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of images to process in each batch")
    parser.add_argument("--no-polygon-refinement", action="store_true",
                        help="Disable polygon refinement of masks")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
        
    # Check if image root directory exists
    if not os.path.exists(args.image_root):
        print(f"Error: Image root directory not found: {args.image_root}")
        return
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load SAM model
    print("Loading Segment Anything Model...")
    model_id = "facebook/sam-vit-base"
    segmenter = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load detection results
    print(f"Loading detection results from {args.input}")
    detection_results = load_detection_results(args.input)
    
    # Generate segmentation masks with batching
    print(f"Generating segmentation masks with batch size {args.batch_size}...")
    segmentation_results = segment_detections_batched(
        detection_results,
        args.image_root,
        segmenter,
        processor,
        device,
        batch_size=args.batch_size,
        polygon_refinement=not args.no_polygon_refinement
    )
    
    # Save results
    print(f"Saving segmentation results to {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(segmentation_results, f, indent=2)
    
    print(f"Segmentation complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
