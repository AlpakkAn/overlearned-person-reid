import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
import cv2

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

# Add parent directory to path to import Cuhk03
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foreground_dataset import Cuhk03

def extract_bbox(mask):
    """Extract bounding box from mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None  # No mask found
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return rmin, rmax, cmin, cmax

def process_image_segformer(image_path, processor, model, device, output_path, save_original_if_no_clothes=False):
    """Process a single image to extract upper clothes or dress using Segformer."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Process with model
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get segmentation prediction
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        
        # Upsample to original size
        upsampled_logits = F.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        
        # Get the class predictions and probabilities
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        probs = F.softmax(upsampled_logits, dim=1)[0].cpu().numpy()
        
        # Extract upper clothes (class 4) and dress (class 7) masks and confidences
        upper_clothes_mask = pred_seg == 4
        dress_mask = pred_seg == 7
        upper_clothes_confidence = probs[4].max() if np.any(upper_clothes_mask) else 0
        dress_confidence = probs[7].max() if np.any(dress_mask) else 0
        
        # Determine which class to use based on confidence
        if upper_clothes_confidence > dress_confidence:
            selected_mask = upper_clothes_mask
            selected_class = "Upper-clothes"
            selected_confidence = upper_clothes_confidence
        else:
            selected_mask = dress_mask
            selected_class = "Dress"
            selected_confidence = dress_confidence
        
        # If no relevant clothing found, conditionally save original image
        if not np.any(selected_mask):
            if save_original_if_no_clothes:
                print(f"No upper clothes or dress found in {image_path}, saving original image")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
            else:
                print(f"No upper clothes or dress found in {image_path}, skipping")
            return True
        
        print(f"Selected {selected_class} with confidence {selected_confidence:.4f} for {image_path}")
        
        # Extract bounding box
        bbox = extract_bbox(selected_mask)
        if bbox is None:
            if save_original_if_no_clothes:
                print(f"Invalid bounding box for {image_path}, saving original image")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
            else:
                print(f"Invalid bounding box for {image_path}, skipping")
            return True
        
        rmin, rmax, cmin, cmax = bbox
        
        # Convert image to numpy and apply mask
        img_np = np.array(image)
        
        # Crop the image and mask to the bounding box
        cropped_img = img_np[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = selected_mask[rmin:rmax+1, cmin:cmax+1]
        
        # Create masked image (background as black)
        masked_crop = np.zeros_like(cropped_img)
        masked_crop[cropped_mask] = cropped_img[cropped_mask]
        
        # Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(masked_crop).save(output_path)
        
        return True
    
    except Exception as e:
        print(f"Error processing {image_path} with Segformer: {e}")
        return False

# Helper functions for Grounding DINO + SAM
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
    def __init__(self, score, label, box, mask=None):
        self.score = score
        self.label = label
        self.box = box
        self.mask = mask

    @classmethod
    def from_dict(cls, detection_dict):
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                    ymin=detection_dict['box']['ymin'],
                                    xmax=detection_dict['box']['xmax'],
                                    ymax=detection_dict['box']['ymax']))

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

def process_image_grounding_dino_sam(image_path, detector, segmenter, processor, device, output_path, save_original_if_no_clothes=False):
    """Process a single image using Grounding DINO + SAM pipeline."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        # Use Grounding DINO to detect upper clothes and dress
        labels = ["a jacket.", "a dress.", "a shirt.", "a blouse.", "a top.", "a t-shirt.", "a tee.", "a sweater.", "a cardigan.",
                  "a hoodie.", "a sweatshirt.", "a vest.", "a tank top.", "a polo shirt.", "a golf shirt."]
        #labels = ["a backpack.", "a handbag.", "a purse.", "a suitcase.", "a messenger bag.",
        #          "a crossbody bag.", "a tote bag.", "a duffel bag.", "a clutch.", "a fanny pack."]
        results = detector(image, candidate_labels=labels, threshold=0.3)
        results = [DetectionResult.from_dict(result) for result in results]
        
        if not results:
            if save_original_if_no_clothes:
                print(f"No upper clothes or dress found in {image_path}, saving original image")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
            else:
                print(f"No upper clothes or dress found in {image_path}, skipping")
            return True

        # Get detection with highest confidence
        selected_result = max(results, key=lambda r: r.score)
        print(f"Selected {selected_result.label} with confidence {selected_result.score:.4f} for {image_path}")
        
        # Get bounding box from detection
        boxes = get_boxes(results=[selected_result])
        segmenter_inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

        # Generate mask using SAM
        segmenter_outputs = segmenter(**segmenter_inputs)
        masks = processor.post_process_masks(
            masks=segmenter_outputs.pred_masks,
            original_sizes=segmenter_inputs.original_sizes,
            reshaped_input_sizes=segmenter_inputs.reshaped_input_sizes
        )[0]

        # Refine masks
        masks = refine_masks(masks, polygon_refinement=True)
        mask = masks[0] if masks else None
        
        if mask is None or not np.any(mask):
            if save_original_if_no_clothes:
                print(f"No valid mask generated for {image_path}, saving original image")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
            else:
                print(f"No valid mask generated for {image_path}, skipping")
            return True
            
        # Extract bounding box
        bbox = extract_bbox(mask)
        if bbox is None:
            if save_original_if_no_clothes:
                print(f"Invalid bounding box for {image_path}, saving original image")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                image.save(output_path)
            else:
                print(f"Invalid bounding box for {image_path}, skipping")
            return True
        
        rmin, rmax, cmin, cmax = bbox
        
        # Crop the image and mask to the bounding box
        cropped_img = image_np[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        
        # Create masked image (background as black)
        masked_crop = np.zeros_like(cropped_img)
        masked_crop[cropped_mask > 0] = cropped_img[cropped_mask > 0]
        
        # Save the result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(masked_crop).save(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error processing {image_path} with Grounding DINO + SAM: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract upper clothes from CUHK03 dataset')
    parser.add_argument('--cuhk', type=str, default='datasets/cuhk03-np',
                        help='Root directory of cuhk03 dataset')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output directory for extracted upper clothes')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Batch size (currently only supports 1)')
    parser.add_argument('--model', type=str, default='segformer', choices=['segformer', 'grounding_dino_sam'],
                        help='Segmentation model to use: "segformer" or "grounding_dino_sam"')
    parser.add_argument('--save-original-if-no-clothes', action='store_true',
                        help='Save original image if no upper clothes are found')
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models based on the selected mode
    if args.model == 'segformer':
        print("Loading Segformer model...")
        processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        model = model.to(device)
        model.eval()
        
        # Set process_image to segformer function
        process_image = lambda img_path, output_path: process_image_segformer(img_path, processor, model, device, output_path, args.save_original_if_no_clothes)
        
    else:  # grounding_dino_sam
        print("Loading Grounding DINO + SAM models...")
        detector_id = "IDEA-Research/grounding-dino-tiny"
        segmenter_id = "facebook/sam-vit-base"
        
        print("Loading detector...")
        detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
        
        print("Loading segmenter...")
        segmenter = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        processor = AutoProcessor.from_pretrained(segmenter_id)
        
        # Set process_image to grounding_dino_sam function
        process_image = lambda img_path, output_path: process_image_grounding_dino_sam(img_path, detector, segmenter, processor, device, output_path, args.save_original_if_no_clothes)
    
    # Process labeled dataset
    print(f"Processing CUHK03 labeled dataset using {args.model} model...")
    
    # Get dataset directories
    labeled_dirs = [
        os.path.join(args.cuhk, 'labeled', 'foreground_query'),
        os.path.join(args.cuhk, 'labeled', 'foregroundImages_cropped')
    ]
    
    # Process each directory
    for input_dir in labeled_dirs:
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue
        
        # Create the equivalent output directory structure
        relative_path = os.path.relpath(input_dir, args.cuhk)
        output_dir = os.path.join(args.out_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset to get list of images
        dataset = Cuhk03(input_dir, transforms=None)
        
        # Process each image
        processed = 0
        skipped = 0
        
        for idx, img_path in enumerate(tqdm(dataset.imgs, desc=f"Processing {relative_path}")):
            full_img_path = os.path.join(input_dir, img_path)
            output_img_path = os.path.join(output_dir, img_path)
            
            success = process_image(full_img_path, output_img_path)
            
            if success:
                processed += 1
            else:
                skipped += 1
        
        print(f"Processed: {processed}, Skipped: {skipped} images from {input_dir}")
    
    print("Processing complete!")

if __name__ == '__main__':
    main()
