import json
import argparse
import re
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import nms, box_iou
from scipy.optimize import linear_sum_assignment
import cv2
import os
import pycocotools.mask as mask_utils  # For COCO RLE mask format
import traceback


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x, y, width, height]
        box2: [x, y, width, height]
        
    Returns:
        float: IoU value
    """
    # Convert to [x1, y1, x2, y2] format
    box1_x1, box1_y1 = box1[0], box1[1]
    box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    box2_x1, box2_y1 = box2[0], box2[1]
    box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Calculate intersection area
    x_left = max(box1_x1, box2_x1)
    y_top = max(box1_y1, box2_y1)
    x_right = min(box1_x2, box2_x2)
    y_bottom = min(box1_y2, box2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def get_base_filename(filename):
    """
    Extract base filename without the object ID suffix.
    Example: "00015_901.jpg" -> "00015.jpg"
    
    Args:
        filename: Original filename with object ID
        
    Returns:
        str: Base filename without object ID
    """
    # Use regex to match the pattern: base_name_objectID.extension
    match = re.match(r'(.+)_\d+(\.\w+)$', filename)
    if match:
        base_name, extension = match.groups()
        return f"{base_name}{extension}"
    return filename  # Return original if no match


def normalize_ovis_filename(filename, is_ovis=False):
    """
    Normalize filename for OVIS dataset by removing 'img_' prefix if needed.
    
    Args:
        filename: Original filename
        is_ovis: Whether to apply OVIS-specific normalization
        
    Returns:
        str: Normalized filename
    """
    if not is_ovis:
        return filename
    
    # For OVIS dataset, remove 'img_' prefix
    parts = filename.split('/')
    if len(parts) > 1 and parts[-1].startswith('img_'):
        parts[-1] = parts[-1].replace('img_', '', 1)
        return '/'.join(parts)
    return filename


def load_ground_truth(gt_json_path):
    """
    Load ground truth annotations from a JSON file (assuming COCO-like format).
    
    Args:
        gt_json_path: Path to the ground truth JSON file
        
    Returns:
        dict: Mapping from base_file_name to a list of tuples 
              (original_file_name, bounding_box, category_name)
    """
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping from image_id to file_name
    id_to_filename = {image['id']: image['file_name'] for image in data.get('images', [])}
    
    # Create a mapping from category_id to category_name
    id_to_category = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    if not id_to_category:
        print("Warning: No categories found in ground truth file. Labels will be missing.")
    
    # Create a mapping from base_file_name to list of (original_file_name, bbox, category_name)
    base_filename_to_data = {}
    for annotation in data.get('annotations', []):
        image_id = annotation.get('image_id')
        category_id = annotation.get('category_id')
        bbox = annotation.get('bbox') # [x, y, width, height]
        
        if image_id in id_to_filename and bbox and category_id is not None:
            original_file_name = id_to_filename[image_id]
            base_file_name = get_base_filename(original_file_name)
            category_name = id_to_category.get(category_id, "Unknown") # Default label if ID not found
            
            if base_file_name not in base_filename_to_data:
                base_filename_to_data[base_file_name] = []
                
            base_filename_to_data[base_file_name].append((original_file_name, bbox, category_name))
        elif image_id not in id_to_filename:
             print(f"Warning: Skipping annotation with unknown image_id {image_id}")
        elif not bbox:
            print(f"Warning: Skipping annotation with missing bbox (image_id: {image_id})")
        elif category_id is None:
            print(f"Warning: Skipping annotation with missing category_id (image_id: {image_id})")
    
    return base_filename_to_data


def load_detections(detection_json_path, is_ovis=False):
    """
    Load detection results from a JSON file. Handles dict or list input.
    Assumes detections contain 'box' or 'bbox', 'label', and optionally 'confidence'.
    
    Args:
        detection_json_path: Path to the detection JSON file
        is_ovis: Whether to apply OVIS-specific filename normalization
        
    Returns:
        dict: Mapping from file_name to a list of tuples (bounding_box, label, confidence)
              where bounding_box is [x, y, width, height] and confidence defaults to 1.0 if missing.
    """
    with open(detection_json_path, 'r') as f:
        detections_input = json.load(f)
    
    filename_to_data = {}
    processed_detections_dict = {}

    # Handle different input formats (dict or list)
    if isinstance(detections_input, dict):
        # For OVIS, normalize the keys by removing 'img_' prefix
        if is_ovis:
            processed_detections_dict = {}
            for key, value in detections_input.items():
                normalized_key = normalize_ovis_filename(key, is_ovis)
                processed_detections_dict[normalized_key] = value
                if normalized_key != key:
                    print(f"Normalized detection key: {key} -> {normalized_key}")
        else:
            processed_detections_dict = detections_input
    elif isinstance(detections_input, list):
        print(f"Converting detection list format ({len(detections_input)} items)")
        for item in detections_input:
            file_key = None
            detection_data = None
            # Try various common keys
            if "file_name" in item:
                file_key = item["file_name"]
            elif "filename" in item:
                 file_key = item["filename"]
            elif "image_id" in item:
                file_key = str(item["image_id"]) # Use image_id as key if filename missing
            
            # Apply OVIS normalization if needed
            if file_key and is_ovis:
                file_key = normalize_ovis_filename(file_key, is_ovis)
            
            # Store the whole item for later processing
            if file_key:
                detection_data = item 
                if file_key not in processed_detections_dict:
                    processed_detections_dict[file_key] = []
                if detection_data:
                     processed_detections_dict[file_key].append(detection_data)
            else:
                 print(f"Warning: Could not determine file key for detection item: {item}")

    else:
        print(f"Warning: Unknown detection JSON format: {type(detections_input)}. Expected dict or list.")
        return {}

    # Process the standardized dictionary
    for file_path_key, detected_objects in processed_detections_dict.items():
        file_name = file_path_key # Assume key is the filename/path
        
        if file_name not in filename_to_data:
            filename_to_data[file_name] = []
        
        if not isinstance(detected_objects, list):
            print(f"Warning: Expected list of detections for {file_name}, got {type(detected_objects)}. Skipping.")
            continue
            
        for detection in detected_objects:
            try:
                box_values = None
                label = None
                confidence = 1.0 # Default confidence if not found

                if not isinstance(detection, dict):
                     print(f"Warning: Expected detection item to be a dict for {file_name}, got {type(detection)}. Skipping.")
                     continue

                # Extract label
                if "label" in detection:
                    label = detection["label"]
                elif "class" in detection: # Alternative key
                    label = detection["class"]
                elif "category" in detection: # Alternative key
                    label = detection["category"]
                else:
                     print(f"Warning: Missing 'label' in detection item for {file_name}: {detection}. Skipping.")
                     continue # Skip if no label found

                # Extract confidence
                if "confidence" in detection and isinstance(detection["confidence"], (int, float)):
                    confidence = detection["confidence"]
                elif "score" in detection and isinstance(detection["score"], (int, float)): # Alternative key
                    confidence = detection["score"]
                else:
                    # Optional: print a warning if confidence is missing but not required
                    # print(f"Debug: Confidence missing for detection in {file_name}, using default 1.0. Item: {detection}")
                    pass 

                # Extract box (prefer 'box' then 'bbox') - assumes [x1, y1, x2, y2]
                if "box" in detection and len(detection["box"]) == 4:
                    box_values = detection["box"]
                elif "bbox" in detection and len(detection["bbox"]) == 4:
                     box_values = detection["bbox"]
                # Handle case where detection itself is the box list? (Less likely with labels present)
                # elif isinstance(detection, list) and len(detection) == 4: 
                #     box_values = detection

                if box_values and all(isinstance(val, (int, float)) for val in box_values):
                    x1, y1, x2, y2 = box_values
                    # Convert to [x, y, width, height]
                    width = x2 - x1
                    height = y2 - y1
                    # Basic sanity check for width/height
                    if width >= 0 and height >= 0:
                        bbox_xywh = [x1, y1, width, height]
                        # Append confidence to the stored tuple
                        filename_to_data[file_name].append((bbox_xywh, label, confidence))
                    else:
                        print(f"Warning: Invalid box dimensions for {file_name}: w={width}, h={height} from {box_values}. Skipping.")
                else:
                     print(f"Warning: Could not extract valid box_values from detection item for {file_name}: {detection}. Skipping.")

            except Exception as e:
                print(f"Error processing detection item for {file_name}: {e}")
                print(f"  Detection item: {detection}")
    
    return filename_to_data


def is_person_label(label, person_category_name):
    """Check if the label string contains the person category name."""
    # Simple check if the target name is a substring. 
    # Handles cases like "a bear a cat ... a person a squirrel"
    return person_category_name.lower() in label.lower()


def perform_nms(boxes_list, confidence_list, iou_threshold=0.7):
    """
    Perform class-agnostic Non-Maximum Suppression on a set of bounding boxes.
    
    Args:
        boxes_list: List of bounding boxes in [x, y, width, height] format
        confidence_list: List of confidence scores
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of indices of boxes to keep
    """
    if not boxes_list:
        return []
        
    # Convert boxes from [x, y, width, height] to [x1, y1, x2, y2]
    boxes_xyxy = torch.tensor([[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes_list])
    scores = torch.tensor(confidence_list)
    
    # Perform NMS
    keep_indices = nms(boxes_xyxy, scores, iou_threshold)
    
    return keep_indices.tolist()


def calculate_mask_iou(mask1, mask2):
    """
    Calculate IoU between two binary masks.
    
    Args:
        mask1: First binary mask (numpy array)
        mask2: Second binary mask (numpy array)
    
    Returns:
        float: IoU value between 0 and 1
    """
    if mask1.shape != mask2.shape:
        raise ValueError(f"Mask shapes must match: {mask1.shape} vs {mask2.shape}")
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    if union == 0:
        return 0.0
    
    return intersection / union


def decode_rle_mask(rle, shape=None):
    """
    Decode COCO RLE-encoded mask.
    
    Args:
        rle: COCO RLE mask (dict with 'counts' and 'size')
        shape: Optional output shape (height, width)
    
    Returns:
        numpy.ndarray: Binary mask as numpy array
    """
    if not isinstance(rle, dict):
        raise ValueError(f"RLE must be a dictionary, got {type(rle)}")
    
    if 'size' not in rle:
        if shape is None:
            raise ValueError("RLE missing 'size' field and no shape provided")
        rle['size'] = [shape[0], shape[1]]
    
    # Check if RLE is in uncompressed format (counts as list)
    if isinstance(rle.get('counts'), list):
        # Convert uncompressed RLE to compressed format
        # frPyObjects expects [rle], height, width for uncompressed format
        h, w = rle['size'][0], rle['size'][1]
        compressed_rle = mask_utils.frPyObjects([rle], h, w)[0]
        mask = mask_utils.decode(compressed_rle)
    else:
        # Already in compressed format
        mask = mask_utils.decode(rle)
    
    return mask


def decode_polygon_mask(polygon, shape):
    """
    Decode polygon mask into binary mask.
    
    Args:
        polygon: List of [x,y] coordinates forming polygon or flat list of coordinates
        shape: Output shape (height, width)
    
    Returns:
        numpy.ndarray: Binary mask as numpy array
    """
    if not polygon:  # Empty polygon
        return np.zeros(shape, dtype=np.uint8)
    
    mask = np.zeros(shape, dtype=np.uint8)
    
    # Handle different polygon formats
    if isinstance(polygon, list):
        # If it's a list of lists (segmentation format)
        if polygon and isinstance(polygon[0], list):
            # Could be [[x1,y1,x2,y2,...]] or [[x1,y1], [x2,y2], ...]
            if len(polygon) == 1 and len(polygon[0]) > 2:
                # Flat list wrapped in another list
                polygon = polygon[0]
            elif len(polygon[0]) == 2:
                # List of [x,y] points - already in correct format
                poly_array = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(mask, [poly_array], 1)
                return mask
        
        # Convert flat list to array of points
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            points = []
            for i in range(0, len(polygon), 2):
                points.append([int(polygon[i]), int(polygon[i+1])])
            poly_array = np.array(points, dtype=np.int32)
            
            # Draw filled polygon
            cv2.fillPoly(mask, [poly_array], 1)
        else:
            print(f"Warning: Polygon has unexpected format or length: {len(polygon) if isinstance(polygon, list) else 'not a list'}")
    
    return mask


def load_gt_masks_from_json(gt_json_path):
    """
    Load ground truth masks from COCO format JSON.
    
    Args:
        gt_json_path: Path to ground truth JSON (COCO format)
        
    Returns:
        dict: Mapping from base_file_name to list of masks and their metadata
    """
    with open(gt_json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to filename and dimensions
    id_to_image_info = {}
    for image in data.get('images', []):
        id_to_image_info[image['id']] = {
            'file_name': image['file_name'],
            'height': image.get('height', 720),
            'width': image.get('width', 1280)
        }
    
    # Create mapping from category_id to category name
    id_to_category = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    # Create mapping from base_file_name to its annotations with masks
    base_filename_to_masks = {}
    
    for annotation in data.get('annotations', []):
        image_id = annotation.get('image_id')
        category_id = annotation.get('category_id')
        segmentation = annotation.get('segmentation')
        
        if image_id is None or category_id is None or not segmentation:
            continue
        
        if image_id not in id_to_image_info:
            continue
        
        image_info = id_to_image_info[image_id]
        original_filename = image_info['file_name']
        base_filename = get_base_filename(original_filename)
        category_name = id_to_category.get(category_id, "Unknown")
        
        # Ensure RLE has size field if it doesn't
        if isinstance(segmentation, dict) and 'size' not in segmentation:
            segmentation['size'] = [image_info['height'], image_info['width']]
        
        if base_filename not in base_filename_to_masks:
            base_filename_to_masks[base_filename] = []
        
        # Extract bbox for reference
        bbox = annotation.get('bbox', [0, 0, 0, 0])
        
        # Store mask data
        base_filename_to_masks[base_filename].append({
            'mask': segmentation,
            'category_name': category_name,
            'bbox': bbox,
            'original_file_name': original_filename
        })
    
    return base_filename_to_masks


def load_detection_masks(mask_json_path, is_ovis=False):
    """
    Load detection masks from the output of segment_detections.py.
    
    Args:
        mask_json_path: Path to segmentation JSON file
        is_ovis: Whether to apply OVIS-specific filename normalization
        
    Returns:
        dict: Mapping from filename to list of detection masks and metadata
    """
    with open(mask_json_path, 'r') as f:
        data = json.load(f)
    
    detection_masks = {}
    
    for filename, detections in data.items():
        if not detections:
            continue
        
        # Apply OVIS normalization if needed
        normalized_filename = normalize_ovis_filename(filename, is_ovis)
        
        detection_masks[normalized_filename] = []
        
        for detection in detections:
            detection_masks[normalized_filename].append({
                'mask_polygon': detection.get('mask_polygon', []),
                'label': detection.get('label', ''),
                'confidence': detection.get('confidence', 0.0),
                'box': detection.get('box', [0, 0, 0, 0])
            })
    
    return detection_masks


def load_gt_masks_from_dir(mask_dir, detections):
    """
    Load ground truth masks from a directory of PNG files.
    
    Args:
        mask_dir: Directory containing mask PNG files
        detections: Detection dictionary with filenames as keys
        
    Returns:
        dict: Mapping from filename to mask array
    """
    gt_masks = {}
    
    for filename in detections.keys():
        # Get base filename without extension
        base_name = os.path.splitext(filename)[0]
        
        # Try different extensions
        for ext in ['.png', '.jpg', '.jpeg']:
            mask_path = os.path.join(mask_dir, base_name + ext)
            if os.path.exists(mask_path):
                # Load mask as binary image
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # Convert to binary
                    mask = (mask > 0).astype(np.uint8)
                    gt_masks[filename] = mask
                    break
    
    return gt_masks


def debug_filename_matching(ground_truth, detections, mask_detections):
    """Debug function to understand filename matching issues."""
    print("\n=== FILENAME MATCHING DEBUG ===")
    
    # Get sample keys from each dataset
    gt_keys = list(ground_truth.keys())[:5]
    det_keys = list(detections.keys())[:5]
    mask_keys = list(mask_detections.keys())[:5]
    
    print(f"\nSample GT keys: {gt_keys}")
    print(f"Sample Detection keys: {det_keys}")
    print(f"Sample Mask Detection keys: {mask_keys}")
    
    # Check for exact matches
    gt_set = set(ground_truth.keys())
    det_set = set(detections.keys())
    mask_set = set(mask_detections.keys())
    
    exact_matches = gt_set.intersection(det_set)
    print(f"\nExact matches between GT and detections: {len(exact_matches)}")
    if exact_matches:
        print(f"  First 5 matches: {list(exact_matches)[:5]}")
    
    # Check if detection keys match mask keys
    det_mask_matches = det_set.intersection(mask_set)
    print(f"\nMatches between detections and mask detections: {len(det_mask_matches)}")
    
    # Check for base filename patterns
    if not exact_matches and gt_keys and det_keys:
        print("\nChecking if filenames have different patterns...")
        print(f"GT pattern: {gt_keys[0]}")
        print(f"Detection pattern: {det_keys[0]}")
        
        # Check if GT has object IDs that need to be stripped
        if '_' in gt_keys[0] and re.match(r'.+_\d+\.\w+$', gt_keys[0]):
            print("  GT filenames appear to have object IDs (_XXX)")
        
        # Try to find matches using original filenames from GT
        original_matches = 0
        for base_name, gt_list in ground_truth.items():
            for gt_data in gt_list:
                orig_name = gt_data.get('original_file_name', '')
                if orig_name in det_set:
                    original_matches += 1
                    if original_matches <= 3:
                        print(f"  Found match using original name: {orig_name}")
                    break
        print(f"\nMatches using original GT filenames: {original_matches}")
    
    print("=== END DEBUG ===\n")


def debug_mask_matching(gt_masks, det_masks, gt_metadata, det_metadata):
    """Debug function to understand mask matching issues."""
    print("\n=== MASK MATCHING DEBUG ===")
    
    if not gt_masks or not det_masks:
        print("No masks to debug")
        return
    
    # Check first GT and detection mask
    gt_mask = gt_masks[0]
    det_mask = det_masks[0]
    
    print(f"GT mask shape: {gt_mask.shape}")
    print(f"Detection mask shape: {det_mask.shape}")
    print(f"GT mask non-zero pixels: {np.sum(gt_mask)}")
    print(f"Detection mask non-zero pixels: {np.sum(det_mask)}")
    
    # Calculate sample IoU
    if gt_mask.shape == det_mask.shape:
        iou = calculate_mask_iou(gt_mask, det_mask)
        print(f"Sample IoU between first GT and detection: {iou:.4f}")
    else:
        print("ERROR: Mask shapes don't match!")
    
    # Check if masks are valid
    if np.sum(gt_mask) == 0:
        print("WARNING: GT mask is empty!")
    if np.sum(det_mask) == 0:
        print("WARNING: Detection mask is empty!")
    
    print("=== END MASK DEBUG ===\n")


def categorize_instances(ground_truth, detections, mask_detections, iou_threshold, person_category_name, 
                         confidence_threshold=0.0, debug=False, nms_threshold=0.7, mask_shape=(720, 1280)):
    """
    Categorize ground truth instances based on detection results using mask IoU.
    Uses Hungarian algorithm for optimal assignment of detections to ground truth.
    
    Args:
        ground_truth: Dict mapping base_file_name to list of (original_file_name, mask, category_name)
        detections: Dict mapping file_name to list of (bbox, label, confidence) 
        mask_detections: Dict mapping file_name to list of detection masks and metadata
        iou_threshold: IoU threshold for considering a match
        person_category_name: The string representing the 'person' category
        confidence_threshold: Minimum confidence score for a detection to be considered
        debug: Whether to print debug information
        nms_threshold: IoU threshold for NMS (default: 0.7)
        mask_shape: Default shape (height, width) for masks
    
    Returns:
        dict: Dictionary containing sets of original_file_names for each category.
    """
    results = {
        "Detected Person Exclusion": set(),
        "Undetected Instance Exclusion": set(),
        "Misclassified Instance Inclusion": set(),
        "Misclassified Instance Exclusion": set()
    }
    
    # Debug filename matching once
    if debug:
        debug_filename_matching(ground_truth, detections, mask_detections)
    
    processed_gt_instances = set()  # Keep track of GT instances already categorized
    total_files_processed = 0
    files_with_matches = 0
    
    # --- Iterate through Ground Truth ---
    for base_file_name, gt_data_list in ground_truth.items():
        total_files_processed += 1
        
        # Find potential detection masks for this base filename or its variants
        candidate_key = None
        detected_boxes_for_file = []
        detected_masks_for_file = []
        
        # Prioritize base filename match
        if base_file_name in detections and base_file_name in mask_detections:
            candidate_key = base_file_name
            detected_boxes_for_file = detections[base_file_name]
            detected_masks_for_file = mask_detections[base_file_name]
        else:
            # Check original filenames if base didn't match
            for gt_data in gt_data_list:
                original_file_name = gt_data.get('original_file_name')
                if original_file_name in detections and original_file_name in mask_detections:
                    candidate_key = original_file_name
                    detected_boxes_for_file = detections[original_file_name]
                    detected_masks_for_file = mask_detections[original_file_name]
                    if debug and files_with_matches < 3:
                        print(f"Debug: Found detections using original filename key: {original_file_name}")
                    files_with_matches += 1
                    break  # Use the first match found
        
        # Skip this file if no detections found
        if not detected_boxes_for_file or not detected_masks_for_file:
            # All GT boxes for this file go to Undetected Instance Exclusion if they are persons
            for gt_data in gt_data_list:
                gt_category_name = gt_data.get('category_name')
                gt_original_file_name = gt_data.get('original_file_name')
                is_gt_person = (gt_category_name.lower() == person_category_name.lower())
                if is_gt_person:
                    results["Undetected Instance Exclusion"].add(gt_original_file_name)
            continue
        
        # Ensure we have the same number of masks as detections
        if len(detected_boxes_for_file) != len(detected_masks_for_file):
            print(f"Warning: Mismatch between detection boxes ({len(detected_boxes_for_file)}) and masks ({len(detected_masks_for_file)}) for {candidate_key}")
            # Use the minimum number to be safe
            min_count = min(len(detected_boxes_for_file), len(detected_masks_for_file))
            detected_boxes_for_file = detected_boxes_for_file[:min_count]
            detected_masks_for_file = detected_masks_for_file[:min_count]

        # Apply NMS first (using bounding boxes)
        boxes = [det[0] for det in detected_boxes_for_file]  # Extract bboxes
        scores = [det[2] for det in detected_boxes_for_file]  # Extract confidence scores
        keep_indices = perform_nms(boxes, scores, nms_threshold)
        
        # Keep only the detections that survived NMS
        nms_detections_boxes = [detected_boxes_for_file[i] for i in keep_indices]
        nms_detections_masks = [detected_masks_for_file[i] for i in keep_indices]
        
        if debug and len(nms_detections_boxes) < len(detected_boxes_for_file):
            print(f"Debug: NMS reduced detections from {len(detected_boxes_for_file)} to {len(nms_detections_boxes)} for key {candidate_key or base_file_name}")
        
        # Then filter by confidence threshold
        filtered_detections_boxes = []
        filtered_detections_masks = []
        for i, (box_data, mask_data) in enumerate(zip(nms_detections_boxes, nms_detections_masks)):
            if box_data[2] >= confidence_threshold:  # Check confidence
                filtered_detections_boxes.append(box_data)
                filtered_detections_masks.append(mask_data)
        
        if debug and len(filtered_detections_boxes) < len(nms_detections_boxes):
            print(f"Debug: Filtered {len(nms_detections_boxes) - len(filtered_detections_boxes)} detections below threshold {confidence_threshold} for key {candidate_key or base_file_name}")
        
        # Skip if no detections survived filtering
        if not filtered_detections_boxes:
            # All GT boxes for this file go to Undetected Instance Exclusion if they are persons
            for gt_data in gt_data_list:
                gt_category_name = gt_data.get('category_name')
                gt_original_file_name = gt_data.get('original_file_name')
                is_gt_person = (gt_category_name.lower() == person_category_name.lower())
                if is_gt_person:
                    results["Undetected Instance Exclusion"].add(gt_original_file_name)
            continue

        # --- Process each Ground Truth Instance using Hungarian algorithm with mask IoU ---
        # Prepare GT and detection information
        gt_masks = []
        gt_metadata = []  # Store metadata for each GT instance
        
        det_masks = []
        det_metadata = []  # Store metadata for each detection
        
        # Extract GT information and decode masks
        for gt_data in gt_data_list:
            gt_original_file_name = gt_data.get('original_file_name')
            gt_mask_data = gt_data.get('mask')
            gt_bbox = gt_data.get('bbox')
            gt_category_name = gt_data.get('category_name')
            
            # Skip if already processed
            gt_instance_id = (gt_original_file_name, tuple(gt_bbox))
            if gt_instance_id in processed_gt_instances:
                continue
                
            # Decode RLE mask
            try:
                # Ensure the mask data has required fields
                if isinstance(gt_mask_data, dict) and 'size' not in gt_mask_data:
                    print(f"Warning: GT mask missing 'size' field for {gt_original_file_name}")
                    continue
                    
                gt_mask_array = decode_rle_mask(gt_mask_data)
                gt_masks.append(gt_mask_array)
                gt_metadata.append((gt_original_file_name, gt_category_name))
                if debug and len(gt_masks) == 1 and total_files_processed <= 3:  # Debug first mask
                    print(f"Debug: GT mask shape: {gt_mask_array.shape}, non-zero pixels: {np.sum(gt_mask_array)}")
                    print(f"Debug: GT mask data type: {type(gt_mask_data)}")
                    if isinstance(gt_mask_data, dict):
                        print(f"Debug: GT RLE keys: {gt_mask_data.keys()}")
                        if 'size' in gt_mask_data:
                            print(f"Debug: GT RLE size: {gt_mask_data['size']}")
                        if 'counts' in gt_mask_data:
                            counts_type = type(gt_mask_data['counts'])
                            print(f"Debug: GT RLE counts type: {counts_type}")
                            if isinstance(gt_mask_data['counts'], list):
                                print(f"Debug: GT RLE counts length: {len(gt_mask_data['counts'])}")
            except Exception as e:
                print(f"Error decoding GT mask for {gt_original_file_name}: {e}")
                if debug:
                    print(f"Debug: GT mask data type: {type(gt_mask_data)}, content: {str(gt_mask_data)[:200]}...")
                    traceback.print_exc()
        
        # If all GT instances were already processed, skip
        if not gt_masks:
            continue
        
        # Extract detection information and decode masks
        for i, (det_box, det_mask) in enumerate(zip(filtered_detections_boxes, filtered_detections_masks)):
            det_label = det_box[1]
            det_confidence = det_box[2]
            mask_polygon = det_mask.get('mask_polygon', [])
            
            # Get shape from GT masks if available
            shape = gt_masks[0].shape if gt_masks else mask_shape
            
            # Decode polygon mask
            try:
                det_mask_array = decode_polygon_mask(mask_polygon, shape)
                det_masks.append(det_mask_array)
                det_metadata.append((det_label, det_confidence))
                if debug and len(det_masks) == 1 and total_files_processed <= 3:  # Debug first mask
                    print(f"Debug: Detection mask shape: {det_mask_array.shape}, non-zero pixels: {np.sum(det_mask_array)}")
                    print(f"Debug: Polygon type: {type(mask_polygon)}, length: {len(mask_polygon) if isinstance(mask_polygon, list) else 'N/A'}")
                    if isinstance(mask_polygon, list) and len(mask_polygon) > 0:
                        print(f"Debug: First polygon element type: {type(mask_polygon[0])}")
                        if isinstance(mask_polygon[0], list):
                            print(f"Debug: First polygon element length: {len(mask_polygon[0])}")
                            if len(mask_polygon[0]) > 0:
                                print(f"Debug: Sample polygon coords: {mask_polygon[0][:6]}...")
                        else:
                            print(f"Debug: Sample polygon coords: {mask_polygon[:6]}...")
            except Exception as e:
                print(f"Error decoding detection mask: {e}")
                if debug:
                    print(f"Debug: Polygon data: {str(mask_polygon)[:200]}...")  # Show first 200 chars
        
        # Skip if no valid detection masks
        if not det_masks:
            # All GT instances go to Undetected Instance Exclusion if they are persons
            for gt_idx, (gt_original_file_name, gt_category_name) in enumerate(gt_metadata):
                is_gt_person = (gt_category_name.lower() == person_category_name.lower())
                if is_gt_person:
                    results["Undetected Instance Exclusion"].add(gt_original_file_name)
            continue
        
        # Debug mask matching for first file
        if debug and total_files_processed <= 3:
            debug_mask_matching(gt_masks, det_masks, gt_metadata, det_metadata)
        
        # Calculate IoU matrix using masks
        iou_matrix = np.zeros((len(gt_masks), len(det_masks)))
        for i, gt_mask in enumerate(gt_masks):
            for j, det_mask in enumerate(det_masks):
                try:
                    iou_matrix[i, j] = calculate_mask_iou(gt_mask, det_mask)
                except Exception as e:
                    print(f"Error calculating mask IoU: {e}")
                    if debug:
                        print(f"Debug: GT mask shape: {gt_mask.shape}, Det mask shape: {det_mask.shape}")
                    iou_matrix[i, j] = 0.0
        
        if debug and len(gt_masks) > 0 and len(det_masks) > 0 and total_files_processed <= 3:
            max_iou = np.max(iou_matrix)
            print(f"Debug: Max IoU in matrix: {max_iou:.4f}")
            if max_iou > 0:
                max_i, max_j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                print(f"Debug: Best match IoU: {iou_matrix[max_i, max_j]:.4f} between GT {max_i} and Det {max_j}")
                print(f"Debug: GT category: {gt_metadata[max_i][1]}, Det label: {det_metadata[max_j][0]}")
        
        # Use negative IoU as cost (Hungarian algorithm minimizes cost)
        cost_matrix = 1.0 - iou_matrix
        
        # Apply Hungarian algorithm to find optimal assignment
        gt_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # Create a mapping from GT index to assigned detection index
        gt_to_det = {}
        for gt_idx, det_idx in zip(gt_indices, det_indices):
            if iou_matrix[gt_idx, det_idx] >= iou_threshold:  # Only consider matches above threshold
                gt_to_det[gt_idx] = det_idx
        
        if debug and total_files_processed <= 3:
            print(f"Debug: Hungarian algorithm assigned {len(gt_to_det)} GT masks out of {len(gt_masks)} to detections")
        
        # Process each GT instance based on assignment results
        for gt_idx, (gt_original_file_name, gt_category_name) in enumerate(gt_metadata):
            is_gt_person = (gt_category_name.lower() == person_category_name.lower())
            
            # Add to processed instances (use the bbox as identifier)
            gt_data = gt_data_list[gt_idx]
            gt_bbox = gt_data.get('bbox')
            gt_instance_id = (gt_original_file_name, tuple(gt_bbox))
            processed_gt_instances.add(gt_instance_id)
            
            if gt_idx in gt_to_det:  # GT mask was matched to a detection
                det_idx = gt_to_det[gt_idx]
                det_label, det_confidence = det_metadata[det_idx]
                is_det_person = is_person_label(det_label, person_category_name)
                
                # --- Apply Categorization Logic (same as before) ---
                if debug and total_files_processed <= 3:
                    iou_value = iou_matrix[gt_idx, det_idx]
                    print(f"Debug: GT={gt_original_file_name} ({gt_category_name}), Match=(IOU:{iou_value:.3f}, Label:'{det_label}', Conf:{det_confidence:.3f}), IsDetPerson={is_det_person}")
                
                if is_gt_person and is_det_person:
                    # Case 1: Detected Person Exclusion
                    results["Detected Person Exclusion"].add(gt_original_file_name)
                elif is_gt_person and not is_det_person:
                    # Case 3: Misclassified Instance Inclusion (GT is person, detected as non-person)
                    results["Misclassified Instance Inclusion"].add(gt_original_file_name)
                elif not is_gt_person and is_det_person:
                    # Case 4: Misclassified Instance Exclusion (GT is non-person, detected as person)
                    results["Misclassified Instance Exclusion"].add(gt_original_file_name)
                # else: GT is non-person, detected as non-person (Correct Negative - not tracked)
                
            else:  # GT mask was not matched to any detection
                if debug and total_files_processed <= 3:
                    # Find best match for debugging, even though it's below threshold
                    if iou_matrix.shape[1] > 0:  # If there are any detections
                        best_det_idx = np.argmax(iou_matrix[gt_idx])
                        best_iou = iou_matrix[gt_idx, best_det_idx]
                        best_label = det_metadata[best_det_idx][0]
                        print(f"Debug: GT={gt_original_file_name} ({gt_category_name}), Best IoU={best_iou:.3f} (Below Threshold), Label:'{best_label}'")
                
                if is_gt_person:
                    # Case 2: Undetected Instance Exclusion (GT is person, not detected)
                    results["Undetected Instance Exclusion"].add(gt_original_file_name)
                # else: GT is non-person, not detected (Correct Negative - not tracked)

    print(f"\nProcessed {total_files_processed} GT files, found matches in {files_with_matches} files")
    print("-" * 20)
    for category, file_set in results.items():
        print(f"{category}: {len(file_set)} files")
    print("-" * 20)

    return results


def categorize_from_detections_only(detections, mask_detections, person_category_name, 
                                    mask_dir=None, confidence_threshold=0.0, debug=False, 
                                    nms_threshold=0.7, default_img_size=(1024, 1024)):
    """
    Categorize detections without ground truth annotations using mask IoU.
    Loads whole-image binary masks as ground truth.
    All filenames are treated as ground truth persons.
    
    Args:
        detections: Dict mapping file_name to list of (bbox, label, confidence)
        mask_detections: Dict mapping file_name to list of detection masks and metadata
        person_category_name: String representing 'person' category
        mask_dir: Directory containing ground truth binary masks (PNG files)
        confidence_threshold: Minimum confidence for detections
        debug: Whether to print debug information
        nms_threshold: IoU threshold for NMS (default: 0.7)
        default_img_size: Default image size to use if image cannot be loaded (width, height)
        
    Returns:
        dict: Dictionary containing sets of filenames for each category
    """
    results = {
        "Detected Person Exclusion": set(),
        "Undetected Instance Exclusion": set(),
        "Misclassified Instance Inclusion": set(),
        "Misclassified Instance Exclusion": set() 
    }
    
    print(f"Running in detection-only mode using mask IoU. Loading GT masks from: {mask_dir}")
    
    # Load ground truth masks if mask_dir is provided
    gt_masks_dict = {}
    if mask_dir:
        gt_masks_dict = load_gt_masks_from_dir(mask_dir, detections)
        print(f"Loaded {len(gt_masks_dict)} ground truth masks from {mask_dir}")
    
    # Process each file in detections
    for filename in detections.keys():
        # If this file doesn't have mask detections, categorize as Undetected Instance Exclusion
        if filename not in mask_detections:
            results["Undetected Instance Exclusion"].add(filename)
            if debug:
                print(f"Debug: File {filename} has no mask detections. Categorized as Undetected Instance Exclusion.")
            continue
                
        # Get the ground truth mask for this file
        gt_mask = None
        if filename in gt_masks_dict:
            gt_mask = gt_masks_dict[filename]
        else:
            # Create a full-image mask as fallback
            img_width, img_height = default_img_size
            gt_mask = np.ones((img_height, img_width), dtype=np.uint8)
            if debug:
                print(f"Debug: Using default full-image mask for {filename}")
        
        detection_list = detections[filename]
        mask_detection_list = mask_detections[filename]
        
        # Ensure we have the same number of masks as detections
        if len(detection_list) != len(mask_detection_list):
            print(f"Warning: Mismatch between detection boxes ({len(detection_list)}) and masks ({len(mask_detection_list)}) for {filename}")
            # Use the minimum number to be safe
            min_count = min(len(detection_list), len(mask_detection_list))
            detection_list = detection_list[:min_count]
            mask_detection_list = mask_detection_list[:min_count]
        
        # Skip files with no detections
        if not detection_list or not mask_detection_list:
            # If no detections, add to Undetected Instance Exclusion since we treat all images as persons
            results["Undetected Instance Exclusion"].add(filename)
            if debug:
                print(f"Debug: File {filename} - No detections found.")
            continue
        
        # Apply NMS first
        boxes = [det[0] for det in detection_list]  # Extract bboxes
        scores = [det[2] for det in detection_list]  # Extract confidence scores
        keep_indices = perform_nms(boxes, scores, nms_threshold)
        
        # Keep only the detections that survived NMS
        nms_detections = [detection_list[i] for i in keep_indices]
        nms_mask_detections = [mask_detection_list[i] for i in keep_indices]
        
        if debug and len(nms_detections) < len(detection_list):
            print(f"Debug: NMS reduced detections from {len(detection_list)} to {len(nms_detections)} for file {filename}")
        
        # Then filter by confidence threshold
        filtered_detections = []
        filtered_mask_detections = []
        for i, (det, mask_det) in enumerate(zip(nms_detections, nms_mask_detections)):
            if det[2] >= confidence_threshold:
                filtered_detections.append(det)
                filtered_mask_detections.append(mask_det)
        
        if not filtered_detections:
            # If all detections filtered out, treat as undetected person
            results["Undetected Instance Exclusion"].add(filename)
            if debug:
                print(f"Debug: File {filename} - All detections below confidence threshold.")
            continue
        
        # Decode detection masks
        det_masks = []
        det_metadata = []
        for i, mask_det in enumerate(filtered_mask_detections):
            mask_polygon = mask_det.get('mask_polygon', [])
            det_label = filtered_detections[i][1]
            det_confidence = filtered_detections[i][2]
            
            # Decode polygon to binary mask
            try:
                det_mask = decode_polygon_mask(mask_polygon, gt_mask.shape)
                det_masks.append(det_mask)
                det_metadata.append((det_label, det_confidence))
            except Exception as e:
                print(f"Error decoding detection mask for {filename}: {e}")
        
        if not det_masks:
            # If no valid detection masks, treat as undetected person
            results["Undetected Instance Exclusion"].add(filename)
            continue
        
        # Calculate IoU between GT mask and all detection masks
        iou_values = []
        for det_mask in det_masks:
            iou = calculate_mask_iou(gt_mask, det_mask)
            iou_values.append(iou)
        
        # Find detection with highest IoU
        best_det_idx = np.argmax(iou_values)
        best_iou = iou_values[best_det_idx]
        
        # Get label and confidence for the best detection
        best_label, best_confidence = det_metadata[best_det_idx]
        
        # Check if detection is classified as person
        is_det_person = is_person_label(best_label, person_category_name)
        
        if debug:
            print(f"Debug: File={filename}, BestMatch=(IoU:{best_iou:.3f}, Label:'{best_label}', Conf:{best_confidence:.3f}), IsDetPerson={is_det_person}")
        
        # Apply categorization logic - all files are assumed to be person ground truth
        if is_det_person:
            # Case 1: Detected Person Exclusion - correctly detected as person
            results["Detected Person Exclusion"].add(filename)
        else:
            # Case 3: Misclassified Instance Inclusion - person detected as non-person
            results["Misclassified Instance Inclusion"].add(filename)
    
    print("-" * 20)
    for category, file_set in results.items():
        print(f"{category}: {len(file_set)} files")
    print("-" * 20)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare detected bounding boxes with ground truth and categorize based on 'person' detection.")
    parser.add_argument("--gt_json", type=str, required=False, help="Path to ground truth JSON file (COCO format). If not provided, will enter detection-only mode.")
    parser.add_argument("--detection_json", type=str, required=True, help="Path to detection results JSON file")
    parser.add_argument("--mask_json", type=str, required=True, help="Path to mask results JSON file from segment_detections.py")
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory containing ground truth binary mask PNG files (for detection-only mode)")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file to save categorized file names")
    parser.add_argument("--person_category_name", type=str, default="person", help="Name of the 'person' category in ground truth and detections")
    parser.add_argument("--confidence_threshold", type=float, default=0.0, help="Minimum confidence score for detections to be considered (default: 0.0, no filtering)")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--nms-threshold", type=float, default=0.7, help="IoU threshold for non-maximum suppression (default: 0.7)")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory containing image files (used in detection-only mode to get image dimensions)")
    parser.add_argument("--ovis", action="store_true", help="Apply OVIS dataset filename normalization (remove 'img_' prefix from detection filenames)")
    
    args = parser.parse_args()
    
    # Load detection results (always needed)
    print("Loading detections...")
    detections = load_detections(args.detection_json, is_ovis=args.ovis)
    print(f"Loaded {len(detections)} filenames with detections.")
    
    # Load mask results from segment_detections.py
    print("Loading detection masks...")
    mask_detections = load_detection_masks(args.mask_json, is_ovis=args.ovis)
    print(f"Loaded {len(mask_detections)} filenames with mask detections.")
    
    if not detections:
        print("Error: Detections data is empty. Exiting.")
        return
        
    # Different processing paths based on whether ground truth is provided
    if args.gt_json:
        # Traditional approach: compare detections with ground truth using mask IoU
        print("Loading ground truth with masks...")
        gt_masks = load_gt_masks_from_json(args.gt_json)
        print(f"Loaded {len(gt_masks)} filenames with ground truth masks.")
        
        if not gt_masks:
            print("Error: Ground truth mask data is empty. Exiting.")
            return
            
        print(f"Categorizing instances using mask IoU threshold {args.iou_threshold}, Person category '{args.person_category_name}', Confidence threshold {args.confidence_threshold}...")
        categorized_results = categorize_instances(
            gt_masks, 
            detections, 
            mask_detections,
            args.iou_threshold, 
            args.person_category_name, 
            args.confidence_threshold,
            args.debug,
            args.nms_threshold
        )
    else:
        # Detection-only approach with masks
        print("No ground truth provided. Using detection-only approach with mask IoU.")
        if not args.mask_dir:
            print("Warning: No mask_dir provided. Will use full-image masks as ground truth.")
        categorized_results = categorize_from_detections_only(
            detections,
            mask_detections,
            args.person_category_name,
            args.mask_dir,
            args.confidence_threshold,
            args.debug,
            args.nms_threshold
        )
    
    # Convert sets to sorted lists for JSON serialization
    output_data = {k: sorted(list(v)) for k, v in categorized_results.items()}
    
    # Save categorized file names to output JSON file
    print(f"Saving categorized filenames to {args.output}...")
    try:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully saved results to {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}")


if __name__ == "__main__":
    main()
