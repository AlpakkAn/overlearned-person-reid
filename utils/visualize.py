import argparse
import json
import os
from pathlib import Path

import fiftyone as fo
import fiftyone.brain as fob
import numpy as np
import torch

from collections import defaultdict

# Define a palette of visually distinct colors to ensure clear separation when
# coloring by categorical values with only a few categories
# Colors chosen from ColorBrewer "Set1" which is color-blind friendly
# and provides strong visual contrast
DISTINCT_COLORS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#FF7F00",  # orange
    "#984EA3",  # purple
    "#FFFF33",  # yellow
]

def get_args_parser():
    parser = argparse.ArgumentParser(description='Data visualization', add_help=False)
    parser.add_argument('--name', default='yvis', type=str, help='dataset name')
    parser.add_argument('--dataset-dir',
                        default='',
                        type=str, help='dataset dir')
    parser.add_argument('--embed-dir',
                        default='embeddings',
                        type=str, help='directory containing embedding files')
    parser.add_argument('--embed-img',
                        default='standard_query.pt',
                        type=str, help='first image embeddings file')
    parser.add_argument('--embed-img2',
                        default='',
                        type=str, help='second image embeddings file (optional)')
    parser.add_argument('--embed-seq',
                        default='',
                        type=str, help='sequence embeddings file')
    parser.add_argument('--port', default=5151, type=int, help='port')
    parser.add_argument('--category', default=None, type=str, 
                        help='filter by categories (comma-separated list of category names)')
    parser.add_argument('--max-objects', default=None, type=int, help='maximum number of unique objects to include')
    parser.add_argument('--max-objects2', default=None, type=int, help='maximum number of unique objects for the second category')
    parser.add_argument('--max-objects3', default=None, type=int, help='maximum number of unique objects for the third category')
    parser.add_argument('--seed', default=69, type=int, help='random seed')
    parser.add_argument('--center-weights-path', default=None, type=str, help='Path to model state_dict containing center embeddings')
    parser.add_argument('--center-label-map-path', default=None, type=str, help='Path to JSON mapping center indices to labels')
    return parser


def extract_label(filename):
    base = os.path.basename(filename)
    label = base.split('_')[1].split('.')[0]
    return label


def create_filename_to_category_mapping(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create a dictionary to map image_id to file_name
    image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    # Create a defaultdict to map file names to category IDs
    filename_to_category_mapping = defaultdict(list)

    # Populate the mapping
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        if image_id in image_id_to_filename:
            file_name = image_id_to_filename[image_id]
            filename_to_category_mapping[file_name].append(category_id)

    category_id_to_name = {}

    # Create a mapping from category_id to category_name
    for category in data['categories']:
        category_id_to_name[category['id']] = category['name']

    return dict(filename_to_category_mapping), category_id_to_name


parser = argparse.ArgumentParser('Data visualization', parents=[get_args_parser()])
args = parser.parse_args()

name = args.name
dataset_dir = args.dataset_dir

try:
    dataset = fo.load_dataset(name)
except:
    json_file_path = os.path.join(dataset_dir.rsplit('/', 1)[0], 'fg_instances_cropped.json')
    mapping, category_id_to_name = create_filename_to_category_mapping(json_file_path)

    filepaths_labels_cats = []
    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".jpg"):  # or any other image extension
                label = extract_label(file)
                filepath = os.path.join(subdir, file)
                file_name = os.path.join(subdir.rsplit('/', 1)[1], file)
                category_id = mapping[file_name][0]
                category_name = category_id_to_name[category_id]
                filepaths_labels_cats.append((filepath, label, category_name))

    # Sort by file paths
    filepaths_labels_cats.sort(key=lambda x: x[0])

    # Create the dataset
    dataset = fo.Dataset(name)

    # Add sorted samples to the dataset
    for filepath, label, category_name in filepaths_labels_cats:
        sample = fo.Sample(filepath=filepath,
                           ground_truth=fo.Classification(label=label),
                           category=fo.Classification(label=category_name))
        dataset.add_sample(sample)

# Save original dataset before filtering
original_dataset = dataset

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

# Store the original dataset view before any filtering
original_dataset_view = dataset.view()
final_filtered_dataset = None # Use this to store the result

# Handle category and max-objects filtering
categories = []
if args.category is not None:
    categories = [cat.strip() for cat in args.category.split(',')]

# --- Special Case: Two Categories, Two Max Objects ---
if len(categories) == 2 and args.max_objects is not None and args.max_objects2 is not None:
    cat1, cat2 = categories
    max1, max2 = args.max_objects, args.max_objects2
    print(f"Applying specialized filtering: Category '{cat1}' (max {max1} objects) and Category '{cat2}' (max {max2} objects)...")

    # Filter for category 1 and apply max_objects
    view1 = original_dataset_view.match({"category.label": cat1})
    labels1 = view1.distinct("ground_truth.label")
    if len(labels1) > max1:
        np.random.seed(args.seed) # Seed for reproducibility
        selected_labels1 = np.random.choice(labels1, max1, replace=False).tolist()
        view1 = view1.match({"ground_truth.label": {"$in": selected_labels1}})
        print(f"Filtered '{cat1}' to {len(view1)} samples ({max1} unique objects)")
    else:
        print(f"Category '{cat1}' has {len(view1)} samples ({len(labels1)} unique objects), no object limit applied.")

    # Filter for category 2 and apply max_objects2
    view2 = original_dataset_view.match({"category.label": cat2})
    labels2 = view2.distinct("ground_truth.label")
    if len(labels2) > max2:
        np.random.seed(args.seed) # Use same seed for reproducibility
        selected_labels2 = np.random.choice(labels2, max2, replace=False).tolist()
        view2 = view2.match({"ground_truth.label": {"$in": selected_labels2}})
        print(f"Filtered '{cat2}' to {len(view2)} samples ({max2} unique objects)")
    else:
         print(f"Category '{cat2}' has {len(view2)} samples ({len(labels2)} unique objects), no object limit applied.")

    # Combine the IDs from both views
    filtered_ids = view1.values("id") + view2.values("id")
    final_filtered_dataset = original_dataset_view.select(filtered_ids) # Store the final filtered dataset view
    print(f"Combined filtered dataset has {len(final_filtered_dataset)} samples.")

# --- Special Case: Three Categories, Three Max Objects ---
elif len(categories) == 3 and args.max_objects is not None and args.max_objects2 is not None and args.max_objects3 is not None:
    cat1, cat2, cat3 = categories
    max1, max2, max3 = args.max_objects, args.max_objects2, args.max_objects3
    print(f"Applying specialized filtering: Category '{cat1}' (max {max1} objects), Category '{cat2}' (max {max2} objects), and Category '{cat3}' (max {max3} objects)...")

    # Filter for category 1 and apply max_objects
    view1 = original_dataset_view.match({"category.label": cat1})
    labels1 = view1.distinct("ground_truth.label")
    if len(labels1) > max1:
        np.random.seed(args.seed) # Seed for reproducibility
        selected_labels1 = np.random.choice(labels1, max1, replace=False).tolist()
        view1 = view1.match({"ground_truth.label": {"$in": selected_labels1}})
        print(f"Filtered '{cat1}' to {len(view1)} samples ({max1} unique objects)")
    else:
        print(f"Category '{cat1}' has {len(view1)} samples ({len(labels1)} unique objects), no object limit applied.")

    # Filter for category 2 and apply max_objects2
    view2 = original_dataset_view.match({"category.label": cat2})
    labels2 = view2.distinct("ground_truth.label")
    if len(labels2) > max2:
        np.random.seed(args.seed) # Use same seed for reproducibility
        selected_labels2 = np.random.choice(labels2, max2, replace=False).tolist()
        view2 = view2.match({"ground_truth.label": {"$in": selected_labels2}})
        print(f"Filtered '{cat2}' to {len(view2)} samples ({max2} unique objects)")
    else:
         print(f"Category '{cat2}' has {len(view2)} samples ({len(labels2)} unique objects), no object limit applied.")

    # Filter for category 3 and apply max_objects3
    view3 = original_dataset_view.match({"category.label": cat3})
    labels3 = view3.distinct("ground_truth.label")
    if len(labels3) > max3:
        np.random.seed(args.seed) # Use same seed for reproducibility
        selected_labels3 = np.random.choice(labels3, max3, replace=False).tolist()
        view3 = view3.match({"ground_truth.label": {"$in": selected_labels3}})
        print(f"Filtered '{cat3}' to {len(view3)} samples ({max3} unique objects)")
    else:
         print(f"Category '{cat3}' has {len(view3)} samples ({len(labels3)} unique objects), no object limit applied.")

    # Combine the IDs from all three views
    filtered_ids = view1.values("id") + view2.values("id") + view3.values("id")
    final_filtered_dataset = original_dataset_view.select(filtered_ids) # Store the final filtered dataset view
    print(f"Combined filtered dataset has {len(final_filtered_dataset)} samples.")

# --- Standard Filtering Cases ---
else:
    # Apply standard category filtering first
    if categories:
        dataset = dataset.match({"category.label": {"$in": categories}})
        print(f"Filtered dataset to categories: {categories}")

    # Apply standard max_objects filtering
    if args.max_objects is not None:
        unique_labels = dataset.distinct("ground_truth.label")
        if len(unique_labels) > args.max_objects:
            np.random.seed(args.seed)
            selected_labels = np.random.choice(unique_labels, args.max_objects, replace=False).tolist()
            dataset = dataset.match({"ground_truth.label": {"$in": selected_labels}})
            print(f"Filtered dataset to {args.max_objects} unique objects out of {len(unique_labels)}")
        else:
            print(f"Dataset already has {len(unique_labels)} <= {args.max_objects} unique objects. No further object filtering applied.")

    # Assign the result of standard filtering
    final_filtered_dataset = dataset.view() # Take a view of the modified dataset

    # Warn if max_objects2 was given but not used
    if args.max_objects2 is not None and not (len(categories) == 2 and args.max_objects is not None):
         print("Warning: --max-objects2 is specified but the conditions for its use (exactly two categories and --max-objects) were not met. --max-objects2 ignored.")
    
    # Warn if max_objects3 was given but not used
    if args.max_objects3 is not None and not (len(categories) == 3 and args.max_objects is not None and args.max_objects2 is not None):
         print("Warning: --max-objects3 is specified but the conditions for its use (exactly three categories, --max-objects, and --max-objects2) were not met. --max-objects3 ignored.")


# --- Post Filtering ---
# Update the main dataset variable to the final filtered result
dataset = final_filtered_dataset
print("Final filtered dataset view:")
print(dataset)

# Load first image embeddings
embed_img_path = os.path.join(args.embed_dir, args.embed_img)
print(f"Loading first image embeddings from {embed_img_path}")
img_data = torch.load(embed_img_path, map_location=torch.device('cpu'))
img_embeddings = img_data['embeddings']
img_labels = img_data['labels']

# Store the filtered sample labels for reuse
filtered_sample_labels = [sample.ground_truth.label for sample in dataset]

# Filter and process first image embeddings
# Check if any filtering was applied that requires embedding adjustment
if args.category is not None or args.max_objects is not None or args.max_objects2 is not None or args.max_objects3 is not None:
    # Create a mapping from labels to their embedding indices
    label_to_emb_indices = {}
    for i, label in enumerate(img_labels):
        label = int(label.item())
        if label not in label_to_emb_indices:
            label_to_emb_indices[label] = []
        label_to_emb_indices[label].append(i)

    # Filter embeddings to match the order of samples in the filtered dataset
    filtered_emb_indices = []
    for label in filtered_sample_labels:
        label = int(label)
        if label in label_to_emb_indices and label_to_emb_indices[label]:
            # Use the first matching embedding index
            emb_idx = label_to_emb_indices[label].pop(0)
            filtered_emb_indices.append(emb_idx)

    # Filter the embeddings based on their type
    if hasattr(img_embeddings, 'index_select'):
        # If it's a PyTorch tensor
        filtered_indices_tensor = torch.tensor(filtered_emb_indices, dtype=torch.long)
        img_embeddings = img_embeddings.index_select(0, filtered_indices_tensor)
    else:
        # If it's a NumPy array
        img_embeddings = img_embeddings[filtered_emb_indices]

    print(f"Filtered first image embeddings: {len(filtered_emb_indices)} out of {len(img_labels)}")

# Normalize first image embeddings
img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)

# Initialize second image embeddings to None
img_embeddings2 = None
img_labels2 = None

# Load second image embeddings if provided
if args.embed_img2 and os.path.exists(os.path.join(args.embed_dir, args.embed_img2)):
    embed_img2_path = os.path.join(args.embed_dir, args.embed_img2)
    print(f"Loading second image embeddings from {embed_img2_path}")
    img_data2 = torch.load(embed_img2_path, map_location=torch.device('cpu'))
    img_embeddings2 = img_data2['embeddings']
    img_labels2 = img_data2['labels']

    # Filter and process second image embeddings
    # Check if any filtering was applied that requires embedding adjustment
    if args.category is not None or args.max_objects is not None or args.max_objects2 is not None or args.max_objects3 is not None:
        # Create a mapping from labels to their embedding indices
        label_to_emb_indices2 = {}
        for i, label in enumerate(img_labels2):
            label = int(label.item())
            if label not in label_to_emb_indices2:
                label_to_emb_indices2[label] = []
            label_to_emb_indices2[label].append(i)

        # Filter embeddings to match the order of samples in the filtered dataset
        filtered_emb_indices2 = []
        for label in filtered_sample_labels:
            label = int(label)
            if label in label_to_emb_indices2 and label_to_emb_indices2[label]:
                # Use the first matching embedding index
                emb_idx = label_to_emb_indices2[label].pop(0)
                filtered_emb_indices2.append(emb_idx)

        # Filter the embeddings based on their type
        if hasattr(img_embeddings2, 'index_select'):
            # If it's a PyTorch tensor
            filtered_indices_tensor2 = torch.tensor(filtered_emb_indices2, dtype=torch.long)
            img_embeddings2 = img_embeddings2.index_select(0, filtered_indices_tensor2)
        else:
            # If it's a NumPy array
            img_embeddings2 = img_embeddings2[filtered_emb_indices2]

        print(f"Filtered second image embeddings: {len(filtered_emb_indices2)} out of {len(img_labels2)}")

    # Normalize second image embeddings
    img_embeddings2 = img_embeddings2 / img_embeddings2.norm(dim=-1, keepdim=True)

# Load sequence embeddings if provided
seq_embeddings = None
seq_labels = None
if args.embed_seq and os.path.exists(os.path.join(args.embed_dir, args.embed_seq)):
    # Load sequence embeddings
    embed_seq_path = os.path.join(args.embed_dir, args.embed_seq)
    print(f"Loading sequence embeddings from {embed_seq_path}")
    seq_data = torch.load(embed_seq_path, map_location=torch.device('cpu'))
    seq_embeddings = seq_data['embeddings']
    seq_labels = seq_data['labels']

    print(f"Loaded sequence embeddings: shape {seq_embeddings.shape}, labels {len(seq_labels)}")

    # If dataset is filtered by category or max objects, filter sequence embeddings accordingly
    # Check if any filtering was applied that requires embedding adjustment
    if args.category is not None or args.max_objects is not None or args.max_objects2 is not None or args.max_objects3 is not None:
        # Create a mapping from labels to their embedding indices for sequence embeddings
        seq_label_to_emb_indices = {}
        for i, label in enumerate(seq_labels):
            label = int(label.item())
            if label not in seq_label_to_emb_indices:
                seq_label_to_emb_indices[label] = []
            seq_label_to_emb_indices[label].append(i)

        # Filter sequence embeddings to match the filtered dataset
        filtered_seq_emb_indices = []
        for label in filtered_sample_labels:
            label = int(label)
            if label in seq_label_to_emb_indices and seq_label_to_emb_indices[label]:
                # Use the first matching embedding index
                emb_idx = seq_label_to_emb_indices[label].pop(0)
                filtered_seq_emb_indices.append(emb_idx)

        # Filter the sequence embeddings based on their type
        if hasattr(seq_embeddings, 'index_select'):
            # If it's a PyTorch tensor
            filtered_indices_tensor = torch.tensor(filtered_seq_emb_indices, dtype=torch.long)
            seq_embeddings = seq_embeddings.index_select(0, filtered_indices_tensor)
        else:
            # If it's a NumPy array
            seq_embeddings = seq_embeddings[filtered_seq_emb_indices]

        print(f"Filtered sequence embeddings: {len(filtered_seq_emb_indices)} out of {len(seq_labels)}")

    # Normalize sequence embeddings
    seq_embeddings = seq_embeddings / seq_embeddings.norm(dim=-1, keepdim=True)

# Create a combined visualization dataset
combined_dataset_name = f"{name}_combined"
try:
    if fo.dataset_exists(combined_dataset_name):
        fo.delete_dataset(combined_dataset_name)
        print(f"Deleted existing dataset '{combined_dataset_name}'")
except Exception as e:
    print(f"Warning when checking/deleting existing dataset: {e}")

# Create a new dataset for visualization
viz_dataset = fo.Dataset(combined_dataset_name)

# Pre-fetch all samples and create a label-to-sample mapping for efficiency
all_samples = list(dataset)
label_to_sample = {}
for sample in all_samples:
    label_to_sample[sample.ground_truth.label] = sample

# Add samples for first image embeddings
for i, label in enumerate(img_labels):
    label_str = str(int(label.item()))
    if label_str in label_to_sample:
        sample = label_to_sample[label_str].copy()
        sample["embedding_type"] = fo.Classification(label="image1")
        viz_dataset.add_sample(sample)

# Add samples for second image embeddings if provided
if img_embeddings2 is not None:
    for i, label in enumerate(img_labels2):
        label_str = str(int(label.item()))
        if label_str in label_to_sample:
            sample = label_to_sample[label_str].copy()
            sample["embedding_type"] = fo.Classification(label="image2")
            viz_dataset.add_sample(sample)

# Add samples for sequence embeddings if provided
if seq_embeddings is not None:
    for i, label in enumerate(seq_labels):
        label_str = str(int(label.item()))
        if label_str in label_to_sample:
            sample = label_to_sample[label_str].copy()
            sample["embedding_type"] = fo.Classification(label="sequence")
            viz_dataset.add_sample(sample)

# --- Load and Add Center Embeddings (if specified) ---
center_embeddings = None
center_label_map = {}
if args.center_weights_path:
    print(f"Loading center weights from {args.center_weights_path}")
    if os.path.exists(args.center_weights_path):
        try:
            # Load the tensor directly, as saved by train.py
            center_embeddings = torch.load(args.center_weights_path, map_location='cpu')

            # Ensure it's a 2D tensor [1, embedding_dim]
            if center_embeddings.dim() == 1:
                center_embeddings = center_embeddings.unsqueeze(0)

            if center_embeddings.shape[0] != 1:
                 print(f"  Warning: Expected a single center, but loaded tensor has shape {center_embeddings.shape}. Using the first row.")
                 center_embeddings = center_embeddings[0:1]

            print(f"  Loaded single center embedding with shape: {center_embeddings.shape}")
            # Normalize center
            center_embeddings = center_embeddings / center_embeddings.norm(dim=-1, keepdim=True)

            # Load label map if provided (for the single center, index "0")
            center_label = "person_center" # Default label
            if args.center_label_map_path and os.path.exists(args.center_label_map_path):
                try:
                    with open(args.center_label_map_path, 'r') as f:
                        center_label_map = json.load(f)
                    center_label = center_label_map.get("0", center_label) # Try to get label for index 0
                    print(f"  Loaded center label map from {args.center_label_map_path}. Using label: '{center_label}'")
                except Exception as e:
                    print(f"  Warning: Failed to load or parse center label map: {e}. Using default label.")
            else:
                print(f"  No center label map provided or found. Using default label '{center_label}'.")

            # Add a single dummy sample for the center
            center_sample = fo.Sample(
                filepath="center_0", # Dummy filepath
                ground_truth=fo.Classification(label=str(center_label)), # Use the determined label
                category=fo.Classification(label=str(center_label)),
                embedding_type=fo.Classification(label="center")
            )
            viz_dataset.add_sample(center_sample)
            print(f"  Added center sample to the visualization dataset.")

        except Exception as e:
            print(f"  Error loading center weights from {args.center_weights_path}: {e}")
            center_embeddings = None # Ensure it's None on error
    else:
        print(f"  Warning: Center weights path does not exist: {args.center_weights_path}")

# Combine all embeddings
all_embeddings = [img_embeddings]
if img_embeddings2 is not None:
    all_embeddings.append(img_embeddings2)
if seq_embeddings is not None:
    all_embeddings.append(seq_embeddings)
if center_embeddings is not None:
    all_embeddings.append(center_embeddings)

combined_embeddings = torch.cat(all_embeddings, dim=0)

# Check if the number of samples matches the number of embeddings
if len(viz_dataset) != len(combined_embeddings):
    print(
        f"Warning: Number of samples ({len(viz_dataset)}) does not match number of embeddings ({len(combined_embeddings)})")

# Use `color_by="value"` so that distinct label values (e.g., categories) receive
# unique colors drawn from our custom pool.
color_scheme = fo.ColorScheme(color_pool=DISTINCT_COLORS, color_by="value")

# Compute visualization on combined embeddings
results = fob.compute_visualization(
    viz_dataset,
    embeddings=combined_embeddings.detach().numpy(),
    num_dims=2,
    method="umap",
    brain_key=f"{name}_combined",
    verbose=True,
    seed=args.seed,
    color_scheme=color_scheme,
)

viz_dataset.load_brain_results(f"{name}_combined")

# Launch FiftyOne App with the custom color scheme so that the distinct colors
# are applied throughout the UI (sample grid, embeddings panel, etc.)
session = fo.launch_app(viz_dataset, port=args.port, color_scheme=color_scheme)
session.wait()