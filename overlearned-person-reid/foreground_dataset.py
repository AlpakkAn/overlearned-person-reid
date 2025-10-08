import os
import glob
import json
import hashlib
import copy

import torch
from torchvision.transforms.transforms import ToTensor
from PIL import Image


class ForegroundDataset(torch.utils.data.Dataset):
    _json_cache = {}

    def __init__(self, root, transforms=None, persons_as_neg=False,
                 remove_persons=False, persons_only=False, ovis=False):
        self.root = root
        self.transforms = transforms
        self.special_category_26 = persons_as_neg
        self.forbidden_labels = set()
        self.person_category_id = 1 if ovis else 26

        # ------------------------------------------------------------------
        # Speed-up: cache annotation JSON to avoid expensive disk I/O & parsing
        # ------------------------------------------------------------------
        json_path = os.path.join(root.rsplit('/', 1)[0], 'fg_instances_cropped.json')

        # Use a simple global dict as cache because the Dataset object can be
        # instantiated many times within a single run (e.g. query, gallery,
        # distractors, etc.). Parsing the same large JSON file repeatedly was
        # a significant overhead before the actual embedding computation.

        if not hasattr(ForegroundDataset, "_json_cache"):
            ForegroundDataset._json_cache = {}

        if json_path in ForegroundDataset._json_cache:
            data = ForegroundDataset._json_cache[json_path]
        else:
            with open(json_path, "r") as jf:
                data = json.load(jf)
            ForegroundDataset._json_cache[json_path] = data

        # Filter images based on person category if requested
        if remove_persons or persons_only:
            # Get all image IDs that match our criteria
            if remove_persons:
                # Get image IDs of non-person images (category_id != person_category_id)
                filtered_img_ids = set()
                for ann in data['annotations']:
                    if ann['category_id'] != self.person_category_id:
                        filtered_img_ids.add(ann['image_id'])
            elif persons_only:
                # Get image IDs of person images (category_id == person_category_id)
                filtered_img_ids = set()
                for ann in data['annotations']:
                    if ann['category_id'] == self.person_category_id:
                        filtered_img_ids.add(ann['image_id'])
            
            # Filter a *copy* of the original images list from the potentially cached data
            current_images = []
            for img in data['images']: # Iterate over original list from cache/file
                if img['id'] in filtered_img_ids:
                    current_images.append(copy.deepcopy(img)) # Build the filtered list for *this instance*
        else:
            # If no filtering, use a deep copy of the original list for this instance
            # This prevents downstream modifications from affecting the cache
            current_images = copy.deepcopy(data['images'])

        # Create a mapping of image filenames to their IDs
        filename_to_id = {img['file_name']: img['id'] for img in current_images}
        
        # Create a mapping of image IDs to category IDs
        img_to_category = {}
        for ann in data['annotations']:
            img_to_category[ann['image_id']] = ann['category_id']
            
        # Sort the image filenames
        self.imgs = list(sorted([img['file_name'] for img in current_images]))
        
        # Generate labels in the same order as the sorted images
        labels = []
        for img_name in self.imgs:
            img_id = filename_to_id[img_name]
            category_id = img_to_category.get(img_id, -1)
            
            # Extract original label from filename
            original_label = int(img_name.rsplit('_', 1)[1].rsplit('.', 1)[0])
            
            # Check for person category unconditionally to build forbidden_labels list
            if category_id == self.person_category_id:
                # Add original label to forbidden_labels set regardless of special_category_26
                self.forbidden_labels.add(original_label)
            
            if self.special_category_26 and category_id == self.person_category_id:
                # For person category, use a deterministic hash of the filename.
                # The MD5 hash is used to generate a unique ID that remains consistent across runs.
                md5_hash = hashlib.md5(img_name.encode('utf-8')).hexdigest()
                unique_id = int(md5_hash, 16) % 90000 + 10000
                labels.append(unique_id)
            else:
                # For other cases, use the original label from filename
                labels.append(original_label)
        
        # Convert set to list
        self.forbidden_labels = list(self.forbidden_labels)
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = ToTensor()(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class LabelModifiedDatasetWrapper(torch.utils.data.Dataset):
    """
    A dataset wrapper that returns modified labels but keeps original data.
    This allows us to use the original dataset structure but change the labels used for loss computation.
    """
    def __init__(self, original_dataset, modified_labels):
        self.original_dataset = original_dataset
        self.modified_labels = modified_labels
        
    def __getitem__(self, idx):
        item = self.original_dataset[idx]
        # Replace the label with our modified label
        if isinstance(item, tuple) and len(item) == 2:
            # Most common case: (data, label)
            return item[0], self.modified_labels[idx]
        else:
            # Just in case the dataset returns something else
            return item
            
    def __len__(self):
        return len(self.original_dataset)


class Cuhk03(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        imgs = []
        for filename in glob.iglob(root + '/*.*',
                                   recursive=False):
            fn = os.path.relpath(filename).rsplit('/', 1)
            imgs.append(fn[-1])
        self.imgs = imgs

        self.labels = torch.tensor([int(img.split('_', 1)[0]) for img in self.imgs])

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)
            return img, label
        else:
            img = ToTensor()(img)
            return img, label

    def __len__(self):
        return len(self.imgs)
