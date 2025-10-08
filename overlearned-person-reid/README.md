# Overlearned Person Re-identification and Its Mitigation

Code for the paper "Emergent AI Surveillance: Overlearned Person Re-Identification and Its Mitigation in Law Enforcement Context".

If you use this code in your research, please cite our paper: https://arxiv.org/abs/2510.06026


## Setup

### Requirements
1. Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Clone the repository:
   ```bash
   git clone https://github.com/AlpakkAn/overlearned-person-reid.git
   cd overlearned-person-reid/
   ```
3. Create environment:
   ```bash
   conda env create -f environment.yml
   ```
   Note: If you encounter CUDA version errors, create a fresh conda environment and follow PyTorch installation instructions (https://pytorch.org/), then install the other packages manually.

We used Python 3.8.18 in a CUDA 11.8 environment.

### Key Dependencies
- PyTorch with CUDA support
- pytorch-metric-learning
- timm (for pre-trained models)
- optuna (for hyperparameter optimization)
- FAISS (for efficient similarity search)
- OpenCV, PIL, matplotlib (for image processing and visualization)
- zennit, lxt (for AttnLRP explanations)

## Dataset Setup

### Download Required Datasets
1. **YouTube-VIS 2021/2022**: Download train split from https://youtube-vos.org/dataset/vis/
2. **OVIS**: Download train split and annotations_train.json from https://songbai.site/ovis/index.html#download
3. **CUHK03**: For person re-identification evaluation. Masks: https://github.com/developfeng/MGCAM/tree/master/data
4. **Market-1501**: For person re-identification evaluation. Masks: https://github.com/developfeng/MGCAM/tree/master/data

### Prepare Dataset Structure
1. Create directory structure:
   ```
   datasets/
   ├── vis/
   │   ├── JPEGImages/
   │   └── instances.json
   └── ovis/
       ├── JPEGImages/
       └── annotations_train.json
   ```

2. Run dataset setup:
   ```bash
   python setup_dataset.py --vis datasets/vis --ovis datasets/ovis
   ```

### Final Dataset Structure
After setup, the structure will be:
```
datasets/
├── vis/
│   ├── instances.json
│   ├── JPEGImages/
│   ├── train/
│   │   ├── instances.json
│   │   ├── fg_instances.json
│   │   ├── foregroundImages_cropped/
│   │   └── fg_instances_cropped.json
│   ├── valid/
│   │   ├── instances.json
│   │   ├── fg_instances.json
│   │   ├── foregroundImages_cropped/
│   │   └── fg_instances_cropped.json
│   └── test/
│       ├── instances.json
│       ├── fg_instances.json
│       ├── foregroundImages_cropped/
│       └── fg_instances_cropped.json
└── ovis/
    └── test/
        ├── foregroundImages_cropped/
        └── fg_instances_cropped.json
```

## Training

### Basic Training
```bash
python train.py --dataset datasets/vis --batch-size 64 --epochs 100 --out-dir results
```

### Advanced Training Options
```bash
# Use different loss functions
python train.py --loss confusion --conf_weight 10.0 --conf_gamma 5.0

# Enable hyperparameter tuning
python train.py --tune --n_trials 50

# Evaluate person queries during training
python train.py --eval_person_queries

# Use different models
python train.py --trunk torchvision.vit_b_16

# Enable debugging mode
python train.py --debug --subset
```

### Training Parameters
- `--loss`: Choose from 'ms' (multi-similarity), 'confusion', 'center'
- `--batch-size`: Training batch size (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate for trunk (default: 0.00001)
- `--lr_embed`: Learning rate for embedder (default: 0.00001)
- `--embed_dim`: Embedding dimension (default: 512)
- `--trunk`: Model backbone (default: ViT-B/16 CLIP)
- `--freeze`: Freeze backbone weights
- `--tune`: Enable hyperparameter optimization
- `--eval_person_queries`: Evaluate person instances during training

### Hyperparameters in Paper
These are the hyperparameters we ended up with in our paper after tuning:

#### MS (CLIP)
- `lr`: 2.6036594850360824e-06
- `lr_embed`: 1.8008398609432192e-06

We used the default values for `ms_alpha`, `ms_beta`, and `ms_base`.

#### Confusion
- `lr`: 8.069262871942484e-06
- `lr_embed`: 9.896641755220117e-06
- `conf_weight`: 23.752884837928374
- `conf_gamma`: 31.85830217870389
- `conf_margin`: 0.012645424823545037

## Testing

### Basic Evaluation
```bash
python test.py --dataset datasets/vis --trunk path/to/trunk.pth --embedder path/to/embedder.pth --test-persons
```

### Comprehensive Evaluation
```bash
# Evaluate on multiple datasets
python test.py --dataset datasets/vis --cuhk datasets/cuhk03-np --market datasets/market1501 --trunk path/to/trunk.pth --embedder path/to/embedder.pth --test-persons

# Generate visualizations
python test.py --out-dir visualizations --trunk path/to/trunk.pth --embedder path/to/embedder.pth --test-persons

# AttnLRP explanations
python test.py --attnlrp --trunk path/to/trunk.pth --embedder path/to/embedder.pth
```

### Index Exclusion
```bash
# Get object detections first
python utils/detect.py --root_dir datasets/vis/JPEGImages --box_threshold 0.2 --text_threshold 0.1 --dataset_split datasets/vis/test/instances.json --output det_vis.json

# Get segmentation masks
python utils/segment_detections.py --input det_vis.json --image-root datasets/vis/JPEGImages --output mask_vis.json

# Create index exclusion file
python utils/index_exclusion.py --gt_json datasets/vis/test/fg_instances.json --detection_json det_vis.json --mask_json mask_vis.json --output vis_exclusion.json

# Create index exclusion file (for cuhk or market)
python utils/index_exclusion.py --detection_json det_cuhk.json --mask_json mask_cuhk.json --mask_dir datasets/cuhk03-np/labeled/bounding_box_test_seg/ --image-root datasets/cuhk03-np/labeled/bounding_box_test --output cuhk_exclusion.json

# Evaluate with index exclusion file
python test.py --dataset datasets/vis --trunk path/to/trunk.pth --embedder path/to/embedder.pth --test-persons --index-exclusion-json vis_exclusion.json
```

### Testing Parameters
- `--split`: Dataset split to evaluate ('test', 'valid')
- `--distractors`: Path to distractor dataset
- `--cuhk`: Path to CUHK03 dataset
- `--market`: Path to Market-1501 dataset
- `--out-dir`: Output directory for visualizations
- `--test-persons`: Evaluate using person images as queries
- `--save-embeddings`: Save computed embeddings
- `--index-exclusion-json`: JSON file for excluding specific indices
- `--attnlrp`: Generate attention-based explanations
- `--plots-only`: Generate only visualizations (skip evaluation)

## Utility Scripts

### Dataset Utilities
- `setup_dataset.py`: Convert and prepare datasets for training
- `foreground_dataset.py`: Dataset classes for loading cropped foreground objects

### Visualization and Analysis
- `utils/visualize.py`: Generate embedding space visualizations
- `utils/index_exclusion.py`: Create index exclusion files for evaluation
- `utils/detect.py`: Object detection utilities
- `utils/segment_detections.py`: Segmentation mask generation

### Preprocessing
- `utils/crop_persons.py`: Crop person images from CUHK03
- `utils/get_clothes.py`: Extract clothing regions from person images

## Loss Functions

### Multi-Similarity Loss
Standard metric learning loss for embedding similarity.
```bash
python train.py --loss ms --ms_alpha 2.0 --ms_beta 50.0 --ms_base 0.5
```

### Confusion Loss
Custom loss that pushes embeddings of samples belonging to the person category away from each other.
```bash
python train.py --loss confusion --conf_weight 24.0 --conf_gamma 32.0 --conf_margin 0.01
```

## Hyperparameter Optimization

Enable hyperparameter tuning with Optuna:
```bash
python train.py --tune --n_trials 50 --epochs 10
```

Optimizes:
- Learning rates (trunk and embedder)
- Loss-specific parameters
- Architecture parameters

## AttnLRP Explanations
Generate attention-based explanations for Vision Transformers:
```bash
python test.py --attnlrp --trunk path/to/vit_trunk.pth
```

**Note**: For the AttnLRP heatmaps in our paper, we trained `torchvision.vit_b_16` models in the same manner as the corresponding models described in the paper, and generated the heatmaps from these torchvision models.


## Output Structure

Training outputs:
```
results/
├── trunk_final.pth
├── embedder_final.pth
├── hyperparameter_study.db (if tuning enabled)
```

Testing outputs:
```
output_dir/
├── embeddings/
├── visualizations/
│   ├── best_queries/
│   ├── worst_queries/
│   └── attnlrp/ (if enabled)
```
