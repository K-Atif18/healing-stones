# GSoc 2025 | Healing Stones

A comprehensive pipeline for analyzing and reconstructing fragmented artifacts using multi-scale cluster analysis. This system processes 3D point cloud fragments, creates hierarchical clusters at different scales, and extracts assembly knowledge for automated reconstruction.

## Overview

The pipeline consists of three main components:

1. **Cluster Generation** - Creates multi-scale surface clusters from fragment point clouds
2. **Ground Truth Extraction** - Extracts cluster-level ground truth from positioned fragments  
3. **Assembly Knowledge Extraction** - Generates assembly matches and topology analysis

## Features

- **Multi-scale clustering** at 1k, 5k, and 10k point levels
- **Color-based surface segmentation** (break surfaces vs. original surfaces)
- **Hierarchical cluster organization** with spatial indexing
- **Ground truth validation** with proximity-based contact detection
- **Assembly matching** using geometric similarity metrics
- **Comprehensive reporting** and visualization support

## Requirements

```python
numpy
open3d
scikit-learn
scipy
networkx
h5py
tqdm
```

## Usage

### 1. Cluster Generation (cluster_generation.py)

Processes raw fragment PLY files to create hierarchical clusters:

```bash
python unified_cluster_generation.py \
    --data_dir "Ground_Truth/artifact_1" \
    --output_dir "output" \
    --point_scales 1000 5000 10000 \
    --green_threshold 0.6
```

**Key Parameters:**
- `--data_dir`: Directory containing fragment PLY files
- `--point_scales`: Target points per cluster (creates 1k/5k/10k scales)
- `--green_threshold`: Threshold for detecting break surfaces (green-colored)
- `--original_downsample`: Max points to keep from original surfaces

**Outputs:**
- `segmented_fragments.pkl` - Fragment segmentation with point assignments
- `feature_clusters.pkl` - Hierarchical cluster organization
- `cluster_mappings.pkl` - Complete mapping structures & spatial indices
- `cluster_registry.pkl` - Direct cluster object access
- `unified_pipeline_report.json` - Processing summary

### 2. Ground Truth Extraction (Multi_Scale_gt.py)

Extracts cluster-level ground truth from positioned fragments:

```bash
python multi_scale_gt_extractor.py \
    --positioned-dir "Ground_Truth/reconstructed/artifact_1" \
    --clusters "output/feature_clusters.pkl" \
    --segments "output/segmented_fragments.pkl" \
    --contact-threshold 2.0
```

**Key Parameters:**
- `--positioned-dir`: Directory with correctly positioned fragment PLY files
- `--contact-threshold`: Distance threshold for point-to-point contact (2mm)
- `--max-cluster-distance`: Maximum distance between cluster centers (50mm)
- `--visualize`: Number of top matches to visualize per scale

**Outputs:**
- `multi_scale_cluster_ground_truth.json` - Cluster-level ground truth matches
- `multi_scale_cluster_ground_truth.h5` - HDF5 format for efficient loading

### 3. Assembly Knowledge Extraction (Third Step)

Generates assembly matches and knowledge graphs:

```bash
python unified_assembly_extractor.py \
    --clusters "output/feature_clusters.pkl" \
    --segments "output/segmented_fragments.pkl" \
    --ground_truth "Ground_Truth/multi_scale_cluster_ground_truth.json" \
    --ply_dir "Ground_Truth/reconstructed/artifact_1"
```

**Key Parameters:**
- `--ground_truth`: Multi-scale ground truth file from step 2
- `--ply_dir`: Directory with positioned PLY files
- `--scales`: Scales to process (1k, 5k, 10k)

**Outputs:**
- `unified_multi_scale_assembly.h5` - Complete assembly analysis
- `unified_assembly_report.txt` - Human-readable summary
- `report/detailed_analysis.txt` - Detailed ground truth analysis

## Data Formats

### Input Requirements

**Fragment PLY Files:**
- Point clouds with RGB colors
- Green-colored break surfaces (G > 0.6, R < 0.4, B < 0.4)
- Positioned fragments for ground truth extraction

**Directory Structure:**
```
Ground_Truth/
├── artifact_1/              # Original fragments
│   ├── fragment_1.ply
│   ├── fragment_2.ply
│   └── ...
└── reconstructed/
    └── artifact_1/           # Positioned fragments
        ├── fragment_1.ply
        ├── fragment_2.ply
        └── ...
```


## Key Algorithms

### Multi-Scale Clustering
- **Color Segmentation**: Separates break surfaces from original surfaces
- **Balanced Assignment**: Ensures equal cluster sizes using iterative assignment  
- **Smart Initialization**: Density-aware center placement for better convergence

### Ground Truth Extraction
- **Contact Detection**: Multiple sampling strategies for robust fragment contact detection
- **Proximity Matching**: Strict 2mm point-to-point proximity for cluster matching
- **Confidence Scoring**: Weighted combination of distance, size, and overlap metrics

### Assembly Matching
- **Geometric Similarity**: Normal alignment, size matching, shape compatibility
- **Spatial Proximity**: Distance-based similarity without hard thresholds
- **Multi-scale Analysis**: Consistent matching across 1k/5k/10k scales

## Performance Metrics

The pipeline tracks several key metrics:

- **Cluster Statistics**: Points per cluster, assignment rates, spatial distribution
- **Ground Truth Coverage**: Recall rates, contact type distribution, confidence scores
- **Assembly Quality**: Match confidence, geometric similarity, topology features


