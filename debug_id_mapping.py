#!/usr/bin/env python3
"""
Debug the cluster ID mapping issue between training and evaluation
"""

import numpy as np
import torch
import h5py
import pickle
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_id_mapping():
    """Debug the cluster ID mapping issue."""
    
    # Load the data
    logger.info("Loading data files...")
    
    # Load cluster assembly file (has the GT matches)
    with h5py.File("output/cluster_assembly_with_gt.h5", 'r') as f:
        cluster_matches = f['cluster_matches']
        fragment_1 = [f.decode('utf8') for f in cluster_matches['fragment_1'][:]]
        fragment_2 = [f.decode('utf8') for f in cluster_matches['fragment_2'][:]]
        cluster_id_1 = cluster_matches['cluster_id_1'][:]
        cluster_id_2 = cluster_matches['cluster_id_2'][:]
        is_gt = cluster_matches['is_ground_truth'][:]
        confidences = cluster_matches['confidences'][:]
    
    # Load cluster features
    with open("output/feature_clusters_fixed.pkl", 'rb') as f:
        cluster_data = pickle.load(f)
    
    # Load segment data
    with open("output/segmented_fragments_with_indices.pkl", 'rb') as f:
        segment_data = pickle.load(f)
    
    logger.info("\nAnalyzing cluster ID mapping...")
    
    # Check a specific GT match
    gt_matches = [(f1, f2, c1, c2, conf) for f1, f2, c1, c2, gt, conf 
                  in zip(fragment_1, fragment_2, cluster_id_1, cluster_id_2, is_gt, confidences) 
                  if gt]
    
    logger.info(f"Total GT matches: {len(gt_matches)}")
    
    # Analyze first few GT matches
    for i, (f1, f2, c1, c2, conf) in enumerate(gt_matches[:5]):
        logger.info(f"\nGT Match {i+1}: {f1}:{c1} <-> {f2}:{c2} (conf: {conf:.3f})")
        
        # Check if these are local or global IDs
        logger.info(f"  Cluster ID 1: {c1}")
        logger.info(f"  Cluster ID 2: {c2}")
    
    # Now check how clusters are organized in the dataset
    logger.info("\n" + "="*60)
    logger.info("Checking cluster organization in dataset...")
    
    # Import the matcher to see how it builds cluster features
    from forward_search_matcher import ForwardSearchMatchingDataset
    
    dataset = ForwardSearchMatchingDataset(
        fragment_embeddings_file="output_2/fragment_embeddings.h5",
        cluster_assembly_file="output/cluster_assembly_with_gt.h5",
        clusters_file="output/feature_clusters_fixed.pkl"
    )
    
    # Check how cluster features are indexed
    logger.info(f"\nCluster features keys sample:")
    keys = list(dataset.cluster_features.keys())[:10]
    for key in keys:
        logger.info(f"  {key}")
    
    # Check specific fragments
    test_fragments = ['frag_1', 'frag_2']
    for frag in test_fragments:
        frag_keys = [(f, c) for f, c in dataset.cluster_features.keys() if f == frag]
        logger.info(f"\n{frag} has {len(frag_keys)} clusters")
        logger.info(f"  Cluster IDs: {sorted([c for _, c in frag_keys])[:10]}...")
    
    # Now check if the GT cluster IDs exist in the dataset
    logger.info("\n" + "="*60)
    logger.info("Checking if GT cluster IDs exist in dataset...")
    
    for i, (f1, f2, c1, c2, conf) in enumerate(gt_matches[:10]):
        key1 = (f1, c1)
        key2 = (f2, c2)
        
        exists1 = key1 in dataset.cluster_features
        exists2 = key2 in dataset.cluster_features
        
        logger.info(f"\nGT Match {i+1}:")
        logger.info(f"  {f1}:{c1} exists: {exists1}")
        logger.info(f"  {f2}:{c2} exists: {exists2}")
        
        if not exists1 or not exists2:
            # Try to find what the correct ID might be
            logger.info("  Looking for correct IDs...")
            
            # Check if it's using local vs global indexing
            if cluster_data and 'clusters' in cluster_data:
                # Find the cluster in the global list
                for j, cluster in enumerate(cluster_data['clusters']):
                    if cluster['cluster_id'] == c1:
                        logger.info(f"    Found cluster {c1} at global index {j}")
                        break
    
    # Check how predict_matches generates cluster pairs
    logger.info("\n" + "="*60)
    logger.info("Checking predict_matches cluster generation...")
    
    # Get clusters for frag_1 and frag_2
    clusters_1 = [k[1] for k in dataset.cluster_features.keys() if k[0] == 'frag_1']
    clusters_2 = [k[1] for k in dataset.cluster_features.keys() if k[0] == 'frag_2']
    
    logger.info(f"\nfrag_1 clusters in predict: {len(clusters_1)}")
    logger.info(f"  Sample IDs: {clusters_1[:10]}")
    
    logger.info(f"\nfrag_2 clusters in predict: {len(clusters_2)}")
    logger.info(f"  Sample IDs: {clusters_2[:10]}")
    
    # Check GT matches for frag_1 <-> frag_2
    frag_gt_matches = [(c1, c2) for f1, f2, c1, c2, _ in gt_matches 
                       if (f1 == 'frag_1' and f2 == 'frag_2') or 
                          (f1 == 'frag_2' and f2 == 'frag_1')]
    
    logger.info(f"\nGT matches for frag_1 <-> frag_2: {len(frag_gt_matches)}")
    for c1, c2 in frag_gt_matches[:5]:
        logger.info(f"  C{c1} <-> C{c2}")
        
        # Check if these exist in the predict lists
        in_list1 = c1 in clusters_1 or c2 in clusters_1
        in_list2 = c1 in clusters_2 or c2 in clusters_2
        logger.info(f"    In predict lists: {in_list1} and {in_list2}")


if __name__ == "__main__":
    debug_id_mapping()