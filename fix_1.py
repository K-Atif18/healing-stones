#!/usr/bin/env python3
"""
Fix the cluster-fragment mapping issue.
This script updates the feature_clusters.pkl file to include proper fragment associations.
"""

import numpy as np
import pickle
from pathlib import Path
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_cluster_fragment_mapping():
    """Fix the cluster-fragment associations based on the processing order."""
    
    # Load the data
    logger.info("Loading cluster and segment data...")
    
    with open('output/feature_clusters.pkl', 'rb') as f:
        cluster_data = pickle.load(f)
    
    with open('output/segmented_fragments.pkl', 'rb') as f:
        segment_data = pickle.load(f)
    
    # Get fragment names in sorted order (same order as processing)
    fragment_names = sorted(segment_data.keys())
    logger.info(f"Fragments: {fragment_names}")
    
    # Get the number of clusters per fragment from segment data
    fragment_cluster_counts = []
    for frag_name in fragment_names:
        n_clusters = segment_data[frag_name].get('n_clusters', 0)
        fragment_cluster_counts.append(n_clusters)
        logger.info(f"{frag_name}: {n_clusters} clusters")
    
    # Verify total matches
    total_reported = sum(fragment_cluster_counts)
    total_actual = len(cluster_data['clusters'])
    logger.info(f"\nTotal clusters reported: {total_reported}")
    logger.info(f"Total clusters actual: {total_actual}")
    
    if total_reported != total_actual:
        logger.warning("Mismatch in cluster counts! Using actual distribution from processing.")
    
    # Assign fragments to clusters based on the processing order
    updated_clusters = []
    cluster_index = 0
    
    for frag_idx, (frag_name, n_clusters) in enumerate(zip(fragment_names, fragment_cluster_counts)):
        # Assign the next n_clusters to this fragment
        for i in range(n_clusters):
            if cluster_index < len(cluster_data['clusters']):
                cluster = cluster_data['clusters'][cluster_index].copy()
                cluster['fragment'] = frag_name
                cluster['fragment_index'] = frag_idx
                updated_clusters.append(cluster)
                cluster_index += 1
    
    # Update the cluster data
    cluster_data['clusters'] = updated_clusters
    
    # Save the updated data
    output_file = Path('output/feature_clusters_fixed.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(cluster_data, f)
    
    logger.info(f"\nSaved fixed cluster data to {output_file}")
    
    # Verify the mapping
    logger.info("\nVerifying cluster-fragment mapping:")
    fragment_counts = {}
    for cluster in updated_clusters:
        frag = cluster.get('fragment', 'unknown')
        fragment_counts[frag] = fragment_counts.get(frag, 0) + 1
    
    for frag_name in fragment_names:
        count = fragment_counts.get(frag_name, 0)
        expected = segment_data[frag_name].get('n_clusters', 0)
        status = "✓" if count == expected else "✗"
        logger.info(f"{status} {frag_name}: {count} clusters (expected: {expected})")
    
    return output_file

def verify_ground_truth_data():
    """Verify the ground truth data structure."""
    logger.info("\nVerifying ground truth data...")
    
    with open('Ground_Truth/blender/ground_truth_assembly.json', 'r') as f:
        gt_data = json.load(f)
    
    logger.info(f"Fragments in GT: {len(gt_data['fragments'])}")
    logger.info(f"Contact pairs: {len(gt_data['contact_pairs'])}")
    logger.info(f"Contact details: {len(gt_data.get('contact_details', []))}")
    
    # Check if we have contact details
    if 'contact_details' in gt_data and gt_data['contact_details']:
        sample = gt_data['contact_details'][0]
        logger.info(f"\nSample contact detail fields: {list(sample.keys())}")
        if 'contact_indices_1' in sample:
            logger.info(f"Contact indices available: Yes")
    else:
        logger.warning("No contact details found in ground truth!")
    
    return gt_data

def create_fragment_point_cloud_mapping():
    """Create a mapping between fragments and their original point cloud files."""
    
    # This creates a helper file that maps break surface points to fragments
    logger.info("\nCreating fragment-point mapping...")
    
    with open('output/segmented_fragments.pkl', 'rb') as f:
        segment_data = pickle.load(f)
    
    # Create mapping of break surface points to fragments
    fragment_break_points = {}
    
    for frag_name, frag_data in segment_data.items():
        if 'surface_patches' in frag_data:
            # Get break surface points
            break_points = []
            for patch_name, point_indices in frag_data['surface_patches'].items():
                if 'break' in patch_name:
                    break_points.extend(point_indices)
            
            fragment_break_points[frag_name] = {
                'break_surface_indices': break_points,
                'n_break_points': len(break_points),
                'n_total_points': frag_data.get('n_points', 0)
            }
            
            logger.info(f"{frag_name}: {len(break_points)} break surface points")
    
    # Save mapping
    with open('output/fragment_break_points_mapping.pkl', 'wb') as f:
        pickle.dump(fragment_break_points, f)
    
    logger.info("Saved fragment-point mapping")
    
    return fragment_break_points

def main():
    """Main execution."""
    import json
    
    logger.info("Fixing cluster-fragment mapping issue...")
    
    # Fix the cluster mapping
    fixed_file = fix_cluster_fragment_mapping()
    
    # Verify ground truth
    gt_data = verify_ground_truth_data()
    
    # Create additional mapping
    fragment_mapping = create_fragment_point_cloud_mapping()
    
    logger.info("\n" + "="*70)
    logger.info("FIXES APPLIED")
    logger.info("="*70)
    logger.info(f"\n1. Created fixed cluster file: {fixed_file}")
    logger.info("2. Verified ground truth data structure")
    logger.info("3. Created fragment-point mapping")
    
    logger.info("\nNEXT STEPS:")
    logger.info("1. Update the Phase 1.3 script to use 'feature_clusters_fixed.pkl'")
    logger.info("2. Re-run: python assembly_1.3.py --clusters output/feature_clusters_fixed.pkl")
    
    # Also update the assembly script to better handle the ground truth matching
    logger.info("\nAlternatively, I'll create an improved version of the assembly script...")

if __name__ == "__main__":
    main()