import os
import shutil
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from typing import List, Tuple, Dict, Set, Optional
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Hard negative mining will use geometric heuristics instead of feature-based similarity.")

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def get_fragments(folder):
    return [f for f in os.listdir(folder) if f.endswith('.ply')]

def normalize_pair(frag_a, frag_b):
    """Normalize pair to ensure consistent ordering (alphabetical)."""
    return tuple(sorted([frag_a, frag_b]))

def collect_all_positive_pairs(testdata_path: str) -> Set[Tuple[str, str]]:
    """Collect all positive pairs from testdata folder to avoid duplicates and conflicts."""
    positive_pairs = set()
    pair_folders = [f for f in os.listdir(testdata_path) 
                   if f.startswith('pair_') and os.path.isdir(os.path.join(testdata_path, f))]
    
    print(f"Scanning {len(pair_folders)} testdata folders for positive pairs...")
    
    for pair_folder in pair_folders:
        pair_folder_path = os.path.join(testdata_path, pair_folder)
        fragments = get_fragments(pair_folder_path)
        
        if len(fragments) == 2:
            normalized_pair = normalize_pair(fragments[0], fragments[1])
            positive_pairs.add(normalized_pair)
            print(f"  Found positive pair: {normalized_pair[0]} <-> {normalized_pair[1]}")
        elif len(fragments) > 2:
            # Handle cases where a folder has more than 2 fragments (all combinations are positive)
            print(f"  Found multi-fragment positive group in {pair_folder}: {fragments}")
            for i in range(len(fragments)):
                for j in range(i + 1, len(fragments)):
                    normalized_pair = normalize_pair(fragments[i], fragments[j])
                    positive_pairs.add(normalized_pair)
                    print(f"    Added positive pair: {normalized_pair[0]} <-> {normalized_pair[1]}")
        else:
            print(f"  Warning: {pair_folder} has {len(fragments)} fragments (expected 2+)")
    
    print(f"Total unique positive pairs found: {len(positive_pairs)}")
    return positive_pairs

def load_point_cloud(file_path):
    """Load point cloud and extract basic geometric features."""
    try:
        if OPEN3D_AVAILABLE:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                print(f"Warning: Empty point cloud in {file_path}")
                return None
            
            # Extract geometric features
            points = np.asarray(pcd.points)
            if len(points) < 3:  # Need at least 3 points for meaningful features
                print(f"Warning: Too few points ({len(points)}) in {file_path}")
                return None
                
            centroid = np.mean(points, axis=0)
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            
            # Compute FPFH features with error handling
            fpfh_data = None
            try:
                # Estimate normals with adaptive parameters
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=min(bbox_size) * 0.05,  # Adaptive radius
                        max_nn=30
                    )
                )
                # Orient normals consistently
                pcd.orient_normals_consistent_tangent_plane(k=30)
                
                # Compute FPFH with adaptive radius
                fpfh_radius = min(bbox_size) * 0.1  # Adaptive radius
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=fpfh_radius,
                        max_nn=100
                    )
                )
                fpfh_data = np.asarray(fpfh.data)
                
                # Check for invalid FPFH features
                if np.any(np.isnan(fpfh_data)) or np.any(np.isinf(fpfh_data)):
                    print(f"Warning: Invalid FPFH features in {file_path}, using fallback")
                    fpfh_data = None
                
            except Exception as e:
                print(f"Warning: FPFH computation failed for {file_path}: {e}")
                fpfh_data = None
            
            return {
                'points': points,
                'centroid': centroid,
                'bbox_size': bbox_size,
                'fpfh': fpfh_data,
                'num_points': len(points)
            }
        else:
            # Fallback: basic geometric analysis without open3d
            # Simple PLY parser for coordinates
            points = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                vertex_count = 0
                reading_vertices = False
                
                for line in lines:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line.startswith('end_header'):
                        reading_vertices = True
                        continue
                    elif reading_vertices and vertex_count > 0:
                        coords = line.strip().split()[:3]
                        if len(coords) == 3:
                            try:
                                points.append([float(x) for x in coords])
                                vertex_count -= 1
                                if vertex_count == 0:
                                    break
                            except ValueError:
                                continue
            
            if not points or len(points) < 3:  # Need at least 3 points
                print(f"Warning: Insufficient points in {file_path}")
                return None
                
            points = np.array(points)
            centroid = np.mean(points, axis=0)
            bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
            
            return {
                'points': points,
                'centroid': centroid,
                'bbox_size': bbox_size,
                'fpfh': None,
                'num_points': len(points)
            }
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def compute_similarity_score(features1, features2):
    """Compute similarity score between two fragments."""
    if features1 is None or features2 is None:
        return 0.0
    
    # Geometric similarity
    centroid_dist = np.linalg.norm(features1['centroid'] - features2['centroid'])
    bbox_similarity = 1.0 / (1.0 + np.linalg.norm(features1['bbox_size'] - features2['bbox_size']))
    size_similarity = min(features1['num_points'], features2['num_points']) / max(features1['num_points'], features2['num_points'])
    
    # Feature similarity (if available)
    feature_similarity = 0.5
    if OPEN3D_AVAILABLE and features1['fpfh'] is not None and features2['fpfh'] is not None:
        try:
            # Compute average feature similarity
            fpfh1_mean = np.mean(features1['fpfh'], axis=1)
            fpfh2_mean = np.mean(features2['fpfh'], axis=1)
            feature_similarity = np.dot(fpfh1_mean, fpfh2_mean) / (np.linalg.norm(fpfh1_mean) * np.linalg.norm(fpfh2_mean))
            feature_similarity = max(0, feature_similarity)  # Ensure non-negative
        except:
            feature_similarity = 0.5
    
    # Combine similarities
    total_similarity = (bbox_similarity * 0.3 + size_similarity * 0.3 + feature_similarity * 0.4) / (1.0 + centroid_dist * 0.1)
    return total_similarity

def load_fragment_features(data_path: str, fragments: List[str]) -> Dict[str, dict]:
    """Load features for all fragments with progress tracking."""
    print(f"Loading features for {len(fragments)} fragments...")
    
    def load_single_feature(frag):
        return frag, load_point_cloud(os.path.join(data_path, frag))
    
    features = {}
    with ThreadPoolExecutor(max_workers=min(8, len(fragments))) as executor:
        results = list(executor.map(load_single_feature, fragments))
    
    for frag, feat in results:
        features[frag] = feat
    
    valid_features = sum(1 for f in features.values() if f is not None)
    print(f"Successfully loaded features for {valid_features}/{len(fragments)} fragments")
    return features

def generate_negative_pair(all_frags: List[str], positive_pairs: Set[Tuple[str, str]], 
                          fragment_features: Dict[str, dict], hard_negative_ratio: float = 0.5,
                          current_pos_pair: Optional[List[str]] = None):
    """Generate negative pair with option for hard negatives, avoiding all known positive pairs."""
    
    max_attempts = 100  # Prevent infinite loops
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        # Decide if this should be a hard negative
        use_hard_negative = (random.random() < hard_negative_ratio and 
                           fragment_features and 
                           current_pos_pair is not None and 
                           len(current_pos_pair) > 0)
        
        if use_hard_negative and current_pos_pair is not None:  # Extra safety check
            # Generate hard negative: find fragments similar to current positive pair
            pos_features = [fragment_features.get(frag) for frag in current_pos_pair 
                          if fragment_features.get(frag) is not None]
            
            if len(pos_features) >= 1:
                # Find fragments most similar to the positive fragments
                similarities = []
                for frag in all_frags:
                    frag_features = fragment_features.get(frag)
                    if frag_features is None:
                        continue
                    
                    # Compute max similarity to any positive fragment in current pair
                    max_sim = max(compute_similarity_score(pos_feat, frag_features) for pos_feat in pos_features)
                    similarities.append((frag, max_sim))
                
                if similarities:
                    # Sort by similarity and pick from top candidates
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = similarities[:max(1, len(similarities) // 3)]  # Top 33%
                    
                    # Try to select two different fragments from top candidates
                    for _ in range(min(20, len(top_candidates))):  # Limited attempts for hard negatives
                        frag_a = random.choice(top_candidates)[0]
                        remaining_candidates = [f for f, _ in top_candidates if f != frag_a]
                        
                        if remaining_candidates:
                            frag_b = random.choice(remaining_candidates)
                        else:
                            # Fallback to any different fragment
                            available_frags = [f for f in all_frags if f != frag_a]
                            if not available_frags:
                                break
                            frag_b = random.choice(available_frags)
                        
                        # Check if this pair is NOT a known positive pair
                        candidate_pair = normalize_pair(frag_a, frag_b)
                        if candidate_pair not in positive_pairs:
                            return frag_a, frag_b, True  # True indicates hard negative
        
        # Generate easy negative (random selection) or fallback from hard negative
        available_frags = list(all_frags)
        if len(available_frags) < 2:
            return None, None, False
            
        frag_a = random.choice(available_frags)
        frag_b = random.choice(available_frags)
        
        # Ensure different fragments
        retry_count = 0
        while frag_a == frag_b and retry_count < 20:
            frag_b = random.choice(available_frags)
            retry_count += 1
        
        if frag_a == frag_b:
            continue  # Try again with different approach
        
        # Check if this pair is NOT a known positive pair
        candidate_pair = normalize_pair(frag_a, frag_b)
        if candidate_pair not in positive_pairs:
            return frag_a, frag_b, False  # False indicates easy negative
    
    print(f"Warning: Could not generate valid negative pair after {max_attempts} attempts")
    return None, None, False

def check_for_existing_pairs(pairs_output: str, current_pair: List[str]) -> bool:
    """Check if a positive pair already exists in the output directory."""
    if not os.path.exists(pairs_output):
        return False
    
    normalized_current = normalize_pair(current_pair[0], current_pair[1])
    
    # Check all existing positive pair folders
    for folder in os.listdir(pairs_output):
        if folder.startswith('pos_pair_'):
            folder_path = os.path.join(pairs_output, folder)
            if os.path.isdir(folder_path):
                existing_frags = get_fragments(folder_path)
                if len(existing_frags) == 2:
                    normalized_existing = normalize_pair(existing_frags[0], existing_frags[1])
                    if normalized_current == normalized_existing:
                        return True
    return False

def process_single_pair(args):
    """Process a single positive pair and generate its negative pairs."""
    (pair_folder, testdata_path, data_path, pairs_output, 
     num_neg_per_pos, all_fragments, fragment_features, 
     positive_pairs, pair_idx, existing_pos_pairs) = args
    
    pair_folder_path = os.path.join(testdata_path, pair_folder)
    pos_frags = get_fragments(pair_folder_path)
    
    if len(pos_frags) != 2:
        return f"Skipped {pair_folder}: does not contain exactly 2 .ply fragments.", 0, 0
    
    # Check if this positive pair already exists in output
    if check_for_existing_pairs(pairs_output, pos_frags):
        return f"Skipped {pair_folder}: positive pair already exists in output.", 0, 0
    
    # Save positive pair
    pos_pair_folder = os.path.join(pairs_output, f'pos_pair_{pair_idx}')
    create_dir(pos_pair_folder)
    
    for frag in pos_frags:
        shutil.copy(os.path.join(pair_folder_path, frag), os.path.join(pos_pair_folder, frag))
    
    # Generate negative pairs
    neg_pairs_created = 0
    hard_neg_count = 0
    base_neg_counter = pair_idx * num_neg_per_pos
    
    for i in range(num_neg_per_pos):
        frag_a, frag_b, is_hard = generate_negative_pair(
            all_fragments, positive_pairs, fragment_features, 
            hard_negative_ratio=0.5, current_pos_pair=pos_frags
        )
        
        if frag_a is None or frag_b is None:
            print(f"Warning: Could not generate negative pair {i+1} for {pair_folder}")
            continue
        
        neg_counter = base_neg_counter + i + 1
        neg_pair_folder = os.path.join(pairs_output, f'neg_pair_{neg_counter}')
        create_dir(neg_pair_folder)
        
        try:
            shutil.copy(os.path.join(data_path, frag_a), os.path.join(neg_pair_folder, frag_a))
            shutil.copy(os.path.join(data_path, frag_b), os.path.join(neg_pair_folder, frag_b))
            neg_pairs_created += 1
            if is_hard:
                hard_neg_count += 1
        except FileNotFoundError as e:
            print(f"Warning: Could not copy fragments for negative pair: {e}")
            # Clean up the created directory
            if os.path.exists(neg_pair_folder):
                shutil.rmtree(neg_pair_folder)
    
    return f"Processed {pair_folder}: 1 pos, {neg_pairs_created} neg ({hard_neg_count} hard)", 1, neg_pairs_created

def main(dataset_root, neg_to_pos_ratio=3.0, hard_negative_ratio=0.5, max_workers=None, clear_output=True):
    """
    Main function with enhanced negative pair generation and parallelization.
    Args:
        dataset_root: Root directory of the dataset
        neg_to_pos_ratio: Desired ratio of negative to positive pairs (default: 3.0)
        hard_negative_ratio: Ratio of hard negatives (0.0 to 1.0, default: 0.5)
        max_workers: Maximum number of parallel workers (default: CPU count)
        clear_output: Whether to clear the output directory before generation (default: True)
    """
    start_time = time.time()
    
    testdata_path = os.path.join(dataset_root, 'testdata')
    data_path = os.path.join(dataset_root, 'data')
    pairs_output = os.path.join(dataset_root, 'pairs')
    
    # Handle output directory
    if clear_output and os.path.exists(pairs_output):
        print(f"Clearing existing output directory: {pairs_output}")
        shutil.rmtree(pairs_output)
        print("✓ Output directory cleared")
    
    create_dir(pairs_output)
    
    # FIRST: Collect all positive pairs from testdata
    positive_pairs = collect_all_positive_pairs(testdata_path)
    num_positives = len(positive_pairs)
    
    # Calculate negatives per positive
    num_neg_per_pos = max(1, int(round(neg_to_pos_ratio)))
    print(f"\nConfigured negative:positive ratio: {neg_to_pos_ratio}:1 (rounded to {num_neg_per_pos} negatives per positive)")
    
    # Collect all .ply fragments from data/
    print("\nCollecting fragments from data directory...")
    all_fragments = get_fragments(data_path)
    print(f"Found {len(all_fragments)} fragments in data directory")
    
    # Validate that positive pair fragments exist in data directory
    missing_fragments = set()
    for pair in positive_pairs:
        for frag in pair:
            if frag not in all_fragments:
                missing_fragments.add(frag)
    
    if missing_fragments:
        print(f"WARNING: The following fragments from positive pairs are missing in data directory:")
        for frag in missing_fragments:
            print(f"  - {frag}")
        print("These positive pairs will be skipped during processing.")
    
    # Load fragment features for hard negative mining
    fragment_features = {}
    if hard_negative_ratio > 0:
        fragment_features = load_fragment_features(data_path, all_fragments)
    
    # Get all pair folders
    pair_folders = [f for f in os.listdir(testdata_path) 
                   if f.startswith('pair_') and os.path.isdir(os.path.join(testdata_path, f))]
    
    print(f"\nFound {len(pair_folders)} pair folders to process")
    
    # Check for existing positive pairs in output
    existing_pos_pairs = set()
    if os.path.exists(pairs_output):
        for folder in os.listdir(pairs_output):
            if folder.startswith('pos_pair_'):
                folder_path = os.path.join(pairs_output, folder)
                if os.path.isdir(folder_path):
                    existing_frags = get_fragments(folder_path)
                    if len(existing_frags) == 2:
                        existing_pos_pairs.add(normalize_pair(existing_frags[0], existing_frags[1]))
    
    # Prepare arguments for parallel processing
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(pair_folders))
    
    args_list = []
    for i, pair_folder in enumerate(pair_folders, 1):
        args_list.append((
            pair_folder, testdata_path, data_path, pairs_output,
            num_neg_per_pos, all_fragments, fragment_features, 
            positive_pairs, i, existing_pos_pairs
        ))
    
    # Process pairs in parallel
    print(f"Processing pairs with {max_workers} workers...")
    total_pos_pairs = 0
    total_neg_pairs = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_pair, args_list))
    
    # Collect results
    for result_msg, pos_count, neg_count in results:
        print(result_msg)
        total_pos_pairs += pos_count
        total_neg_pairs += neg_count
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total unique positive pairs identified: {len(positive_pairs)}")
    print(f"Total positive pairs processed: {total_pos_pairs}")
    print(f"Total negative pairs generated: {total_neg_pairs}")
    print(f"Actual ratio (neg:pos): {total_neg_pairs/max(1, total_pos_pairs):.2f}:1")
    print(f"Hard negative ratio: {hard_negative_ratio:.1%}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Output directory: {pairs_output}")
    
    # Summary of positive pairs
    print(f"\nPositive pairs summary:")
    for i, pair in enumerate(sorted(positive_pairs), 1):
        print(f"  {i:2d}. {pair[0]} <-> {pair[1]}")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate fragment pairs with balanced positive:negative ratio.")
    parser.add_argument('--dataset_root', type=str, default="dataset_root", help="Root directory of the dataset")
    parser.add_argument('--neg_to_pos_ratio', type=float, default=3.0, help="Negative:positive ratio (e.g. 3.0 for 3:1)")
    parser.add_argument('--hard_negative_ratio', type=float, default=0.5, help="Fraction of negatives that are hard negatives")
    parser.add_argument('--max_workers', type=int, default=None, help="Number of parallel workers")
    parser.add_argument('--clear_output', action='store_true', help="Clear output directory before generation")
    args = parser.parse_args()
    main(
        args.dataset_root,
        neg_to_pos_ratio=args.neg_to_pos_ratio,
        hard_negative_ratio=args.hard_negative_ratio,
        max_workers=args.max_workers,
        clear_output=args.clear_output
    )