import os
import json
import numpy as np
from tqdm import tqdm
from feature_extractor import FeatureExtractor
from itertools import islice
from sklearn.neighbors import NearestNeighbors

def get_point_cloud_files(pair_dir):
    """Get point cloud files from pair directory"""
    return [os.path.join(pair_dir, f) for f in os.listdir(pair_dir) if f.endswith('.ply') or f.endswith('.pcd')]

def extract_edge_features(points, normals, curvatures, local_density):
    """Extract features for each edge segment"""
    edges = []
    for i in range(len(points)):
        # Connect to next point (wrapping around to start)
        next_idx = (i + 1) % len(points)
        
        # Basic geometric features
        start_point = points[i]
        end_point = points[next_idx]
        edge_vector = end_point - start_point
        length = float(np.linalg.norm(edge_vector))
        direction = edge_vector / length if length > 0 else np.zeros(3)
        midpoint = (start_point + end_point) / 2
        
        # Normal-based features
        if normals is not None:
            start_normal = normals[i]
            end_normal = normals[next_idx]
            normal_angle = np.arccos(np.clip(np.dot(start_normal, end_normal), -1, 1))
            avg_normal = (start_normal + end_normal) / 2
            normal_alignment = np.abs(np.dot(direction, avg_normal))
        else:
            normal_angle = 0
            normal_alignment = 0
        
        # Curvature and density features
        avg_curvature = (curvatures[i] + curvatures[next_idx]) / 2
        avg_density = (local_density[i] + local_density[next_idx]) / 2
        
        edges.append({
            'indices': (i, next_idx),
            'length': length,
            'direction': direction,
            'midpoint': midpoint,
            'normal_angle': float(normal_angle),
            'normal_alignment': float(normal_alignment),
            'curvature': float(avg_curvature),
            'local_density': float(avg_density)
        })
    return edges

def filter_edge_pairs(edges1, edges2, length_tolerance=0.2, direction_tolerance=0.7, curvature_tolerance=0.3):
    """Filter edge pairs based on multiple criteria"""
    compatible_pairs = []
    
    # First, group edges by length for faster filtering
    edges2_by_length = {}
    for idx2, edge2 in enumerate(edges2):
        length2 = edge2['length']
        length_bin = int(length2 * 10)  # Adjust bin size if needed
        if length_bin not in edges2_by_length:
            edges2_by_length[length_bin] = []
        edges2_by_length[length_bin].append((idx2, edge2))
    
    # For each edge in first fragment, find compatible edges in second fragment
    for idx1, edge1 in enumerate(edges1):
        length1 = edge1['length']
        direction1 = edge1['direction']
        curvature1 = edge1['curvature']
        
        # Check edges of similar length
        length_bin = int(length1 * 10)
        for nearby_bin in [length_bin-1, length_bin, length_bin+1]:
            if nearby_bin not in edges2_by_length:
                continue
            
            for idx2, edge2 in edges2_by_length[nearby_bin]:
                # Check multiple criteria
                length2 = edge2['length']
                direction2 = edge2['direction']
                curvature2 = edge2['curvature']
                
                # Length similarity
                if abs(length1 - length2) / max(length1, length2) > length_tolerance:
                    continue
                
                # Direction alignment
                alignment = abs(np.dot(direction1, direction2))
                if alignment < direction_tolerance:
                    continue
                
                # Curvature similarity
                if abs(curvature1 - curvature2) > curvature_tolerance:
                    continue
                
                # Check normal alignment if available
                normal_score = min(edge1['normal_alignment'], edge2['normal_alignment'])
                if normal_score < 0.5:  # Skip if normals are not well-aligned with edge direction
                    continue
                
                compatible_pairs.append((idx1, idx2))
    
    return compatible_pairs

def compute_edge_features_dict(edge):
    """Convert edge features to a dictionary for JSON storage"""
    return {
        'length': float(edge['length']),
        'midpoint': edge['midpoint'].tolist(),
        'direction': edge['direction'].tolist(),
        'normal_angle': edge['normal_angle'],
        'normal_alignment': edge['normal_alignment'],
        'curvature': edge['curvature'],
        'local_density': edge['local_density']
    }

def process_pair(pair_dir):
    point_cloud_files = get_point_cloud_files(pair_dir)
    if len(point_cloud_files) != 2:
        print(f"[WARN] {pair_dir}: Expected 2 point cloud files, found {len(point_cloud_files)}")
        return
    print(f"[DEBUG] Processing {pair_dir} with files: {[os.path.basename(f) for f in point_cloud_files]}")
    
    fe = FeatureExtractor()
    boundaries = []
    
    for pf in point_cloud_files:
        info = fe.extract_point_cloud_boundary(pf)
        if info is None:
            print(f"[WARN] Failed to extract boundary for {pf}")
            return
        boundaries.append(info)
        print(f"[DEBUG] {os.path.basename(pf)}: Found {len(info['boundary_indices'])} boundary points")
    
    # Extract edges with features for each fragment
    edges1 = extract_edge_features(
        boundaries[0]['boundary_coords'],
        boundaries[0]['boundary_normals'],
        boundaries[0]['curvature'],
        boundaries[0]['local_density']
    )
    edges2 = extract_edge_features(
        boundaries[1]['boundary_coords'],
        boundaries[1]['boundary_normals'],
        boundaries[1]['curvature'],
        boundaries[1]['local_density']
    )
    print(f"[DEBUG] Found {len(edges1)} edges in first fragment and {len(edges2)} edges in second fragment")
    
    # Filter edge pairs based on compatibility
    compatible_pairs = filter_edge_pairs(edges1, edges2)
    print(f"[DEBUG] Found {len(compatible_pairs)} compatible edge pairs after filtering")
    
    # Generate candidates only for compatible pairs
    candidates = []
    for idx1, idx2 in tqdm(compatible_pairs, desc="Processing compatible edge pairs"):
        edge1 = edges1[idx1]
        edge2 = edges2[idx2]
        candidates.append({
            'edge1_idx': idx1,
            'edge2_idx': idx2,
            'edge1_features': compute_edge_features_dict(edge1),
            'edge2_features': compute_edge_features_dict(edge2),
            'label': 0  # Default label
        })
    
    # Save results
    out_path = os.path.join(pair_dir, f"{os.path.basename(pair_dir)}boundary.json")
    with open(out_path, 'w') as f:
        json.dump(candidates, f, indent=2)
    print(f"[INFO] {pair_dir}: Saved {len(candidates)} candidate edge pairs to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate boundary matching candidates for fragment pairs")
    parser.add_argument('--base_dir', type=str, default='dataset_root/testdata',
                      help="Base directory containing pair_X directories")
    parser.add_argument('--pair', type=int, help="Specific pair to process (optional)")
    args = parser.parse_args()
    
    if args.pair is not None:
        # Process specific pair
        pair_dir = os.path.join(args.base_dir, f'pair_{args.pair}')
        if not os.path.exists(pair_dir):
            print(f"Error: Directory not found: {pair_dir}")
            exit(1)
        process_pair(pair_dir)
    else:
        # Process all pairs
        for entry in os.listdir(args.base_dir):
            if entry.startswith('pair_'):
                pair_dir = os.path.join(args.base_dir, entry)
                if os.path.isdir(pair_dir):
                    process_pair(pair_dir) 