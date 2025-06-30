import os
import json
import numpy as np
from feature_extractor import FeatureExtractor

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    import matplotlib.pyplot as plt

# --- Utility Functions ---
def get_point_cloud_files(pair_dir):
    return [os.path.join(pair_dir, f) for f in os.listdir(pair_dir) if f.endswith('.ply') or f.endswith('.pcd')]

def extract_edges(points):
    """Return list of (start, end) index pairs for consecutive boundary points"""
    edges = []
    for i in range(len(points)):
        next_idx = (i + 1) % len(points)
        edges.append((i, next_idx))
    return edges

def edge_length(coords, edge):
    return np.linalg.norm(coords[edge[0]] - coords[edge[1]])

def compute_edge_midpoint(coords, edge_idx1, edge_idx2):
    """Compute midpoint of an edge given by two vertex indices"""
    pt1 = coords[edge_idx1]
    pt2 = coords[edge_idx2]
    return (pt1 + pt2) / 2

def are_edges_matching(coords1, edge1, coords2, edge2, normals1=None, normals2=None, 
                      curvatures1=None, curvatures2=None, distance_threshold=2.0, 
                      angle_threshold=0.7, curvature_threshold=0.3):
    """Check if two edges are matching based on multiple criteria"""
    # Get edge midpoints
    mid1 = compute_edge_midpoint(coords1, edge1[0], edge1[1])
    mid2 = compute_edge_midpoint(coords2, edge2[0], edge2[1])
    
    # Check distance between midpoints
    distance = np.linalg.norm(mid1 - mid2)
    if distance > distance_threshold:
        return False
        
    # Get edge vectors
    vec1 = coords1[edge1[1]] - coords1[edge1[0]]
    vec2 = coords2[edge2[1]] - coords2[edge2[0]]
    
    # Normalize vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Check if vectors are roughly parallel or antiparallel
    alignment = abs(np.dot(vec1, vec2))
    if alignment < angle_threshold:
        return False
    
    # Check normal alignment if available
    if normals1 is not None and normals2 is not None:
        normal1 = (normals1[edge1[0]] + normals1[edge1[1]]) / 2
        normal2 = (normals2[edge2[0]] + normals2[edge2[1]]) / 2
        normal_alignment = abs(np.dot(normal1, normal2))
        if normal_alignment < angle_threshold:
            return False
    
    # Check curvature similarity if available
    if curvatures1 is not None and curvatures2 is not None:
        curv1 = (curvatures1[edge1[0]] + curvatures1[edge1[1]]) / 2
        curv2 = (curvatures2[edge2[0]] + curvatures2[edge2[1]]) / 2
        if abs(curv1 - curv2) > curvature_threshold:
            return False
    
    return True

def list_available_pairs(base_dir):
    pairs = []
    for i in range(1, 18):  # Pairs 1 through 17
        pair_dir = os.path.join(base_dir, f'pair_{i}')
        if os.path.exists(pair_dir):
            pairs.append(i)
    return pairs

def find_close_points(coords1, coords2, distance_threshold=30.0):
    """Find points that are close to points in the other fragment"""
    close_points1 = set()
    close_points2 = set()
    
    # For each point in fragment 1, find close points in fragment 2
    for i, p1 in enumerate(coords1):
        for j, p2 in enumerate(coords2):
            if np.linalg.norm(p1 - p2) < distance_threshold:
                close_points1.add(i)
                close_points2.add(j)
    
    return list(close_points1), list(close_points2)

def visualize_boundaries(vis, pcd1, pcd2, coords1, coords2, highlight_edge1=None, highlight_edge2=None):
    """Visualize point clouds with their boundaries and optionally highlight specific edges"""
    # Add point clouds with colors
    pcd1.paint_uniform_color([0.7, 0.7, 0.7])  # Gray
    pcd2.paint_uniform_color([0.3, 0.3, 1.0])  # Blue
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    
    # Find points that are close to the other fragment
    close_points1, close_points2 = find_close_points(coords1, coords2)
    
    # Create line sets for boundaries near potential joins
    if close_points1:
        close_lines1 = []
        for i in range(len(close_points1)):
            next_i = (i + 1) % len(close_points1)
            if abs(close_points1[i] - close_points1[next_i]) <= 2:  # Only connect if indices are sequential
                close_lines1.append([close_points1[i], close_points1[next_i]])
        
        if close_lines1:
            boundary1_lines = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(coords1),
                lines=o3d.utility.Vector2iVector(close_lines1)
            )
            boundary1_lines.paint_uniform_color([1, 0.4, 0.4])  # Stronger red
            vis.add_geometry(boundary1_lines)
    
    if close_points2:
        close_lines2 = []
        for i in range(len(close_points2)):
            next_i = (i + 1) % len(close_points2)
            if abs(close_points2[i] - close_points2[next_i]) <= 2:  # Only connect if indices are sequential
                close_lines2.append([close_points2[i], close_points2[next_i]])
        
        if close_lines2:
            boundary2_lines = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(coords2),
                lines=o3d.utility.Vector2iVector(close_lines2)
            )
            boundary2_lines.paint_uniform_color([0.4, 1, 0.4])  # Stronger green
            vis.add_geometry(boundary2_lines)
    
    # Highlight specific edges if provided
    if highlight_edge1 is not None:
        pts = coords1[[highlight_edge1[0], highlight_edge1[1]]]
        highlight1 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        highlight1.paint_uniform_color([1, 0, 0])  # Pure red
        vis.add_geometry(highlight1)
    
    if highlight_edge2 is not None:
        pts = coords2[[highlight_edge2[0], highlight_edge2[1]]]
        highlight2 = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        )
        highlight2.paint_uniform_color([0, 1, 0])  # Pure green
        vis.add_geometry(highlight2)
    
    # Set better viewing options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 3.0  # Larger points
    opt.line_width = 5.0  # Thicker lines

# --- Main Labelling Tool ---
def auto_label_pair(pair_dir, json_path, distance_threshold=2.0, angle_threshold=0.5, curvature_threshold=0.3):
    point_cloud_files = get_point_cloud_files(pair_dir)
    if len(point_cloud_files) != 2:
        print(f"Expected 2 point cloud files in {pair_dir}, found {len(point_cloud_files)}")
        return
        
    print(f"\nProcessing pair in {os.path.basename(pair_dir)}...")
    
    fe = FeatureExtractor()
    boundaries = []
    point_clouds = []
    
    for pf in point_cloud_files:
        info = fe.extract_point_cloud_boundary(pf)
        if info is None:
            print(f"Failed to extract boundary for {pf}")
            return
        boundaries.append(info)
        point_clouds.append(o3d.io.read_point_cloud(pf))
    
    # Load candidate pairs from JSON
    with open(json_path, 'r') as f:
        candidates = json.load(f)
    print(f"Loaded {len(candidates)} candidate edge pairs.")
    
    # Extract edges using sequential indices
    edges1 = extract_edges(boundaries[0]['boundary_coords'])
    edges2 = extract_edges(boundaries[1]['boundary_coords'])
    
    print(f"Created {len(edges1)} edges for fragment 1")
    print(f"Created {len(edges2)} edges for fragment 2")
    
    # Count matches found
    matches_found = 0
    
    # Check each candidate
    for idx, candidate in enumerate(candidates):
        # Make sure edge indices are within bounds
        edge1_idx = candidate['edge1_idx'] % len(edges1)
        edge2_idx = candidate['edge2_idx'] % len(edges2)
        edge1 = edges1[edge1_idx]
        edge2 = edges2[edge2_idx]
        
        # Check if edges match based on multiple criteria
        is_match = are_edges_matching(
            boundaries[0]['boundary_coords'], edge1,
            boundaries[1]['boundary_coords'], edge2,
            boundaries[0]['boundary_normals'],
            boundaries[1]['boundary_normals'],
            boundaries[0]['curvature'],
            boundaries[1]['curvature'],
            distance_threshold=distance_threshold,
            angle_threshold=angle_threshold,
            curvature_threshold=curvature_threshold
        )
        
        # Update label
        candidates[idx]['label'] = 1 if is_match else 0
        if is_match:
            matches_found += 1
            
            # Visualize matching pairs if Open3D is available
            if OPEN3D_AVAILABLE:
                vis = o3d.visualization.VisualizerWithKeyCallback()
                vis.create_window(window_name=f"Matching Pair Found - {matches_found}")
                
                # Visualize boundaries and highlight matching edges
                visualize_boundaries(
                    vis, point_clouds[0], point_clouds[1],
                    boundaries[0]['boundary_coords'],
                    boundaries[1]['boundary_coords'],
                    edge1, edge2
                )
                
                # Let user verify the match
                vis.run()
                vis.destroy_window()
                
                # Ask for confirmation
                while True:
                    confirm = input(f"Is this a correct match? (y/n/q to quit): ").strip().lower()
                    if confirm in ['y', 'n', 'q']:
                        break
                
                if confirm == 'q':
                    print("Labeling aborted.")
                    return
                elif confirm == 'n':
                    candidates[idx]['label'] = 0
                    matches_found -= 1
    
    # Save results
    with open(json_path, 'w') as f:
        json.dump(candidates, f, indent=2)
    print(f"\nFound {matches_found} matching edge pairs.")
    print(f"Results saved to {json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automatically label true join regions for aligned fragment pairs.")
    parser.add_argument('--base_dir', type=str, default='dataset_root/testdata', help="Base directory containing pair_X directories")
    parser.add_argument('--pair', type=int, help="Pair number to label (1-17)")
    parser.add_argument('--threshold', type=float, default=10.0, help="Distance threshold for matching edges (in same units as point cloud)")
    parser.add_argument('--angle', type=float, default=0.5, help="Angle threshold for matching edges (cosine similarity, default 0.5 ≈ 60°)")
    parser.add_argument('--curvature', type=float, default=0.3, help="Curvature similarity threshold (default 0.3)")
    args = parser.parse_args()

    # List available pairs if no pair specified
    if args.pair is None:
        available_pairs = list_available_pairs(args.base_dir)
        print(f"Available pairs: {available_pairs}")
        print("Usage: python labelling.py --pair N  (where N is the pair number)")
        print("Optional: --threshold T  (where T is the distance threshold, default 10.0)")
        print("Optional: --angle A  (where A is the angle threshold, default 0.5)")
        print("Optional: --curvature C  (where C is the curvature threshold, default 0.3)")
        exit(0)

    # Validate pair number
    if not 1 <= args.pair <= 17:
        print("Error: Pair number must be between 1 and 17")
        exit(1)

    # Construct paths
    pair_dir = os.path.join(args.base_dir, f'pair_{args.pair}')
    json_path = os.path.join(pair_dir, f'pair_{args.pair}boundary.json')

    # Validate paths
    if not os.path.exists(pair_dir):
        print(f"Error: Directory not found: {pair_dir}")
        exit(1)
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        exit(1)

    # Run auto-labelling tool
    auto_label_pair(pair_dir, json_path, args.threshold, args.angle, args.curvature) 