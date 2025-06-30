import os
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def load_point_clouds(pair_dir):
    """Load point cloud files from pair directory"""
    files = [f for f in os.listdir(pair_dir) if f.endswith('.ply') or f.endswith('.pcd')]
    if len(files) != 2:
        raise ValueError(f"Expected 2 point cloud files in {pair_dir}, found {len(files)}")
    
    pcds = []
    for f in files:
        path = os.path.join(pair_dir, f)
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            raise ValueError(f"Empty point cloud in {path}")
        pcds.append(pcd)
    
    return pcds

def find_break_surface(pcd1, pcd2, distance_threshold=2.0, min_neighbors=3):
    """Find points that are likely part of the break surface based on proximity to other fragment"""
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)
    
    # For each point in fragment 1, find nearest neighbors in fragment 2
    nbrs = NearestNeighbors(n_neighbors=min_neighbors, algorithm='auto')
    nbrs.fit(points2)
    distances1, _ = nbrs.kneighbors(points1)
    
    # Points are part of break surface if they have enough close neighbors
    avg_distances1 = np.mean(distances1, axis=1)
    break_mask1 = avg_distances1 < distance_threshold
    
    # Repeat for fragment 2
    nbrs.fit(points1)
    distances2, _ = nbrs.kneighbors(points2)
    avg_distances2 = np.mean(distances2, axis=1)
    break_mask2 = avg_distances2 < distance_threshold
    
    return break_mask1, break_mask2

def estimate_normals(pcd, radius_multiplier=2):
    """Estimate normals with adaptive radius"""
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_extent()
    radius = min(bbox_size) * radius_multiplier
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

def expand_break_surface(points, break_mask, radius=5.0):
    """Expand break surface region by including nearby points"""
    # Get points that are already part of break surface
    break_points = points[break_mask]
    
    # Find points within radius of break surface points
    nbrs = NearestNeighbors(radius=radius, algorithm='auto')
    nbrs.fit(points)
    indices = nbrs.radius_neighbors(break_points, return_distance=False)
    
    # Create expanded mask
    expanded_mask = np.zeros(len(points), dtype=bool)
    for idx_set in indices:
        expanded_mask[idx_set] = True
    
    return expanded_mask

def refine_break_surface(pcd, break_mask, normal_angle_threshold=np.pi/3):
    """Refine break surface using normal consistency"""
    normals = np.asarray(pcd.normals)
    points = np.asarray(pcd.points)
    
    # Find points with similar normals in local neighborhoods
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto')
    nbrs.fit(points[break_mask])
    
    refined_mask = break_mask.copy()
    break_points = points[break_mask]
    break_normals = normals[break_mask]
    
    for i, (point, normal) in enumerate(zip(break_points, break_normals)):
        # Get local neighborhood
        distances, indices = nbrs.kneighbors([point])
        neighbor_normals = break_normals[indices[0]]
        
        # Check normal consistency
        angles = np.arccos(np.clip(np.dot(neighbor_normals, normal), -1, 1))
        if np.mean(angles) > normal_angle_threshold:
            refined_mask[np.where(break_mask)[0][i]] = False
    
    return refined_mask

def create_color_gradient(num_points, base_color):
    """Create a gradient of colors centered around the base color"""
    gradient = np.zeros((num_points, 3))
    for i in range(3):  # RGB channels
        if base_color[i] > 0:  # If this channel is active in base color
            # Create gradient from darker to brighter
            gradient[:, i] = np.linspace(base_color[i] * 0.5, min(1.0, base_color[i] * 1.5), num_points)
        else:
            # Add a small variation to non-primary channels
            gradient[:, i] = np.linspace(0, 0.2, num_points)
    return gradient

def visualize_fragments_with_break_surface(pcd1, pcd2, break_mask1, break_mask2):
    """Create visualization of fragments with highlighted break surfaces"""
    # Create copies for visualization
    vis_pcd1 = o3d.geometry.PointCloud(pcd1)
    vis_pcd2 = o3d.geometry.PointCloud(pcd2)
    
    # Get points
    points1 = np.asarray(vis_pcd1.points)
    points2 = np.asarray(vis_pcd2.points)
    
    # Expand break surface regions
    expanded_mask1 = expand_break_surface(points1, break_mask1)
    expanded_mask2 = expand_break_surface(points2, break_mask2)
    
    # Set default colors (grey)
    colors1 = np.tile(np.array([0.7, 0.7, 0.7]), (len(points1), 1))
    colors2 = np.tile(np.array([0.7, 0.7, 0.7]), (len(points2), 1))
    
    # Create color gradients for break surfaces
    gradient1 = create_color_gradient(np.sum(expanded_mask1), [1, 0.3, 0.3])  # Reddish
    gradient2 = create_color_gradient(np.sum(expanded_mask2), [0.3, 1, 0.3])  # Greenish
    
    # Apply gradients to break surfaces
    colors1[expanded_mask1] = gradient1
    colors2[expanded_mask2] = gradient2
    
    # Highlight original break surface points more intensely
    colors1[break_mask1] = [1, 0, 0]  # Pure red
    colors2[break_mask2] = [0, 1, 0]  # Pure green
    
    vis_pcd1.colors = o3d.utility.Vector3dVector(colors1)
    vis_pcd2.colors = o3d.utility.Vector3dVector(colors2)
    
    # Create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Break Surface Visualization")
    
    # Add point clouds
    vis.add_geometry(vis_pcd1)
    vis.add_geometry(vis_pcd2)
    
    # Set better viewing options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 3.0  # Larger points
    
    # Add key callback to quit
    def close_callback(vis):
        vis.close()
        return False
    
    vis.register_key_callback(ord("Q"), close_callback)
    
    print("\nVisualization controls:")
    print("- Left click + drag: Rotate")
    print("- Right click + drag: Pan")
    print("- Mouse wheel: Zoom")
    print("- Q: Quit")
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize break surfaces between fragment pairs")
    parser.add_argument('--base_dir', type=str, default='dataset_root/testdata',
                      help="Base directory containing pair_X directories")
    parser.add_argument('--pair', type=int, required=True,
                      help="Pair number to visualize (required)")
    parser.add_argument('--distance', type=float, default=2.0,
                      help="Distance threshold for break surface detection (default: 2.0)")
    parser.add_argument('--neighbors', type=int, default=3,
                      help="Minimum number of close neighbors required (default: 3)")
    parser.add_argument('--radius', type=float, default=5.0,
                      help="Radius for expanding break surface region (default: 5.0)")
    args = parser.parse_args()
    
    # Construct pair directory path
    pair_dir = os.path.join(args.base_dir, f'pair_{args.pair}')
    if not os.path.exists(pair_dir):
        print(f"Error: Directory not found: {pair_dir}")
        return
    
    try:
        # Load point clouds
        print(f"Loading point clouds from {pair_dir}...")
        pcd1, pcd2 = load_point_clouds(pair_dir)
        
        # Estimate normals for both fragments
        print("Estimating normals...")
        estimate_normals(pcd1)
        estimate_normals(pcd2)
        
        # Find initial break surfaces
        print("Finding break surfaces...")
        break_mask1, break_mask2 = find_break_surface(
            pcd1, pcd2, 
            distance_threshold=args.distance,
            min_neighbors=args.neighbors
        )
        
        # Refine break surfaces using normal consistency
        print("Refining break surfaces...")
        refined_mask1 = refine_break_surface(pcd1, break_mask1)
        refined_mask2 = refine_break_surface(pcd2, break_mask2)
        
        # Print statistics
        print(f"\nFragment 1: Found {np.sum(refined_mask1)} break surface points")
        print(f"Fragment 2: Found {np.sum(refined_mask2)} break surface points")
        
        # Visualize results
        print("\nStarting visualization...")
        visualize_fragments_with_break_surface(pcd1, pcd2, refined_mask1, refined_mask2)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 