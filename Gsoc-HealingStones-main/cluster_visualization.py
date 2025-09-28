#!/usr/bin/env python3
"""
Open3D Cluster Viewer - Interactive 3D visualization of clusters on break surface
"""

import numpy as np
import open3d as o3d
import pickle
from pathlib import Path
import argparse
import random

def load_fragment_data(fragment_name, output_dir="output", data_dir="Ground_Truth/artifact_1"):
    """Load data for a single fragment."""
    print(f"Loading data for {fragment_name}...")
    
    # Load point cloud
    ply_file = Path(data_dir) / f"{fragment_name}.ply"
    if not ply_file.exists():
        ply_file = Path(f"Ground_Truth/reconstructed/artifact_1/{fragment_name}.ply")
    
    if not ply_file.exists():
        raise FileNotFoundError(f"Point cloud not found: {fragment_name}.ply")
    
    pcd = o3d.io.read_point_cloud(str(ply_file))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    print(f"Loaded {len(points):,} points")
    
    # Load cluster data
    output_path = Path(output_dir)
    
    # Try hierarchical format first
    try:
        with open(output_path / "feature_clusters.pkl", 'rb') as f:
            cluster_data = pickle.load(f)
        if fragment_name in cluster_data:
            fragment_clusters = cluster_data[fragment_name]
            print("Loaded hierarchical cluster data")
        else:
            raise KeyError(f"Fragment {fragment_name} not found in cluster data")
    except:
        # Try flat format
        try:
            with open(output_path / "feature_clusters_flat.pkl", 'rb') as f:
                flat_data = pickle.load(f)
            
            # Convert to hierarchical for this fragment
            fragment_clusters = {"1k": [], "5k": [], "10k": []}
            
            clusters = flat_data.get('clusters', flat_data) if isinstance(flat_data, dict) else flat_data
            
            for cluster in clusters:
                if isinstance(cluster, dict):
                    frag = cluster.get('fragment', '')
                    scale = cluster.get('scale', '1k')
                else:
                    frag = getattr(cluster, 'fragment_name', '')
                    scale = getattr(cluster, 'scale_name', '1k')
                
                if frag == fragment_name:
                    fragment_clusters[scale].append(cluster)
            
            print("Loaded flat cluster data and converted")
        except Exception as e:
            print(f"Failed to load cluster data: {e}")
            return None, None, None
    
    # Load segmentation data
    try:
        with open(output_path / "segmented_fragments.pkl", 'rb') as f:
            seg_data = pickle.load(f)
        
        if fragment_name in seg_data:
            break_indices = np.array(seg_data[fragment_name]['surface_patches']['break_0'])
            original_indices = np.array(seg_data[fragment_name]['surface_patches']['original'])
            print(f"Found {len(break_indices):,} break surface points")
        else:
            # Create break surface from green color
            if colors is not None:
                green_mask = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.4) & (colors[:, 2] < 0.4)
                break_indices = np.where(green_mask)[0]
                original_indices = np.where(~green_mask)[0]
                print(f"Created {len(break_indices):,} break surface points from green color")
            else:
                break_indices = np.arange(len(points) // 3)  # First 1/3 as break surface
                original_indices = np.arange(len(points) // 3, len(points))
                print(f"Using first {len(break_indices):,} points as break surface")
    
    except Exception as e:
        print(f"No segmentation data found: {e}")
        # Create break surface from green color or use first third
        if colors is not None:
            green_mask = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.4) & (colors[:, 2] < 0.4)
            break_indices = np.where(green_mask)[0]
            original_indices = np.where(~green_mask)[0]
            print(f"Created {len(break_indices):,} break surface points from green color")
        else:
            break_indices = np.arange(len(points) // 3)
            original_indices = np.arange(len(points) // 3, len(points))
            print(f"Using first {len(break_indices):,} points as break surface")
    
    return points, break_indices, original_indices, fragment_clusters

def generate_distinct_colors(n_colors):
    """Generate n distinct colors."""
    if n_colors <= 20:
        # Use predefined colors for better distinction
        colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.0, 0.5, 0.0],  # Dark Green
            [0.5, 0.5, 0.0],  # Olive
            [1.0, 0.0, 0.5],  # Pink
            [0.0, 0.5, 1.0],  # Light Blue
            [0.5, 1.0, 0.0],  # Lime
            [1.0, 0.5, 0.5],  # Light Red
            [0.5, 0.5, 1.0],  # Light Purple
            [0.5, 1.0, 0.5],  # Light Green
            [1.0, 1.0, 0.5],  # Light Yellow
            [0.5, 0.5, 0.5],  # Gray
            [0.8, 0.2, 0.2],  # Dark Red
            [0.2, 0.8, 0.2],  # Bright Green
        ]
        return colors[:n_colors]
    else:
        # Generate random colors for many clusters
        random.seed(42)  # For reproducible colors
        colors = []
        for _ in range(n_colors):
            colors.append([random.random(), random.random(), random.random()])
        return colors

def create_cluster_visualization(points, break_indices, original_indices, fragment_clusters, 
                               fragment_name, scale="1k", show_original=True, show_centers=True):
    """Create Open3D visualization of clusters."""
    
    print(f"\nCreating visualization for {fragment_name} at scale {scale}")
    
    # Get clusters for the specified scale
    clusters = fragment_clusters.get(scale, [])
    print(f"Displaying {len(clusters)} clusters")
    
    if not clusters:
        print(f"No clusters found for scale {scale}")
        print(f"Available scales: {list(fragment_clusters.keys())}")
        return None
    
    # Create list of geometries to display
    geometries = []
    
    # 1. Show original surface in light gray (optional)
    if show_original and len(original_indices) > 0:
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(points[original_indices])
        original_colors = np.full((len(original_indices), 3), [0.8, 0.8, 0.8])  # Light gray
        original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
        geometries.append(original_pcd)
        print(f"Added {len(original_indices):,} original surface points")
    
    # 2. Create break surface point cloud with cluster colors
    break_pcd = o3d.geometry.PointCloud()
    break_points = points[break_indices]
    break_pcd.points = o3d.utility.Vector3dVector(break_points)
    
    # Color break points by cluster assignment
    point_colors = np.full((len(break_indices), 3), [0.5, 0.5, 0.5])  # Default gray
    
    # Generate distinct colors for clusters
    cluster_colors = generate_distinct_colors(len(clusters))
    cluster_centers = []
    
    # Color each cluster
    for i, cluster in enumerate(clusters):
        # Handle both dict and object formats
        if isinstance(cluster, dict):
            cluster_id = cluster.get('cluster_id', f'cluster_{i}')
            barycenter = np.array(cluster.get('barycenter', [0, 0, 0]))
            point_indices = cluster.get('point_indices', [])
        else:
            cluster_id = getattr(cluster, 'cluster_id', f'cluster_{i}')
            barycenter = np.array(getattr(cluster, 'barycenter', [0, 0, 0]))
            point_indices = getattr(cluster, 'point_indices', [])
        
        # Use predefined color
        color = cluster_colors[i]
        
        # Color the points belonging to this cluster
        assigned_count = 0
        for pt_idx in point_indices:
            if 0 <= pt_idx < len(point_colors):
                point_colors[pt_idx] = color
                assigned_count += 1
        
        cluster_centers.append(barycenter)
        print(f"  {cluster_id}: {assigned_count} points assigned, center at [{barycenter[0]:.1f}, {barycenter[1]:.1f}, {barycenter[2]:.1f}]")
    
    # Set colors for break surface
    break_pcd.colors = o3d.utility.Vector3dVector(point_colors)
    geometries.append(break_pcd)
    
    # 3. Create cluster centers as spheres (optional)
    if show_centers and cluster_centers:
        for i, center in enumerate(cluster_centers):
            # Create small sphere at cluster center
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)  # 2mm radius
            sphere.translate(center)
            sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
            geometries.append(sphere)
        
        print(f"Added {len(cluster_centers)} cluster center markers")
    
    return geometries

def visualize_clusters_open3d(points, break_indices, original_indices, fragment_clusters, 
                            fragment_name, scale="1k", show_original=True, show_centers=True):
    """Launch Open3D visualization."""
    
    # Create visualization
    geometries = create_cluster_visualization(
        points, break_indices, original_indices, fragment_clusters,
        fragment_name, scale, show_original, show_centers
    )
    
    if not geometries:
        print("No geometries to display!")
        return
    
    # Configure visualization
    print(f"\nLaunching Open3D viewer...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Mouse wheel: Zoom")
    print("  - Shift + Mouse: Pan")
    print("  - R: Reset view")
    print("  - Q or ESC: Quit")
    
    # Launch viewer
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"{fragment_name} - {scale.upper()} Clusters",
        width=1200,
        height=800,
        left=50,
        top=50
    )

def print_cluster_info(fragment_clusters, fragment_name):
    """Print basic cluster information."""
    print(f"\n=== Cluster Summary for {fragment_name} ===")
    
    total_clusters = 0
    for scale, clusters in fragment_clusters.items():
        n_clusters = len(clusters)
        total_clusters += n_clusters
        print(f"{scale.upper()}: {n_clusters} clusters")
        
        if n_clusters > 0:
            # Calculate average points per cluster
            total_points = 0
            point_counts = []
            for cluster in clusters:
                if isinstance(cluster, dict):
                    points = len(cluster.get('point_indices', []))
                else:
                    points = len(getattr(cluster, 'point_indices', []))
                total_points += points
                point_counts.append(points)
            
            avg_points = total_points / n_clusters if n_clusters > 0 else 0
            min_points = min(point_counts) if point_counts else 0
            max_points = max(point_counts) if point_counts else 0
            
            print(f"     Average points per cluster: {avg_points:.0f}")
            print(f"     Range: {min_points} - {max_points} points")
    
    print(f"Total clusters: {total_clusters}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Open3D Cluster Viewer")
    parser.add_argument("fragment", help="Fragment name (e.g., frag_1)")
    parser.add_argument("--scale", choices=["1k", "5k", "10k"], default="1k",
                       help="Cluster scale to display")
    parser.add_argument("--output_dir", default="output",
                       help="Directory with cluster results")
    parser.add_argument("--data_dir", default="Ground_Truth/artifact_1",
                       help="Directory with PLY files")
    parser.add_argument("--no_original", action="store_true",
                       help="Don't show original surface points")
    parser.add_argument("--no_centers", action="store_true",
                       help="Don't show cluster center markers")
    parser.add_argument("--info_only", action="store_true",
                       help="Only print cluster info, don't show visualization")
    
    args = parser.parse_args()
    
    try:
        # Load data
        points, break_indices, original_indices, fragment_clusters = load_fragment_data(
            args.fragment, args.output_dir, args.data_dir
        )
        
        if points is None:
            print("Failed to load data!")
            return
        
        # Print cluster information
        print_cluster_info(fragment_clusters, args.fragment)
        
        # Show visualization unless info_only
        if not args.info_only:
            visualize_clusters_open3d(
                points, break_indices, original_indices, fragment_clusters,
                args.fragment, args.scale, 
                show_original=not args.no_original,
                show_centers=not args.no_centers
            )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()