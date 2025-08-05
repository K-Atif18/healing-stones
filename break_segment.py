import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import argparse
import os
import time

class PLYPointCloudSegmenter:
    def __init__(self, method='normal_clustering', downsample_factor=1.0):
        """
        Initialize point cloud segmenter.
        
        Args:
            method: Segmentation method ('normal_clustering', 'curvature', 'region_growing', 'ransac_planes')
            downsample_factor: Factor to downsample point cloud (1.0 = no downsampling)
        """
        self.method = method
        self.downsample_factor = downsample_factor
        
    def load_ply_as_pointcloud(self, filepath):
        """Load PLY file and convert to point cloud."""
        try:
            # Try loading as mesh first
            mesh = o3d.io.read_triangle_mesh(filepath)
            if len(mesh.vertices) > 0:
                print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
                
                # Convert mesh to point cloud by sampling
                if len(mesh.triangles) > 0:
                    # Sample points from mesh surface
                    num_points = min(len(mesh.vertices) * 2, 500000)  # Limit for memory
                    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
                    print(f"Sampled {len(pcd.points)} points from mesh surface")
                else:
                    # Use vertices directly if no triangles
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = mesh.vertices
                    print(f"Using {len(pcd.points)} mesh vertices as point cloud")
            else:
                # Try loading directly as point cloud
                pcd = o3d.io.read_point_cloud(filepath)
                print(f"Loaded point cloud with {len(pcd.points)} points")
            
            return pcd
            
        except Exception as e:
            print(f"Error loading PLY file: {e}")
            return None
    
    def downsample_pointcloud(self, pcd):
        """Downsample point cloud if needed."""
        if self.downsample_factor < 1.0:
            original_size = len(pcd.points)
            pcd = pcd.uniform_down_sample(every_k_points=int(1/self.downsample_factor))
            print(f"Downsampled from {original_size} to {len(pcd.points)} points")
        return pcd
    
    def compute_normals_and_curvature(self, pcd, radius=None, max_nn=30):
        """Compute normals and estimate curvature."""
        if radius is None:
            # Estimate radius based on point cloud size
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)
            radius = bbox_size * 0.02  # 2% of bounding box diagonal
        
        print(f"Computing normals with radius={radius:.4f}")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        
        # Estimate curvature using local neighborhood analysis
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        print("Computing curvature estimates...")
        tree = cKDTree(points)
        curvatures = np.zeros(len(points))
        
        for i in range(len(points)):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(points)} points for curvature")
            
            # Find neighbors within radius
            indices = tree.query_ball_point(points[i], radius)
            if len(indices) > 3:
                neighbor_normals = normals[indices]
                # Curvature as variance in normal directions
                curvatures[i] = np.var(neighbor_normals, axis=0).sum()
        
        return curvatures
    
    def segment_by_normal_clustering(self, pcd, eps=0.1, min_samples=50):
        """Segment point cloud by clustering surface normals."""
        print("Segmenting by normal clustering...")
        
        # Compute normals
        self.compute_normals_and_curvature(pcd)
        normals = np.asarray(pcd.normals)
        
        # Cluster normals using DBSCAN
        scaler = StandardScaler()
        normals_scaled = scaler.fit_transform(normals)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clustering.fit_predict(normals_scaled)
        
        print(f"Found {len(np.unique(labels[labels >= 0]))} normal clusters")
        return labels
    
    def segment_by_curvature(self, pcd, curvature_threshold=0.02, eps=0.05, min_samples=100):
        """Segment point cloud based on curvature analysis."""
        print("Segmenting by curvature...")
        
        # Compute curvature
        curvatures = self.compute_normals_and_curvature(pcd)
        points = np.asarray(pcd.points)
        
        # Separate high and low curvature regions
        high_curvature_mask = curvatures > curvature_threshold
        
        # Cluster high curvature points (edges/corners)
        if np.sum(high_curvature_mask) > min_samples:
            high_curv_points = points[high_curvature_mask]
            clustering = DBSCAN(eps=eps, min_samples=min_samples//2)
            high_curv_labels = clustering.fit_predict(high_curv_points)
        else:
            high_curv_labels = np.array([])
        
        # Cluster low curvature points (flat regions)
        low_curvature_mask = ~high_curvature_mask
        if np.sum(low_curvature_mask) > min_samples:
            low_curv_points = points[low_curvature_mask]
            clustering = DBSCAN(eps=eps*2, min_samples=min_samples)
            low_curv_labels = clustering.fit_predict(low_curv_points)
        else:
            low_curv_labels = np.array([])
        
        # Combine labels
        labels = np.full(len(points), -1)
        
        if len(high_curv_labels) > 0:
            max_label = np.max(high_curv_labels) if np.max(high_curv_labels) >= 0 else -1
            labels[high_curvature_mask] = high_curv_labels
        else:
            max_label = -1
        
        if len(low_curv_labels) > 0:
            low_curv_labels[low_curv_labels >= 0] += (max_label + 1)
            labels[low_curvature_mask] = low_curv_labels
        
        print(f"Found {len(np.unique(labels[labels >= 0]))} curvature-based segments")
        return labels
    
    def segment_by_region_growing(self, pcd, angle_threshold=15.0, curvature_threshold=0.02, min_cluster_size=100):
        """Segment using region growing algorithm."""
        print("Segmenting by region growing...")
        
        # Compute normals and curvature
        curvatures = self.compute_normals_and_curvature(pcd)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Build KD-tree for neighbor search
        tree = cKDTree(points)
        
        # Initialize
        labels = np.full(len(points), -1)
        visited = np.zeros(len(points), dtype=bool)
        current_label = 0
        
        angle_threshold_rad = np.deg2rad(angle_threshold)
        
        print("Growing regions...")
        for seed_idx in range(len(points)):
            if seed_idx % 5000 == 0:
                print(f"Processed {seed_idx}/{len(points)} seed points")
            
            if visited[seed_idx] or curvatures[seed_idx] > curvature_threshold:
                continue
            
            # Start new region
            region = []
            queue = [seed_idx]
            
            while queue:
                current_idx = queue.pop(0)
                if visited[current_idx]:
                    continue
                
                visited[current_idx] = True
                region.append(current_idx)
                
                # Find neighbors
                radius = 0.05  # Adjust based on point cloud density
                neighbor_indices = tree.query_ball_point(points[current_idx], radius)
                
                for neighbor_idx in neighbor_indices:
                    if not visited[neighbor_idx] and curvatures[neighbor_idx] <= curvature_threshold:
                        # Check normal similarity
                        angle = np.arccos(np.clip(
                            np.abs(np.dot(normals[current_idx], normals[neighbor_idx])), 0, 1
                        ))
                        
                        if angle < angle_threshold_rad:
                            queue.append(neighbor_idx)
            
            # Accept region if large enough
            if len(region) >= min_cluster_size:
                labels[region] = current_label
                current_label += 1
        
        print(f"Found {current_label} regions through region growing")
        return labels
    
    def segment_by_ransac_planes(self, pcd, distance_threshold=0.01, num_planes=10):
        """Segment point cloud by detecting multiple planes using RANSAC."""
        print("Segmenting by RANSAC plane detection...")
        
        remaining_pcd = pcd
        all_labels = np.full(len(pcd.points), -1)
        original_indices = np.arange(len(pcd.points))
        
        for plane_id in range(num_planes):
            if len(remaining_pcd.points) < 1000:  # Stop if too few points remain
                break
            
            # Detect plane
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            if len(inliers) < 500:  # Skip small planes
                break
            
            # Assign labels
            remaining_indices = original_indices[:len(remaining_pcd.points)]
            plane_indices = remaining_indices[inliers]
            all_labels[plane_indices] = plane_id
            
            print(f"Plane {plane_id}: {len(inliers)} points, equation: {plane_model}")
            
            # Remove inliers for next iteration
            remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
            original_indices = remaining_indices[np.setdiff1d(np.arange(len(remaining_indices)), inliers)]
        
        print(f"Found {len(np.unique(all_labels[all_labels >= 0]))} planes")
        return all_labels
    
    def segment_pointcloud(self, pcd):
        """Main segmentation function."""
        # Downsample if needed
        pcd = self.downsample_pointcloud(pcd)
        
        start_time = time.time()
        
        if self.method == 'normal_clustering':
            labels = self.segment_by_normal_clustering(pcd, eps=0.15, min_samples=100)
        elif self.method == 'curvature':
            labels = self.segment_by_curvature(pcd, curvature_threshold=0.02, eps=0.1, min_samples=200)
        elif self.method == 'region_growing':
            labels = self.segment_by_region_growing(pcd, angle_threshold=20.0, min_cluster_size=200)
        elif self.method == 'ransac_planes':
            labels = self.segment_by_ransac_planes(pcd, distance_threshold=0.02, num_planes=15)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")
        
        processing_time = time.time() - start_time
        print(f"Segmentation completed in {processing_time:.2f} seconds")
        
        return pcd, labels

class PointCloudVisualizer:
    def __init__(self):
        # Generate distinct colors for visualization
        self.colors = np.array([
            [1.0, 0.0, 0.0],    # Red
            [0.0, 1.0, 0.0],    # Green
            [0.0, 0.0, 1.0],    # Blue
            [1.0, 1.0, 0.0],    # Yellow
            [1.0, 0.0, 1.0],    # Magenta
            [0.0, 1.0, 1.0],    # Cyan
            [1.0, 0.5, 0.0],    # Orange
            [0.5, 0.0, 1.0],    # Purple
            [0.0, 0.5, 0.0],    # Dark Green
            [0.5, 0.5, 0.0],    # Olive
            [0.5, 0.0, 0.5],    # Dark Magenta
            [0.0, 0.5, 0.5],    # Teal
            [1.0, 0.75, 0.8],   # Pink
            [0.75, 1.0, 0.8],   # Light Green
            [0.8, 0.75, 1.0],   # Light Purple
            [1.0, 0.8, 0.6],    # Peach
            [0.6, 0.8, 1.0],    # Light Blue
            [0.8, 0.6, 0.4],    # Brown
            [0.4, 0.6, 0.8],    # Steel Blue
            [0.8, 0.4, 0.6],    # Rose
        ])
    
    def visualize_segmented_pointcloud(self, pcd, labels, save_path=None):
        """Visualize point cloud with different colors for each segment."""
        
        # Create color array
        points = np.asarray(pcd.points)
        point_colors = np.zeros((len(points), 3))
        
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        print(f"Visualizing {len(valid_labels)} segments")
        
        # Assign colors to segments
        for i, label in enumerate(valid_labels):
            color_idx = i % len(self.colors)
            point_colors[labels == label] = self.colors[color_idx]
            print(f"Segment {label}: {np.sum(labels == label)} points - Color: {self.colors[color_idx]}")
        
        # Gray for unassigned points
        point_colors[labels == -1] = [0.5, 0.5, 0.5]
        unassigned_count = np.sum(labels == -1)
        if unassigned_count > 0:
            print(f"Unassigned: {unassigned_count} points - Color: Gray")
        
        # Create colored point cloud
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = pcd.points
        colored_pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Visualize
        print("\nDisplaying segmented point cloud in Open3D...")
        print("Controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Ctrl + Mouse: Pan")
        print("- Press 'H' for help")
        print("- Press 'Q' or close window to exit")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PLY Point Cloud Segmentation", width=1400, height=900)
        vis.add_geometry(colored_pcd)
        
        # Configure render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        render_option.point_size = 2.0
        
        vis.run()
        
        if save_path:
            vis.capture_screen_image(save_path)
            print(f"Screenshot saved to {save_path}")
        
        vis.destroy_window()
        
        return colored_pcd
    
    def create_segmentation_statistics(self, labels, method_name, save_path=None):
        """Create statistics plot for segmentation results."""
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        if len(valid_labels) == 0:
            print("No segments found for statistics")
            return
        
        # Count points per segment
        segment_sizes = []
        for label in valid_labels:
            count = np.sum(labels == label)
            segment_sizes.append(count)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of segment sizes
        colors_for_plot = [self.colors[i % len(self.colors)] for i in range(len(segment_sizes))]
        ax1.bar(range(len(segment_sizes)), segment_sizes, color=colors_for_plot)
        ax1.set_xlabel('Segment ID')
        ax1.set_ylabel('Number of Points')
        ax1.set_title(f'Points per Segment ({method_name})')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of segment sizes
        ax2.hist(segment_sizes, bins=min(15, len(segment_sizes)), 
                 color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('Number of Segments')
        ax2.set_title('Distribution of Segment Sizes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Segment PLY files using point cloud techniques')
    parser.add_argument('ply_file', help='Path to PLY file')
    parser.add_argument('--method', choices=['normal_clustering', 'curvature', 'region_growing', 'ransac_planes'],
                       default='normal_clustering', help='Segmentation method (default: normal_clustering)')
    parser.add_argument('--downsample', type=float, default=1.0,
                       help='Downsample factor (1.0 = no downsampling, 0.5 = half points)')
    parser.add_argument('--save_screenshot', type=str, default=None,
                       help='Path to save screenshot')
    parser.add_argument('--save_statistics', type=str, default=None,
                       help='Path to save statistics plot')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.ply_file):
        print(f"Error: File {args.ply_file} not found")
        return
    
    # Initialize segmenter and visualizer
    segmenter = PLYPointCloudSegmenter(method=args.method, downsample_factor=args.downsample)
    visualizer = PointCloudVisualizer()
    
    # Load point cloud
    print(f"Loading PLY file: {args.ply_file}")
    pcd = segmenter.load_ply_as_pointcloud(args.ply_file)
    if pcd is None:
        return
    
    # Segment point cloud
    segmented_pcd, labels = segmenter.segment_pointcloud(pcd)
    
    # Print results
    unique_labels = np.unique(labels)
    num_segments = len(unique_labels[unique_labels >= 0])
    unassigned_points = np.sum(labels == -1)
    
    print(f"\nSegmentation Results ({args.method}):")
    print(f"Number of segments: {num_segments}")
    print(f"Unassigned points: {unassigned_points}")
    print(f"Total points: {len(labels)}")
    
    if num_segments == 0:
        print("No segments found. Try a different method or adjust parameters.")
        return
    
    # Visualize results
    visualizer.visualize_segmented_pointcloud(
        segmented_pcd, labels, save_path=args.save_screenshot
    )
    
    # Create statistics
    visualizer.create_segmentation_statistics(
        labels, args.method, save_path=args.save_statistics
    )

if __name__ == "__main__":
    main()

# Example usage:
# python ply_pointcloud_segmentation.py fragment.ply --method normal_clustering
# python ply_pointcloud_segmentation.py fragment.ply --method curvature --downsample 0.5
# python ply_pointcloud_segmentation.py fragment.ply --method region_growing --save_screenshot result.png
# python ply_pointcloud_segmentation.py fragment.ply --method ransac_planes