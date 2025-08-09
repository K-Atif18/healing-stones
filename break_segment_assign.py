import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import argparse
import os
import time

class StoneFragmentFaceClassifier:
    def __init__(self, downsample_factor=1.0):
        """
        Initialize stone fragment face classifier.
        
        Args:
            downsample_factor: Factor to downsample point cloud (1.0 = no downsampling)
        """
        self.downsample_factor = downsample_factor
        
        # Roughness thresholds for surface classification (simplified to 2 types)
        self.roughness_thresholds = {
            'break_max': 0.005,      # Very smooth (virtual breaks from cell fracture)
        }
        
        # Colors for different surface types
        self.surface_colors = {
            'break': np.array([0.2, 0.2, 0.9]),     # Blue for break surfaces
            'carved': np.array([0.9, 0.2, 0.2]),    # Red for carved surfaces
            'unclassified': np.array([0.7, 0.7, 0.7])  # Gray for unclassified
        }
        
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
    
    def compute_normals(self, pcd, radius=None, max_nn=30):
        """Compute normals for the point cloud."""
        if radius is None:
            # Estimate radius based on point cloud size
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = np.linalg.norm(bbox.max_bound - bbox.min_bound)
            radius = bbox_size * 0.02  # 2% of bounding box diagonal
        
        print(f"Computing normals with radius={radius:.4f}")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        return radius
    
    def segment_by_normal_clustering(self, pcd, eps=0.15, min_samples=100):
        """Segment point cloud by clustering surface normals."""
        print("Segmenting faces by normal clustering...")
        
        # Compute normals
        normal_radius = self.compute_normals(pcd)
        normals = np.asarray(pcd.normals)
        
        # Cluster normals using DBSCAN
        scaler = StandardScaler()
        normals_scaled = scaler.fit_transform(normals)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clustering.fit_predict(normals_scaled)
        
        n_clusters = len(np.unique(labels[labels >= 0]))
        print(f"Found {n_clusters} face segments")
        return labels, normal_radius
    
    def calculate_face_roughness(self, pcd, labels, radius):
        """Calculate roughness for each segmented face."""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        face_roughness = {}
        
        print("\nCalculating roughness for each face...")
        
        for label in valid_labels:
            face_mask = labels == label
            face_points = points[face_mask]
            face_normals = normals[face_mask]
            
            if len(face_points) < 10:
                continue
            
            print(f"Analyzing face {label} with {len(face_points)} points...")
            
            # Calculate roughness metrics for this face
            roughness_metrics = self._calculate_roughness_metrics(
                face_points, face_normals, radius
            )
            
            face_roughness[int(label)] = roughness_metrics
            print(f"  RMS roughness: {roughness_metrics['rms_roughness']:.6f}")
            print(f"  Normal variation: {roughness_metrics['normal_variation']:.6f}")
            print(f"  Combined roughness: {roughness_metrics['combined_roughness']:.6f}")
        
        return face_roughness
    
    def _calculate_roughness_metrics(self, face_points, face_normals, radius):
        """Calculate various roughness metrics for a face."""
        # Create KDTree for this specific face
        face_tree = cKDTree(face_points)
        
        roughness_metrics = {}
        deviations = []
        normal_variations = []
        
        # Sample points for efficiency (max 500 points per face)
        n_samples = min(500, len(face_points))
        sample_indices = np.linspace(0, len(face_points)-1, n_samples, dtype=int)
        
        for i in sample_indices:
            point = face_points[i]
            
            # Find neighbors within radius
            neighbors_idx = face_tree.query_ball_point(point, radius)
            
            if len(neighbors_idx) > 3:
                neighbor_points = face_points[neighbors_idx]
                
                # Calculate local plane using PCA
                centered_points = neighbor_points - np.mean(neighbor_points, axis=0)
                
                # Handle case where all points are nearly the same
                if np.allclose(centered_points, 0, atol=1e-8):
                    continue
                
                try:
                    # Compute covariance matrix
                    cov_matrix = np.cov(centered_points.T)
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    
                    # The eigenvector with smallest eigenvalue is the normal
                    plane_normal = eigenvecs[:, 0]
                    
                    # Calculate distances from points to the plane
                    distances = np.abs(np.dot(centered_points, plane_normal))
                    deviations.extend(distances)
                    
                except (np.linalg.LinAlgError, ValueError):
                    # Handle singular matrix case
                    continue
                
                # Calculate normal variation (angular deviation)
                if i < len(face_normals) and len(neighbors_idx) <= len(face_normals):
                    try:
                        current_normal = face_normals[i]
                        neighbor_normals = face_normals[neighbors_idx]
                        
                        for neighbor_normal in neighbor_normals:
                            # Calculate angle between normals
                            dot_product = np.clip(np.abs(np.dot(current_normal, neighbor_normal)), 0, 1)
                            angle = np.arccos(dot_product)
                            normal_variations.append(angle)
                    except (IndexError, ValueError):
                        continue
        
        # Calculate metrics
        if deviations:
            roughness_metrics['rms_roughness'] = np.sqrt(np.mean(np.array(deviations)**2))
            roughness_metrics['mean_deviation'] = np.mean(deviations)
            roughness_metrics['max_deviation'] = np.max(deviations)
            roughness_metrics['std_deviation'] = np.std(deviations)
        else:
            roughness_metrics['rms_roughness'] = 0.001  # Small default value
            roughness_metrics['mean_deviation'] = 0.0
            roughness_metrics['max_deviation'] = 0.0
            roughness_metrics['std_deviation'] = 0.0
        
        if normal_variations:
            roughness_metrics['normal_variation'] = np.mean(normal_variations)
            roughness_metrics['normal_std'] = np.std(normal_variations)
        else:
            roughness_metrics['normal_variation'] = 0.0
            roughness_metrics['normal_std'] = 0.0
        
        # Combined roughness score (weighted combination)
        roughness_metrics['combined_roughness'] = (
            roughness_metrics['rms_roughness'] * 0.7 +
            roughness_metrics['normal_variation'] * 0.3
        )
        
        return roughness_metrics
    
    def classify_faces(self, face_roughness):
        """Classify faces based on roughness into carved/break categories."""
        face_classifications = {}
        
        print(f"\nClassifying faces based on roughness...")
        print(f"Threshold: Break ≤ {self.roughness_thresholds['break_max']:.3f}, Carved > {self.roughness_thresholds['break_max']:.3f}")
        print("-" * 80)
        
        for face_id, metrics in face_roughness.items():
            combined_roughness = metrics['combined_roughness']
            
            # Simplified classification: break vs carved only
            if combined_roughness <= self.roughness_thresholds['break_max']:
                face_type = 'break'
            else:
                face_type = 'carved'
            
            face_classifications[face_id] = {
                'type': face_type,
                'roughness_metrics': metrics
            }
            
            print(f"Face {face_id:2d}: {face_type:6s} (roughness: {combined_roughness:.6f})")
        
        return face_classifications
    
    def create_classified_visualization(self, pcd, labels, face_classifications):
        """Create visualization with faces colored by their classification."""
        points = np.asarray(pcd.points)
        point_colors = np.zeros((len(points), 3))
        
        # Count surfaces by type (simplified to 2 types)
        surface_type_counts = {'break': 0, 'carved': 0, 'unclassified': 0}
        
        print(f"\nColoring faces by classification...")
        
        # Color points based on face classification
        for face_id, classification in face_classifications.items():
            face_type = classification['type']
            color = self.surface_colors[face_type]
            
            face_mask = labels == face_id
            point_colors[face_mask] = color
            surface_type_counts[face_type] += 1
            
            print(f"Face {face_id}: {face_type} - {np.sum(face_mask)} points")
        
        # Gray for unassigned points
        unassigned_mask = labels == -1
        point_colors[unassigned_mask] = self.surface_colors['unclassified']
        unassigned_count = np.sum(unassigned_mask)
        if unassigned_count > 0:
            print(f"Unassigned: {unassigned_count} points")
        
        # Create colored point cloud
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = pcd.points
        colored_pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Print classification summary
        print(f"\n" + "="*50)
        print(f"FACE CLASSIFICATION SUMMARY")
        print(f"="*50)
        print(f"🔵 Break surfaces:   {surface_type_counts['break']:2d} faces (smooth)")
        print(f"🔴 Carved surfaces:  {surface_type_counts['carved']:2d} faces (rough)")
        print(f"⚫ Unclassified:     {surface_type_counts['unclassified']:2d} faces")
        print(f"="*50)
        
        return colored_pcd, surface_type_counts
    
    def visualize_classified_faces(self, colored_pcd, surface_type_counts):
        """Display the classified faces in Open3D viewer."""
        print("\nDisplaying classified faces in Open3D...")
        print("Legend:")
        print("🔵 BLUE = Break surfaces (very smooth - from virtual fracturing)")
        print("🔴 RED  = Carved surfaces (rough - all other surfaces)")
        print("⚫ GRAY = Unclassified surfaces")
        print("\nControls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Ctrl + Mouse: Pan")
        print("- Press 'H' for help")
        print("- Press 'Q' or close window to exit")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Stone Fragment - Face Classification", 
            width=1400, 
            height=900
        )
        vis.add_geometry(colored_pcd)
        
        # Configure render options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.05, 0.05, 0.05])  # Very dark background
        render_option.point_size = 3.0
        
        # Get view control and set a good viewing angle
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        vis.run()
        vis.destroy_window()
    
    def process_single_fragment(self, ply_file_path):
        """Process a single PLY fragment file."""
        print(f"Processing fragment: {ply_file_path}")
        print("="*60)
        
        # Load point cloud
        pcd = self.load_ply_as_pointcloud(ply_file_path)
        if pcd is None:
            return None
        
        # Downsample if needed
        pcd = self.downsample_pointcloud(pcd)
        
        # Segment faces by normal clustering
        start_time = time.time()
        labels, normal_radius = self.segment_by_normal_clustering(pcd, eps=0.15, min_samples=100)
        
        # Check if we found any faces
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        if len(valid_labels) == 0:
            print("No faces found. Try adjusting segmentation parameters.")
            return None
        
        print(f"Found {len(valid_labels)} faces to analyze")
        
        # Calculate roughness for each face
        face_roughness = self.calculate_face_roughness(pcd, labels, normal_radius)
        
        # Classify faces based on roughness
        face_classifications = self.classify_faces(face_roughness)
        
        # Create visualization
        colored_pcd, surface_counts = self.create_classified_visualization(
            pcd, labels, face_classifications
        )
        
        # Display in Open3D
        self.visualize_classified_faces(colored_pcd, surface_counts)
        
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        
        return {
            'pcd': pcd,
            'labels': labels,
            'face_classifications': face_classifications,
            'surface_counts': surface_counts
        }
    
    def segment_by_normal_clustering(self, pcd, eps=0.15, min_samples=100):
        """Segment point cloud by clustering surface normals."""
        print("Segmenting faces by normal clustering...")
        
        # Compute normals
        normal_radius = self.compute_normals(pcd)
        normals = np.asarray(pcd.normals)
        
        # Cluster normals using DBSCAN
        scaler = StandardScaler()
        normals_scaled = scaler.fit_transform(normals)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clustering.fit_predict(normals_scaled)
        
        n_clusters = len(np.unique(labels[labels >= 0]))
        print(f"Found {n_clusters} face segments")
        return labels, normal_radius
    
    def calculate_face_roughness(self, pcd, labels, radius):
        """Calculate roughness for each segmented face."""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        unique_labels = np.unique(labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        face_roughness = {}
        
        print("\nCalculating roughness for each face...")
        
        for label in valid_labels:
            face_mask = labels == label
            face_points = points[face_mask]
            face_normals = normals[face_mask]
            
            if len(face_points) < 10:
                continue
            
            print(f"Analyzing face {label} with {len(face_points)} points...")
            
            # Calculate roughness metrics for this face
            roughness_metrics = self._calculate_roughness_metrics(
                face_points, face_normals, radius
            )
            
            face_roughness[int(label)] = roughness_metrics
            print(f"  RMS roughness: {roughness_metrics['rms_roughness']:.6f}")
            print(f"  Normal variation: {roughness_metrics['normal_variation']:.6f}")
            print(f"  Combined roughness: {roughness_metrics['combined_roughness']:.6f}")
        
        return face_roughness
    
    def _calculate_roughness_metrics(self, face_points, face_normals, radius):
        """Calculate various roughness metrics for a face."""
        # Create KDTree for this specific face
        face_tree = cKDTree(face_points)
        
        roughness_metrics = {}
        deviations = []
        normal_variations = []
        
        # Sample points for efficiency (max 200 points per face for roughness calculation)
        n_samples = min(200, len(face_points))
        sample_indices = np.linspace(0, len(face_points)-1, n_samples, dtype=int)
        
        for i in sample_indices:
            point = face_points[i]
            
            # Find neighbors within radius
            neighbors_idx = face_tree.query_ball_point(point, radius)
            
            if len(neighbors_idx) > 3:
                neighbor_points = face_points[neighbors_idx]
                
                # Calculate local plane using PCA
                centered_points = neighbor_points - np.mean(neighbor_points, axis=0)
                
                # Handle case where all points are nearly the same
                if np.allclose(centered_points, 0, atol=1e-8):
                    continue
                
                try:
                    # Use SVD for better numerical stability
                    U, s, Vt = np.linalg.svd(centered_points, full_matrices=False)
                    
                    # Check if we have enough variation
                    if len(s) > 0 and s[0] > 1e-8:
                        # The last row of Vt is the normal to the best-fit plane
                        plane_normal = Vt[-1]
                        
                        # Calculate distances from points to the plane
                        distances = np.abs(np.dot(centered_points, plane_normal))
                        deviations.extend(distances)
                    
                except (np.linalg.LinAlgError, ValueError):
                    continue
                
                # Calculate normal variation (angular deviation)
                if i < len(face_normals):
                    try:
                        current_normal = face_normals[i]
                        
                        # Get neighbor normals (ensure indices are valid)
                        valid_neighbor_indices = [idx for idx in neighbors_idx if idx < len(face_normals)]
                        
                        if valid_neighbor_indices:
                            neighbor_normals = face_normals[valid_neighbor_indices]
                            
                            for neighbor_normal in neighbor_normals:
                                # Calculate angle between normals
                                dot_product = np.clip(np.abs(np.dot(current_normal, neighbor_normal)), 0, 1)
                                angle = np.arccos(dot_product)
                                normal_variations.append(angle)
                    except (IndexError, ValueError):
                        continue
        
        # Calculate final metrics
        if deviations:
            deviations_array = np.array(deviations)
            roughness_metrics['rms_roughness'] = np.sqrt(np.mean(deviations_array**2))
            roughness_metrics['mean_deviation'] = np.mean(deviations_array)
            roughness_metrics['max_deviation'] = np.max(deviations_array)
            roughness_metrics['std_deviation'] = np.std(deviations_array)
        else:
            # Default values for very smooth surfaces
            roughness_metrics['rms_roughness'] = 0.0005
            roughness_metrics['mean_deviation'] = 0.0
            roughness_metrics['max_deviation'] = 0.0
            roughness_metrics['std_deviation'] = 0.0
        
        if normal_variations:
            normal_variations_array = np.array(normal_variations)
            roughness_metrics['normal_variation'] = np.mean(normal_variations_array)
            roughness_metrics['normal_std'] = np.std(normal_variations_array)
        else:
            roughness_metrics['normal_variation'] = 0.0
            roughness_metrics['normal_std'] = 0.0
        
        # Combined roughness score (weighted combination)
        roughness_metrics['combined_roughness'] = (
            roughness_metrics['rms_roughness'] * 0.7 +
            roughness_metrics['normal_variation'] * 0.3
        )
        
        # Additional surface quality metrics
        roughness_metrics['point_count'] = len(face_points)
        roughness_metrics['surface_area_estimate'] = len(face_points) * (radius**2)  # Rough estimate
        
        return roughness_metrics

def main():
    """Main function to process a single stone fragment."""
    parser = argparse.ArgumentParser(description='Classify stone fragment faces as carved/original/break')
    parser.add_argument('ply_file', help='Path to PLY file')
    parser.add_argument('--downsample', type=float, default=1.0,
                       help='Downsample factor (1.0 = no downsampling, 0.5 = half points)')
    parser.add_argument('--eps', type=float, default=0.15,
                       help='DBSCAN eps parameter for face segmentation')
    parser.add_argument('--min_samples', type=int, default=100,
                       help='DBSCAN min_samples parameter for face segmentation')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.ply_file):
        print(f"Error: File {args.ply_file} not found")
        return
    
    # Initialize classifier
    classifier = StoneFragmentFaceClassifier(downsample_factor=args.downsample)
    
    # Process the fragment
    result = classifier.process_single_fragment(args.ply_file)
    
    if result:
        print(f"\nAnalysis complete!")
        print(f"Total faces found: {len(result['face_classifications'])}")
        print(f"Break surfaces: {result['surface_counts']['break']}")
        print(f"Carved surfaces: {result['surface_counts']['carved']}")

if __name__ == "__main__":
    main()

# Example usage:
# python face_classifier.py fragment.ply
# python face_classifier.py fragment.ply --downsample 0.8 --eps 0.12
# python face_classifier.py ply_fragments/frag_3_cell_04.ply