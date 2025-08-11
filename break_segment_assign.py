import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree
import argparse
import os
import time
import pickle
import json
from pathlib import Path
import glob

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
        
        # Colors for break surfaces (always blue)
        self.break_color = np.array([0.2, 0.2, 0.9])  # Blue for break surfaces
        
        # Generate distinct colors for carved surfaces
        self.carved_colors = np.array([
            [1.0, 0.0, 0.0],    # Red
            [0.0, 0.8, 0.0],    # Green
            [1.0, 0.6, 0.0],    # Orange
            [0.8, 0.0, 0.8],    # Magenta
            [0.0, 0.8, 0.8],    # Cyan
            [1.0, 1.0, 0.0],    # Yellow
            [0.6, 0.0, 1.0],    # Purple
            [1.0, 0.4, 0.6],    # Pink
            [0.4, 0.8, 0.2],    # Lime Green
            [0.8, 0.4, 0.0],    # Brown
            [0.6, 0.6, 0.0],    # Olive
            [0.8, 0.0, 0.4],    # Dark Pink
            [0.0, 0.6, 0.6],    # Teal
            [0.4, 0.0, 0.8],    # Indigo
            [0.8, 0.8, 0.0],    # Gold
            [0.6, 0.2, 0.8],    # Violet
            [0.2, 0.8, 0.4],    # Sea Green
            [0.8, 0.2, 0.6],    # Rose
            [0.4, 0.6, 0.8],    # Sky Blue
            [0.8, 0.6, 0.2],    # Amber
        ])
        
        # Gray for unclassified
        self.unclassified_color = np.array([0.7, 0.7, 0.7])
        
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
        
        # Count surfaces by type
        surface_type_counts = {'break': 0, 'carved': 0, 'unclassified': 0}
        
        print(f"\nColoring faces by classification...")
        
        # Separate break and carved faces
        break_faces = []
        carved_faces = []
        
        for face_id, classification in face_classifications.items():
            face_type = classification['type']
            if face_type == 'break':
                break_faces.append(face_id)
            elif face_type == 'carved':
                carved_faces.append(face_id)
        
        # Color break surfaces (all blue)
        for face_id in break_faces:
            face_mask = labels == face_id
            point_colors[face_mask] = self.break_color
            surface_type_counts['break'] += 1
            print(f"Face {face_id}: BREAK (blue) - {np.sum(face_mask)} points")
        
        # Color carved surfaces (different colors for each)
        for i, face_id in enumerate(carved_faces):
            face_mask = labels == face_id
            color_idx = i % len(self.carved_colors)
            point_colors[face_mask] = self.carved_colors[color_idx]
            surface_type_counts['carved'] += 1
            
            color_name = self._get_color_name(self.carved_colors[color_idx])
            print(f"Face {face_id}: CARVED ({color_name}) - {np.sum(face_mask)} points")
        
        # Gray for unassigned points
        unassigned_mask = labels == -1
        point_colors[unassigned_mask] = self.unclassified_color
        unassigned_count = np.sum(unassigned_mask)
        if unassigned_count > 0:
            print(f"Unassigned: {unassigned_count} points")
            surface_type_counts['unclassified'] = 1
        
        # Create colored point cloud
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = pcd.points
        colored_pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # Print classification summary
        print(f"\n" + "="*50)
        print(f"FACE CLASSIFICATION SUMMARY")
        print(f"="*50)
        print(f"🔵 Break surfaces:   {surface_type_counts['break']:2d} faces (smooth)")
        print(f"🎨 Carved surfaces:  {surface_type_counts['carved']:2d} faces (rough - each in different color)")
        if surface_type_counts['unclassified'] > 0:
            print(f"⚫ Unclassified:     {surface_type_counts['unclassified']:2d} faces")
        print(f"="*50)
        
        return colored_pcd, surface_type_counts
    
    def _get_color_name(self, color):
        """Get a descriptive name for a color."""
        r, g, b = color
        
        if r > 0.8 and g < 0.3 and b < 0.3:
            return "red"
        elif g > 0.7 and r < 0.3 and b < 0.3:
            return "green"
        elif r > 0.8 and g > 0.5 and b < 0.3:
            return "orange"
        elif r > 0.7 and g < 0.3 and b > 0.7:
            return "magenta"
        elif r < 0.3 and g > 0.7 and b > 0.7:
            return "cyan"
        elif r > 0.8 and g > 0.8 and b < 0.3:
            return "yellow"
        elif r > 0.5 and g < 0.3 and b > 0.8:
            return "purple"
        elif r > 0.8 and g > 0.3 and b > 0.5:
            return "pink"
        elif r > 0.7 and g > 0.3 and b < 0.3:
            return "brown"
        elif r < 0.5 and g > 0.5 and b > 0.7:
            return "sky blue"
        else:
            return "mixed"
    
    def save_results(self, results, output_path, save_format='pkl'):
        """
        Save analysis results to file.
        
        Args:
            results: Dictionary containing analysis results
            output_path: Path where to save the results
            save_format: 'pkl' for pickle or 'json' for JSON format
        """
        # Prepare serializable results
        serializable_results = self._prepare_serializable_results(results)
        
        if save_format.lower() == 'pkl':
            output_file = Path(output_path).with_suffix('.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(serializable_results, f)
            print(f"Results saved to {output_file}")
            
        elif save_format.lower() == 'json':
            output_file = Path(output_path).with_suffix('.json')
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=self._json_serializer)
            print(f"Results saved to {output_file}")
            
        else:
            raise ValueError("save_format must be 'pkl' or 'json'")
        
        return output_file
    
    def _prepare_serializable_results(self, results):
        """Prepare results for serialization by converting numpy arrays to lists."""
        serializable = {}
        
        for key, value in results.items():
            if key == 'pcd':
                # Save point cloud data as arrays
                serializable['point_cloud'] = {
                    'points': np.asarray(value.points).tolist(),
                    'colors': np.asarray(value.colors).tolist() if value.has_colors() else None,
                    'normals': np.asarray(value.normals).tolist() if value.has_normals() else None
                }
            elif key == 'colored_pcd':
                # Save colored point cloud data as arrays
                serializable['colored_point_cloud'] = {
                    'points': np.asarray(value.points).tolist(),
                    'colors': np.asarray(value.colors).tolist() if value.has_colors() else None,
                    'normals': np.asarray(value.normals).tolist() if value.has_normals() else None
                }
            elif key == 'labels':
                serializable['labels'] = value.tolist() if isinstance(value, np.ndarray) else value
            elif isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif key in ['pcd', 'colored_pcd']:  # Skip Open3D objects
                continue
            else:
                serializable[key] = value
        
        return serializable
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy objects."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def load_results(self, file_path):
        """
        Load previously saved results.
        
        Args:
            file_path: Path to the saved results file
            
        Returns:
            Dictionary containing the loaded results
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                results = pickle.load(f)
            print(f"Results loaded from {file_path}")
            
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                results = json.load(f)
            print(f"Results loaded from {file_path}")
            
        else:
            raise ValueError("File must have .pkl or .json extension")
        
        return results
    
    def visualize_classified_faces(self, colored_pcd, surface_type_counts):
        """Display the classified faces in Open3D viewer."""
        print("\nDisplaying classified faces in Open3D...")
        print("Legend:")
        print("🔵 BLUE = Break surfaces (very smooth - from virtual fracturing)")
        print("🎨 MULTIPLE COLORS = Carved surfaces (rough - each face has unique color)")
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
    
    def process_single_fragment(self, ply_file_path, save_results=True, save_format='pkl', 
                              output_dir=None, show_visualization=True):
        """
        Process a single PLY fragment file.
        
        Args:
            ply_file_path: Path to PLY file
            save_results: Whether to save results to file
            save_format: 'pkl' or 'json'
            output_dir: Directory to save results (default: same as input file)
            show_visualization: Whether to show 3D visualization
        """
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
        
        # Prepare results
        results = {
            'filename': os.path.basename(ply_file_path),
            'filepath': ply_file_path,
            'processing_time': time.time() - start_time,
            'parameters': {
                'downsample_factor': self.downsample_factor,
                'roughness_thresholds': self.roughness_thresholds,
                'normal_radius': normal_radius
            },
            'pcd': pcd,
            'labels': labels,
            'face_classifications': face_classifications,
            'surface_counts': surface_counts,
            'colored_pcd': colored_pcd
        }
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                output_dir = os.path.dirname(ply_file_path)
            
            base_name = Path(ply_file_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_analysis")
            self.save_results(results, output_path, save_format)
        
        # Display visualization if requested
        if show_visualization:
            self.visualize_classified_faces(colored_pcd, surface_counts)
        
        processing_time = time.time() - start_time
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        
        return results
    
    def process_folder(self, input_folder, output_folder=None, save_format='pkl', 
                      show_individual_visualizations=False, create_summary=True):
        """
        Process all PLY files in a folder.
        
        Args:
            input_folder: Path to folder containing PLY files
            output_folder: Path to output folder (default: input_folder + '_results')
            save_format: 'pkl' or 'json'
            show_individual_visualizations: Whether to show visualization for each file
            create_summary: Whether to create a summary report
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            print(f"Error: Input folder {input_folder} not found")
            return None
        
        # Find all PLY files
        ply_files = list(input_path.glob("*.ply"))
        if not ply_files:
            print(f"No PLY files found in {input_folder}")
            return None
        
        print(f"Found {len(ply_files)} PLY files to process")
        
        # Set up output directory
        if output_folder is None:
            output_folder = f"{input_folder}_results"
        
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        print(f"Results will be saved to: {output_path}")
        
        # Process each file
        all_results = {}
        batch_summary = {
            'total_files': len(ply_files),
            'successful_files': 0,
            'failed_files': 0,
            'total_faces': 0,
            'total_break_faces': 0,
            'total_carved_faces': 0,
            'processing_start_time': time.time(),
            'file_results': {}
        }
        
        for i, ply_file in enumerate(ply_files, 1):
            print(f"\n{'='*80}")
            print(f"Processing file {i}/{len(ply_files)}: {ply_file.name}")
            print(f"{'='*80}")
            
            try:
                # Process single fragment
                result = self.process_single_fragment(
                    str(ply_file), 
                    save_results=True,
                    save_format=save_format,
                    output_dir=str(output_path),
                    show_visualization=show_individual_visualizations
                )
                
                if result:
                    all_results[ply_file.name] = result
                    batch_summary['successful_files'] += 1
                    batch_summary['total_faces'] += len(result['face_classifications'])
                    batch_summary['total_break_faces'] += result['surface_counts']['break']
                    batch_summary['total_carved_faces'] += result['surface_counts']['carved']
                    
                    # Store file summary
                    batch_summary['file_results'][ply_file.name] = {
                        'faces_found': len(result['face_classifications']),
                        'break_faces': result['surface_counts']['break'],
                        'carved_faces': result['surface_counts']['carved'],
                        'processing_time': result['processing_time']
                    }
                    
                    print(f"✅ Successfully processed {ply_file.name}")
                else:
                    batch_summary['failed_files'] += 1
                    print(f"❌ Failed to process {ply_file.name}")
                    
            except Exception as e:
                print(f"❌ Error processing {ply_file.name}: {e}")
                batch_summary['failed_files'] += 1
        
        # Complete batch summary
        batch_summary['processing_end_time'] = time.time()
        batch_summary['total_processing_time'] = batch_summary['processing_end_time'] - batch_summary['processing_start_time']
        
        # Save batch summary
        if create_summary:
            summary_file = output_path / f"batch_summary.{save_format}"
            if save_format == 'pkl':
                with open(summary_file, 'wb') as f:
                    pickle.dump(batch_summary, f)
            else:
                with open(summary_file, 'w') as f:
                    json.dump(batch_summary, f, indent=2, default=self._json_serializer)
            
            print(f"\nBatch summary saved to: {summary_file}")
        
        # Print final summary
        self._print_batch_summary(batch_summary)
        
        return all_results, batch_summary
    
    def _print_batch_summary(self, batch_summary):
        """Print a formatted batch processing summary."""
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*80}")
        print(f"Total files processed: {batch_summary['total_files']}")
        print(f"✅ Successful: {batch_summary['successful_files']}")
        print(f"❌ Failed: {batch_summary['failed_files']}")
        print(f"Total processing time: {batch_summary['total_processing_time']:.2f} seconds")
        print(f"\nAGGREGATE STATISTICS:")
        print(f"Total faces found: {batch_summary['total_faces']}")
        print(f"🔵 Total break surfaces: {batch_summary['total_break_faces']}")
        print(f"🎨 Total carved surfaces: {batch_summary['total_carved_faces']}")
        
        if batch_summary['total_faces'] > 0:
            break_percentage = (batch_summary['total_break_faces'] / batch_summary['total_faces']) * 100
            carved_percentage = (batch_summary['total_carved_faces'] / batch_summary['total_faces']) * 100
            print(f"Break surface ratio: {break_percentage:.1f}%")
            print(f"Carved surface ratio: {carved_percentage:.1f}%")
        
        print(f"\nPER-FILE BREAKDOWN:")
        print(f"{'Filename':<30} {'Faces':<6} {'Break':<6} {'Carved':<7} {'Time(s)':<8}")
        print(f"{'-'*30} {'-'*6} {'-'*6} {'-'*7} {'-'*8}")
        
        for filename, file_result in batch_summary['file_results'].items():
            print(f"{filename:<30} {file_result['faces_found']:<6} "
                  f"{file_result['break_faces']:<6} {file_result['carved_faces']:<7} "
                  f"{file_result['processing_time']:<8.2f}")
        
        print(f"{'='*80}")
    
    def load_and_visualize_results(self, results_file):
        """
        Load saved results and visualize them.
        
        Args:
            results_file: Path to saved results file (.pkl or .json)
        """
        results = self.load_results(results_file)
        
        # Reconstruct point cloud from saved data
        if 'colored_point_cloud' in results:
            pcd_data = results['colored_point_cloud']
            colored_pcd = o3d.geometry.PointCloud()
            colored_pcd.points = o3d.utility.Vector3dVector(np.array(pcd_data['points']))
            if pcd_data['colors']:
                colored_pcd.colors = o3d.utility.Vector3dVector(np.array(pcd_data['colors']))
            if pcd_data['normals']:
                colored_pcd.normals = o3d.utility.Vector3dVector(np.array(pcd_data['normals']))
            
            # Show visualization
            self.visualize_classified_faces(colored_pcd, results['surface_counts'])
        elif 'point_cloud' in results:
            # Fallback to original point cloud data
            pcd_data = results['point_cloud']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(pcd_data['points']))
            if pcd_data['normals']:
                pcd.normals = o3d.utility.Vector3dVector(np.array(pcd_data['normals']))
            
            # Recreate visualization from labels and classifications
            labels = np.array(results['labels'])
            face_classifications = results['face_classifications']
            colored_pcd, surface_counts = self.create_classified_visualization(
                pcd, labels, face_classifications
            )
            self.visualize_classified_faces(colored_pcd, surface_counts)
        
        return results


def find_ply_files(directory):
    """Find all PLY files in a directory."""
    ply_files = []
    for ext in ['*.ply', '*.PLY']:
        ply_files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(ply_files)


def main():
    """Main function to process stone fragments."""
    parser = argparse.ArgumentParser(description='Classify stone fragment faces as carved/break')
    
    # Input options
    parser.add_argument('input', nargs='?', help='Path to PLY file or folder containing PLY files')
    parser.add_argument('--folder', action='store_true', 
                       help='Process entire folder instead of single file')
    
    # Output options
    parser.add_argument('--output', help='Output directory for results (default: auto-generated)')
    parser.add_argument('--format', choices=['pkl', 'json'], default='pkl',
                       help='Output format for saved results (default: pkl)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    
    # Processing options
    parser.add_argument('--downsample', type=float, default=1.0,
                       help='Downsample factor (1.0 = no downsampling, 0.5 = half points)')
    parser.add_argument('--eps', type=float, default=0.15,
                       help='DBSCAN eps parameter for face segmentation')
    parser.add_argument('--min_samples', type=int, default=100,
                       help='DBSCAN min_samples parameter for face segmentation')
    
    # Visualization options
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization (useful for batch processing)')
    parser.add_argument('--show-individual', action='store_true',
                       help='Show visualization for each file when processing folder')
    
    # Load existing results
    parser.add_argument('--load', help='Load and visualize existing results file')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = StoneFragmentFaceClassifier(downsample_factor=args.downsample)
    
    # Handle loading existing results
    if args.load:
        if not os.path.exists(args.load):
            print(f"Error: Results file {args.load} not found")
            return
        
        print(f"Loading results from: {args.load}")
        classifier.load_and_visualize_results(args.load)
        return
    
    # Check if input is provided when not loading
    if not args.input:
        print("Error: Input path is required when not using --load")
        parser.print_help()
        return
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"Error: Input {args.input} not found")
        return
    
    # Process folder or single file
    if args.folder or os.path.isdir(args.input):
        print(f"Processing folder: {args.input}")
        
        # Set up output directory
        output_dir = args.output
        if output_dir is None:
            output_dir = f"{args.input}_results"
        
        # Process all files in folder
        results, summary = classifier.process_folder(
            input_folder=args.input,
            output_folder=output_dir,
            save_format=args.format,
            show_individual_visualizations=args.show_individual,
            create_summary=not args.no_save
        )
        
        if results:
            print(f"\nBatch processing complete!")
            print(f"Results saved to: {output_dir}")
    
    else:
        print(f"Processing single file: {args.input}")
        
        # Process single file
        result = classifier.process_single_fragment(
            ply_file_path=args.input,
            save_results=not args.no_save,
            save_format=args.format,
            output_dir=args.output,
            show_visualization=not args.no_viz
        )
        
        if result:
            print(f"\nAnalysis complete!")
            print(f"Total faces found: {len(result['face_classifications'])}")
            print(f"Break surfaces: {result['surface_counts']['break']}")
            print(f"Carved surfaces: {result['surface_counts']['carved']}")


if __name__ == "__main__":
    main()


# Example usage:
# 
# Single file processing:
# python face_classifier.py fragment.ply
# python face_classifier.py fragment.ply --format json --output ./results
#
# Folder processing:
# python face_classifier.py ./ply_fragments --folder
# python face_classifier.py ./ply_fragments --folder --format json --no-viz
# python face_classifier.py ./ply_fragments --folder --show-individual
#
# Load and visualize existing results:
# python face_classifier.py --load fragment_analysis.pkl
# python face_classifier.py --load ./results/batch_summary.json
#
# Advanced options:
# python face_classifier.py ./fragments --folder --downsample 0.8 --eps 0.12 --output ./analysis_results