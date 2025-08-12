# Example usage:
#
# Basic usage - match break surfaces in a results folder:
# python break_surface_matcher.py ./fragment_results
#
# With custom similarity threshold and visualization:
# python break_surface_matcher.py ./fragment_results --min-similarity 0.7 --visualize
#
# Disable normal similarity (focus on shape and size):
# python break_surface_matcher.py ./fragment_results --no-normal --visualize-count 5
#
# Custom weight configuration (more emphasis on shape):
# python break_surface_matcher.py ./fragment_results --weight-shape 0.6 --weight-size 0.3 --weight-normal 0.1
#
# Visualize specific match:
# python break_surface_matcher.py ./fragment_results --visualize-match 1
#
# Fragment overview visualization:
# python break_surface_matcher.py ./fragment_results --fragment-overview
#
# Visualize specific fragments:
# python break_surface_matcher.py ./fragment_results --visualize-fragments frag_3_cell_01 frag_3_cell_06
#
# Complete analysis with visualization:
# python break_surface_matcher.py ./fragment_results --visualize --visualize-count 5 --show-matches 20
#
# Shape-focused matching (no normals):
# python break_surface_matcher.py ./fragment_results --no-normal --weight-shape 0.7 --weight-size 0.3import numpy as np
import open3d as o3d
import pickle
import json
from pathlib import Path
import argparse
import os
import glob
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BreakSurfaceMatcher:
    def __init__(self, normal_threshold=0.1, shape_threshold=0.3, size_threshold=0.5,
                 use_normal_similarity=True, weights=None):
        """
        Initialize break surface matcher.
        
        Args:
            normal_threshold: Maximum angle difference for normal similarity (radians)
            shape_threshold: Maximum shape descriptor difference for shape similarity
            size_threshold: Maximum relative size difference for size similarity (0-1)
            use_normal_similarity: Whether to include normal similarity in matching
            weights: Dictionary of weights for different similarity components
        """
        self.normal_threshold = normal_threshold
        self.shape_threshold = shape_threshold  
        self.size_threshold = size_threshold
        self.use_normal_similarity = use_normal_similarity
        
        # Default weights - can be customized
        if weights is None:
            if use_normal_similarity:
                self.weights = {
                    'normal': 0.4,    # Normal alignment
                    'size': 0.25,     # Size matching  
                    'shape': 0.25,    # Shape characteristics
                    'moment': 0.1     # Geometric moments
                }
            else:
                self.weights = {
                    'normal': 0.0,    # Skip normal similarity
                    'size': 0.4,      # Increased size weight
                    'shape': 0.5,     # Increased shape weight
                    'moment': 0.1     # Geometric moments
                }
        else:
            self.weights = weights
            
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        # Storage for loaded fragments and their break surfaces
        self.fragments = {}
        self.break_surfaces = {}
        self.surface_descriptors = {}
        
    def load_fragment_results(self, results_file):
        """Load a single fragment analysis result."""
        try:
            file_path = Path(results_file)
            
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    results = pickle.load(f)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    results = json.load(f)
            else:
                print(f"Unsupported file format: {file_path.suffix}")
                return None
                
            fragment_name = file_path.stem.replace('_analysis', '')
            return fragment_name, results
            
        except Exception as e:
            print(f"Error loading {results_file}: {e}")
            return None, None
    
    def load_all_fragments(self, results_folder):
        """Load all fragment analysis results from a folder."""
        results_path = Path(results_folder)
        
        # Find all analysis files
        analysis_files = []
        analysis_files.extend(results_path.glob("*_analysis.pkl"))
        analysis_files.extend(results_path.glob("*_analysis.json"))
        
        if not analysis_files:
            print(f"No analysis files found in {results_folder}")
            return False
        
        print(f"Found {len(analysis_files)} analysis files")
        
        # Load each fragment
        for file_path in analysis_files:
            fragment_name, results = self.load_fragment_results(file_path)
            
            if results is not None:
                self.fragments[fragment_name] = results
                print(f"Loaded fragment: {fragment_name}")
            
        print(f"Successfully loaded {len(self.fragments)} fragments")
        return len(self.fragments) > 0
    
    def extract_break_surfaces(self):
        """Extract break surfaces from all loaded fragments."""
        print("\nExtracting break surfaces from fragments...")
        
        for fragment_name, results in self.fragments.items():
            fragment_break_surfaces = {}
            
            # Get point cloud data
            if 'point_cloud' in results:
                points = np.array(results['point_cloud']['points'])
                normals = np.array(results['point_cloud']['normals']) if results['point_cloud']['normals'] else None
            else:
                print(f"Warning: No point cloud data found for {fragment_name}")
                continue
                
            labels = np.array(results['labels'])
            face_classifications = results['face_classifications']
            
            # Extract each break surface
            for face_id, classification in face_classifications.items():
                if classification['type'] == 'break':
                    face_id = int(face_id)  # Ensure integer
                    face_mask = labels == face_id
                    
                    if np.sum(face_mask) == 0:
                        continue
                        
                    break_surface = {
                        'points': points[face_mask],
                        'normals': normals[face_mask] if normals is not None else None,
                        'roughness_metrics': classification['roughness_metrics'],
                        'point_count': np.sum(face_mask),
                        'face_id': face_id
                    }
                    
                    surface_key = f"{fragment_name}_face_{face_id}"
                    fragment_break_surfaces[face_id] = break_surface
                    
            self.break_surfaces[fragment_name] = fragment_break_surfaces
            print(f"  {fragment_name}: {len(fragment_break_surfaces)} break surfaces")
        
        total_break_surfaces = sum(len(surfaces) for surfaces in self.break_surfaces.values())
        print(f"\nTotal break surfaces extracted: {total_break_surfaces}")
        
    def compute_surface_descriptors(self):
        """Compute descriptors for all break surfaces."""
        print("\nComputing surface descriptors...")
        
        for fragment_name, surfaces in self.break_surfaces.items():
            fragment_descriptors = {}
            
            for face_id, surface in surfaces.items():
                descriptor = self._compute_single_surface_descriptor(surface)
                fragment_descriptors[face_id] = descriptor
                
            self.surface_descriptors[fragment_name] = fragment_descriptors
            print(f"  {fragment_name}: computed descriptors for {len(fragment_descriptors)} surfaces")
    
    def _compute_single_surface_descriptor(self, surface):
        """Compute comprehensive descriptor for a single break surface."""
        points = surface['points']
        normals = surface['normals']
        
        if len(points) < 3:
            return None
            
        descriptor = {}
        
        # 1. Basic geometric properties
        descriptor['point_count'] = len(points)
        descriptor['centroid'] = np.mean(points, axis=0)
        
        # 2. Size descriptors
        # Bounding box
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        bbox_size = max_bounds - min_bounds
        descriptor['bbox_volume'] = np.prod(bbox_size)
        descriptor['bbox_diagonal'] = np.linalg.norm(bbox_size)
        descriptor['bbox_dimensions'] = bbox_size
        
        # Surface area estimate (using convex hull)
        try:
            if len(points) >= 4:
                hull = ConvexHull(points)
                descriptor['convex_area'] = hull.area
                descriptor['convex_volume'] = hull.volume
            else:
                descriptor['convex_area'] = 0
                descriptor['convex_volume'] = 0
        except:
            descriptor['convex_area'] = 0
            descriptor['convex_volume'] = 0
        
        # 3. Shape descriptors using PCA
        try:
            pca = PCA()
            pca.fit(points)
            descriptor['principal_axes'] = pca.components_
            descriptor['explained_variance'] = pca.explained_variance_
            descriptor['explained_variance_ratio'] = pca.explained_variance_ratio_
            
            # Shape ratios from eigenvalues
            eig_vals = pca.explained_variance_
            if len(eig_vals) >= 3 and eig_vals[0] > 1e-10:
                descriptor['planarity'] = (eig_vals[1] - eig_vals[2]) / eig_vals[0]
                descriptor['sphericity'] = eig_vals[2] / eig_vals[0]
                descriptor['linearity'] = (eig_vals[0] - eig_vals[1]) / eig_vals[0]
            else:
                descriptor['planarity'] = 0
                descriptor['sphericity'] = 0  
                descriptor['linearity'] = 0
        except:
            descriptor['principal_axes'] = np.eye(3)
            descriptor['explained_variance'] = np.array([1, 0, 0])
            descriptor['explained_variance_ratio'] = np.array([1, 0, 0])
            descriptor['planarity'] = 0
            descriptor['sphericity'] = 0
            descriptor['linearity'] = 0
        
        # 4. Normal-based descriptors
        if normals is not None and len(normals) > 0:
            # Average normal
            avg_normal = np.mean(normals, axis=0)
            avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-10)
            descriptor['average_normal'] = avg_normal
            
            # Normal variation
            normal_dots = np.array([np.dot(n, avg_normal) for n in normals])
            descriptor['normal_consistency'] = np.mean(normal_dots)
            descriptor['normal_std'] = np.std(normal_dots)
        else:
            # Estimate normal from PCA
            descriptor['average_normal'] = descriptor['principal_axes'][2]  # Third component (smallest variance)
            descriptor['normal_consistency'] = 1.0
            descriptor['normal_std'] = 0.0
        
        # 5. Geometric moments (for shape characterization)
        centered_points = points - descriptor['centroid']
        
        # Second moments
        descriptor['moment_xx'] = np.mean(centered_points[:, 0]**2)
        descriptor['moment_yy'] = np.mean(centered_points[:, 1]**2)
        descriptor['moment_zz'] = np.mean(centered_points[:, 2]**2)
        descriptor['moment_xy'] = np.mean(centered_points[:, 0] * centered_points[:, 1])
        descriptor['moment_xz'] = np.mean(centered_points[:, 0] * centered_points[:, 2])
        descriptor['moment_yz'] = np.mean(centered_points[:, 1] * centered_points[:, 2])
        
        # 6. Boundary/edge characteristics
        if len(points) >= 4:
            try:
                # Project to 2D plane for boundary analysis
                primary_axis = descriptor['principal_axes'][0]
                secondary_axis = descriptor['principal_axes'][1]
                
                # Project points to 2D
                proj_2d = np.column_stack([
                    np.dot(centered_points, primary_axis),
                    np.dot(centered_points, secondary_axis)
                ])
                
                # Convex hull in 2D
                hull_2d = ConvexHull(proj_2d)
                descriptor['perimeter'] = hull_2d.area  # In 2D, area gives perimeter
                descriptor['boundary_points'] = len(hull_2d.vertices)
                
                # Circularity (compactness)
                if descriptor['perimeter'] > 0:
                    descriptor['circularity'] = 4 * np.pi * hull_2d.volume / (descriptor['perimeter']**2)
                else:
                    descriptor['circularity'] = 0
                    
            except:
                descriptor['perimeter'] = 0
                descriptor['boundary_points'] = 0
                descriptor['circularity'] = 0
        else:
            descriptor['perimeter'] = 0
            descriptor['boundary_points'] = 0
            descriptor['circularity'] = 0
        
        return descriptor
    
    def compare_surfaces(self, desc1, desc2):
        """Compare two surface descriptors and return similarity scores."""
        if desc1 is None or desc2 is None:
            return {'overall_similarity': 0, 'normal_similarity': 0, 'shape_similarity': 0, 'size_similarity': 0}
        
        similarities = {}
        
        # 1. Normal similarity
        normal1 = desc1['average_normal']
        normal2 = desc2['average_normal']
        
        # Consider both parallel and anti-parallel normals (break surfaces can face opposite directions)
        dot_product = np.abs(np.dot(normal1, normal2))
        dot_product = np.clip(dot_product, 0, 1)
        angle = np.arccos(dot_product)
        similarities['normal_similarity'] = 1.0 - (angle / (np.pi/2))  # Normalize to 0-1
        
        # 2. Size similarity
        size1 = desc1['bbox_diagonal']
        size2 = desc2['bbox_diagonal']
        
        if size1 > 0 and size2 > 0:
            size_ratio = min(size1, size2) / max(size1, size2)
            similarities['size_similarity'] = size_ratio
        else:
            similarities['size_similarity'] = 0
        
        # Point count similarity
        count1 = desc1['point_count']
        count2 = desc2['point_count']
        count_ratio = min(count1, count2) / max(count1, count2)
        
        # Area similarity
        area1 = desc1['convex_area']
        area2 = desc2['convex_area']
        if area1 > 0 and area2 > 0:
            area_ratio = min(area1, area2) / max(area1, area2)
        else:
            area_ratio = count_ratio  # Fallback to point count ratio
        
        # Combined size similarity
        similarities['size_similarity'] = (size_ratio + count_ratio + area_ratio) / 3
        
        # 3. Shape similarity
        shape_features1 = np.array([
            desc1['planarity'],
            desc1['sphericity'], 
            desc1['linearity'],
            desc1['circularity'],
            desc1['explained_variance_ratio'][0] if len(desc1['explained_variance_ratio']) > 0 else 0,
            desc1['explained_variance_ratio'][1] if len(desc1['explained_variance_ratio']) > 1 else 0,
        ])
        
        shape_features2 = np.array([
            desc2['planarity'],
            desc2['sphericity'],
            desc2['linearity'], 
            desc2['circularity'],
            desc2['explained_variance_ratio'][0] if len(desc2['explained_variance_ratio']) > 0 else 0,
            desc2['explained_variance_ratio'][1] if len(desc2['explained_variance_ratio']) > 1 else 0,
        ])
        
        # Euclidean distance in shape feature space
        shape_distance = np.linalg.norm(shape_features1 - shape_features2)
        similarities['shape_similarity'] = 1.0 / (1.0 + shape_distance)
        
        # 4. Geometric moments similarity
        moments1 = np.array([desc1['moment_xx'], desc1['moment_yy'], desc1['moment_zz'],
                            desc1['moment_xy'], desc1['moment_xz'], desc1['moment_yz']])
        moments2 = np.array([desc2['moment_xx'], desc2['moment_yy'], desc2['moment_zz'], 
                            desc2['moment_xy'], desc2['moment_xz'], desc2['moment_yz']])
        
        # Normalize moments by size
        moments1 = moments1 / (desc1['bbox_diagonal']**2 + 1e-10)
        moments2 = moments2 / (desc2['bbox_diagonal']**2 + 1e-10)
        
        moment_distance = np.linalg.norm(moments1 - moments2)
        similarities['moment_similarity'] = 1.0 / (1.0 + moment_distance)
        
        # 5. Overall similarity (weighted combination)
        similarities['overall_similarity'] = (
            self.weights['normal'] * similarities['normal_similarity'] +
            self.weights['size'] * similarities['size_similarity'] +
            self.weights['shape'] * similarities['shape_similarity'] + 
            self.weights['moment'] * similarities['moment_similarity']
        )
        
        return similarities
    
    def find_matching_break_surfaces(self, min_overall_similarity=0.6):
        """Find matching break surfaces across all fragments."""
        print(f"\nFinding matching break surfaces (min similarity: {min_overall_similarity:.2f})...")
        
        # Create list of all surfaces with fragment info
        all_surfaces = []
        surface_keys = []
        
        for fragment_name, surfaces in self.surface_descriptors.items():
            for face_id, descriptor in surfaces.items():
                if descriptor is not None:
                    all_surfaces.append(descriptor)
                    surface_keys.append((fragment_name, face_id))
        
        print(f"Comparing {len(all_surfaces)} break surfaces...")
        
        # Compare all pairs
        matches = []
        comparison_count = 0
        
        for i in range(len(all_surfaces)):
            for j in range(i + 1, len(all_surfaces)):
                fragment1, face1 = surface_keys[i]
                fragment2, face2 = surface_keys[j]
                
                # Skip surfaces from the same fragment
                if fragment1 == fragment2:
                    continue
                
                desc1 = all_surfaces[i]
                desc2 = all_surfaces[j]
                
                similarities = self.compare_surfaces(desc1, desc2)
                comparison_count += 1
                
                if similarities['overall_similarity'] >= min_overall_similarity:
                    match = {
                        'fragment1': fragment1,
                        'face1': face1,
                        'fragment2': fragment2, 
                        'face2': face2,
                        'similarities': similarities,
                        'surface1_info': {
                            'point_count': desc1['point_count'],
                            'area': desc1['convex_area'],
                            'size': desc1['bbox_diagonal']
                        },
                        'surface2_info': {
                            'point_count': desc2['point_count'],
                            'area': desc2['convex_area'], 
                            'size': desc2['bbox_diagonal']
                        }
                    }
                    matches.append(match)
        
        print(f"Made {comparison_count} pairwise comparisons")
        print(f"Found {len(matches)} potential matches")
        
        # Sort matches by similarity
        matches.sort(key=lambda x: x['similarities']['overall_similarity'], reverse=True)
        
        return matches
    
    def print_matches(self, matches, top_n=None):
        """Print matching break surfaces in a formatted way."""
        if not matches:
            print("No matches found.")
            return
        
        if top_n is not None:
            matches = matches[:top_n]
            print(f"\nTop {len(matches)} Break Surface Matches:")
        else:
            print(f"\nAll {len(matches)} Break Surface Matches:")
        
        print("="*100)
        
        for i, match in enumerate(matches, 1):
            sim = match['similarities']
            
            print(f"\nMatch #{i}:")
            print(f"  Fragments: {match['fragment1']} (face {match['face1']}) ↔ {match['fragment2']} (face {match['face2']})")
            print(f"  Overall Similarity: {sim['overall_similarity']:.3f}")
            print(f"  Normal Similarity:  {sim['normal_similarity']:.3f}")
            print(f"  Size Similarity:    {sim['size_similarity']:.3f}")
            print(f"  Shape Similarity:   {sim['shape_similarity']:.3f}")
            
            # Surface information
            s1 = match['surface1_info']
            s2 = match['surface2_info']
            print(f"  Surface 1: {s1['point_count']} points, area={s1['area']:.3f}, size={s1['size']:.3f}")
            print(f"  Surface 2: {s2['point_count']} points, area={s2['area']:.3f}, size={s2['size']:.3f}")
            print("-" * 80)
    
    def save_matches(self, matches, output_file, format='json'):
        """Save matches to file."""
        output_path = Path(output_file)
        
        # Prepare serializable data
        serializable_matches = []
        for match in matches:
            serializable_match = {}
            for key, value in match.items():
                if isinstance(value, dict):
                    serializable_match[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                             for k, v in value.items()}
                else:
                    serializable_match[key] = value
            serializable_matches.append(serializable_match)
        
        # Add metadata
        match_data = {
            'metadata': {
                'total_matches': len(matches),
                'fragments_analyzed': list(self.fragments.keys()),
                'total_break_surfaces': sum(len(surfaces) for surfaces in self.break_surfaces.values()),
                'matching_parameters': {
                    'normal_threshold': self.normal_threshold,
                    'shape_threshold': self.shape_threshold, 
                    'size_threshold': self.size_threshold
                },
                'timestamp': time.time()
            },
            'matches': serializable_matches
        }
        
        if format.lower() == 'json':
            output_path = output_path.with_suffix('.json')
            with open(output_path, 'w') as f:
                json.dump(match_data, f, indent=2)
        elif format.lower() == 'pkl':
            output_path = output_path.with_suffix('.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(match_data, f)
        
        print(f"Matches saved to: {output_path}")
        return output_path
    
    def visualize_match(self, match, save_image=None, show_normals=False):
        """Visualize a specific match by showing the two break surfaces."""
        fragment1 = match['fragment1']
        face1 = match['face1']
        fragment2 = match['fragment2']
        face2 = match['face2']
        
        # Get surface data
        surface1 = self.break_surfaces[fragment1][face1]
        surface2 = self.break_surfaces[fragment2][face2]
        
        # Create point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(surface1['points'])
        pcd1.paint_uniform_color([1, 0, 0])  # Red
        
        pcd2 = o3d.geometry.PointCloud() 
        pcd2.points = o3d.utility.Vector3dVector(surface2['points'])
        pcd2.paint_uniform_color([0, 0, 1])  # Blue
        
        # Add normals if available and requested
        if show_normals and surface1['normals'] is not None:
            pcd1.normals = o3d.utility.Vector3dVector(surface1['normals'])
        if show_normals and surface2['normals'] is not None:
            pcd2.normals = o3d.utility.Vector3dVector(surface2['normals'])
        
        # Offset second surface for better visualization
        bbox1 = pcd1.get_axis_aligned_bounding_box()
        bbox2 = pcd2.get_axis_aligned_bounding_box()
        offset = (bbox1.max_bound - bbox1.min_bound)[0] + (bbox2.max_bound - bbox2.min_bound)[0]
        pcd2.translate([offset * 1.5, 0, 0])
        
        # Create coordinate frames for better visualization
        coord1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=bbox1.get_center())
        coord2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=bbox2.get_center() + np.array([offset * 1.5, 0, 0]))
        
        print(f"\nVisualizing match between:")
        print(f"  RED: {fragment1} face {face1}")
        print(f"  BLUE: {fragment2} face {face2}")
        print(f"  Overall similarity: {match['similarities']['overall_similarity']:.3f}")
        print(f"  Normal similarity: {match['similarities']['normal_similarity']:.3f}")
        print(f"  Shape similarity: {match['similarities']['shape_similarity']:.3f}")
        print(f"  Size similarity: {match['similarities']['size_similarity']:.3f}")
        
        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Break Surface Match - Similarity: {match['similarities']['overall_similarity']:.3f}", 
                         width=1400, height=900)
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)
        vis.add_geometry(coord1)
        vis.add_geometry(coord2)
        
        # Set viewing options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.05, 0.05, 0.05])
        render_option.point_size = 3.0
        
        # Get view control and set a good viewing angle
        view_control = vis.get_view_control()
        view_control.set_zoom(0.7)
        
        vis.run()
        vis.destroy_window()
    
    def visualize_multiple_matches(self, matches, num_matches=5, save_images=False, output_dir=None):
        """Visualize multiple matches in sequence."""
        if not matches:
            print("No matches to visualize.")
            return
            
        num_matches = min(num_matches, len(matches))
        print(f"\nVisualizing top {num_matches} matches...")
        
        for i in range(num_matches):
            match = matches[i]
            print(f"\n{'='*60}")
            print(f"Match {i+1}/{num_matches}")
            print(f"{'='*60}")
            
            if save_images and output_dir:
                image_path = Path(output_dir) / f"match_{i+1:02d}.png"
                self.visualize_match(match, save_image=str(image_path))
            else:
                self.visualize_match(match)
                
            if i < num_matches - 1:
                input("Press Enter to view next match (or Ctrl+C to stop)...")
    
    def visualize_fragment_overview(self, fragment_names=None, max_fragments=4):
        """Visualize multiple fragments with their break surfaces highlighted."""
        if fragment_names is None:
            fragment_names = list(self.fragments.keys())[:max_fragments]
        elif isinstance(fragment_names, str):
            fragment_names = [fragment_names]
            
        fragment_names = fragment_names[:max_fragments]
        
        print(f"Visualizing {len(fragment_names)} fragments with break surfaces...")
        
        # Colors for different fragments
        colors = np.array([
            [1.0, 0.0, 0.0],    # Red
            [0.0, 0.0, 1.0],    # Blue  
            [0.0, 0.8, 0.0],    # Green
            [1.0, 0.5, 0.0],    # Orange
            [0.8, 0.0, 0.8],    # Magenta
            [0.0, 0.8, 0.8],    # Cyan
        ])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Fragment Overview - Break Surfaces", width=1600, height=1000)
        
        for i, fragment_name in enumerate(fragment_names):
            if fragment_name not in self.fragments:
                continue
                
            # Get original point cloud
            results = self.fragments[fragment_name]
            if 'point_cloud' in results:
                points = np.array(results['point_cloud']['points'])
            else:
                continue
                
            labels = np.array(results['labels'])
            face_classifications = results['face_classifications']
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Color all points gray initially
            point_colors = np.full((len(points), 3), 0.3)  # Dark gray
            
            # Color break surfaces with fragment-specific color
            fragment_color = colors[i % len(colors)]
            
            for face_id, classification in face_classifications.items():
                if classification['type'] == 'break':
                    face_id = int(face_id)
                    face_mask = labels == face_id
                    point_colors[face_mask] = fragment_color
            
            pcd.colors = o3d.utility.Vector3dVector(point_colors)
            
            # Offset fragments for better visualization
            if i > 0:
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox_size = bbox.max_bound - bbox.min_bound
                offset = np.array([bbox_size[0] * 1.2 * i, 0, 0])
                pcd.translate(offset)
            
            vis.add_geometry(pcd)
            
            # Add text label (approximated with coordinate frame)
            if i == 0:
                bbox_center = pcd.get_axis_aligned_bounding_box().get_center()
                coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0, origin=bbox_center)
                vis.add_geometry(coord)
        
        # Set viewing options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.05, 0.05, 0.05])
        render_option.point_size = 2.0
        
        view_control = vis.get_view_control()
        view_control.set_zoom(0.6)
        
        print("Fragment colors:")
        for i, name in enumerate(fragment_names):
            color_name = ['Red', 'Blue', 'Green', 'Orange', 'Magenta', 'Cyan'][i % 6]
            print(f"  {name}: {color_name}")
        print("\nGray: Non-break surfaces")
        print("Colored: Break surfaces")
        
        vis.run()
        vis.destroy_window()
    
    def create_match_summary_plot(self, matches, output_file=None):
        """Create a summary plot of match statistics."""
        if not matches:
            print("No matches to plot.")
            return
        
        # Extract similarity scores
        overall_sims = [m['similarities']['overall_similarity'] for m in matches]
        normal_sims = [m['similarities']['normal_similarity'] for m in matches]
        size_sims = [m['similarities']['size_similarity'] for m in matches]
        shape_sims = [m['similarities']['shape_similarity'] for m in matches]
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Break Surface Match Analysis ({len(matches)} matches)', fontsize=16)
        
        # Overall similarity distribution
        axes[0,0].hist(overall_sims, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[0,0].set_title('Overall Similarity Distribution')
        axes[0,0].set_xlabel('Similarity Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(np.mean(overall_sims), color='red', linestyle='--', label=f'Mean: {np.mean(overall_sims):.3f}')
        axes[0,0].legend()
        
        # Component similarities
        axes[0,1].boxplot([normal_sims, size_sims, shape_sims], 
                         labels=['Normal', 'Size', 'Shape'])
        axes[0,1].set_title('Similarity Components')
        axes[0,1].set_ylabel('Similarity Score')
        
        # Similarity correlation
        axes[1,0].scatter(normal_sims, overall_sims, alpha=0.6, c='blue')
        axes[1,0].set_xlabel('Normal Similarity')
        axes[1,0].set_ylabel('Overall Similarity')
        axes[1,0].set_title('Normal vs Overall Similarity')
        
        # Size vs Shape similarity
        axes[1,1].scatter(size_sims, shape_sims, alpha=0.6, c='orange')
        axes[1,1].set_xlabel('Size Similarity')
        axes[1,1].set_ylabel('Shape Similarity')
        axes[1,1].set_title('Size vs Shape Similarity')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Match summary plot saved to: {output_file}")
        
        plt.show()
    
    def process_results_folder(self, results_folder, output_folder=None, 
                              min_similarity=0.6, save_format='json', 
                              show_top_matches=10, create_plots=True,
                              visualize_matches=False, num_visualize=3):
        """
        Complete processing pipeline for a folder of fragment analysis results.
        
        Args:
            results_folder: Folder containing fragment analysis results
            output_folder: Where to save match results (default: results_folder + '_matches')
            min_similarity: Minimum similarity threshold for matches
            save_format: 'json' or 'pkl'
            show_top_matches: Number of top matches to display
            create_plots: Whether to create summary plots
            visualize_matches: Whether to show 3D visualizations of top matches
            num_visualize: Number of top matches to visualize
        """
        print("="*80)
        print("BREAK SURFACE MATCHING PIPELINE")
        print("="*80)
        print(f"Similarity weights: Normal={self.weights['normal']:.2f}, Size={self.weights['size']:.2f}, Shape={self.weights['shape']:.2f}, Moment={self.weights['moment']:.2f}")
        print(f"Using normal similarity: {self.use_normal_similarity}")
        
        # Load all fragment results
        if not self.load_all_fragments(results_folder):
            print("Failed to load fragments. Exiting.")
            return None
        
        # Extract break surfaces
        self.extract_break_surfaces()
        
        if not self.break_surfaces:
            print("No break surfaces found. Exiting.")
            return None
        
        # Compute descriptors
        self.compute_surface_descriptors()
        
        # Find matches
        matches = self.find_matching_break_surfaces(min_similarity)
        
        if not matches:
            print(f"No matches found with similarity >= {min_similarity}")
            return None
        
        # Set up output folder
        if output_folder is None:
            output_folder = f"{results_folder}_matches"
        
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Save matches
        match_file = output_path / f"break_surface_matches.{save_format}"
        self.save_matches(matches, match_file, save_format)
        
        # Print summary
        self.print_matches(matches, top_n=show_top_matches)
        
        # Create plots if requested
        if create_plots:
            plot_file = output_path / "match_summary.png"
            self.create_match_summary_plot(matches, plot_file)
        
        # Visualize matches if requested
        if visualize_matches and matches:
            print(f"\n{'='*60}")
            print("3D VISUALIZATION OF TOP MATCHES")
            print(f"{'='*60}")
            self.visualize_multiple_matches(matches, num_matches=min(num_visualize, len(matches)))
        
        print(f"\nProcessing complete!")
        print(f"Results saved to: {output_path}")
        
        return matches


def main():
    """Main function for break surface matching."""
    parser = argparse.ArgumentParser(description='Match break surfaces across stone fragments')
    
    # Input
    parser.add_argument('results_folder', help='Folder containing fragment analysis results')
    parser.add_argument('--output', help='Output folder for match results (default: auto-generated)')
    
    # Matching parameters  
    parser.add_argument('--min-similarity', type=float, default=0.6,
                       help='Minimum overall similarity for matches (0-1, default: 0.6)')
    parser.add_argument('--normal-threshold', type=float, default=0.1,
                       help='Normal similarity threshold in radians (default: 0.1)')
    parser.add_argument('--shape-threshold', type=float, default=0.3,
                       help='Shape similarity threshold (default: 0.3)')
    parser.add_argument('--size-threshold', type=float, default=0.5, 
                       help='Size similarity threshold (default: 0.5)')
    
    # Similarity weighting options
    parser.add_argument('--no-normal', action='store_true',
                       help='Disable normal similarity (focus on shape and size only)')
    parser.add_argument('--weight-normal', type=float, default=None,
                       help='Weight for normal similarity (default: 0.4 if enabled, 0.0 if disabled)')
    parser.add_argument('--weight-size', type=float, default=None,
                       help='Weight for size similarity (default: auto-calculated)')
    parser.add_argument('--weight-shape', type=float, default=None,
                       help='Weight for shape similarity (default: auto-calculated)')
    parser.add_argument('--weight-moment', type=float, default=None,
                       help='Weight for geometric moments (default: 0.1)')
    
    # Output options
    parser.add_argument('--format', choices=['json', 'pkl'], default='json',
                       help='Output format for results (default: json)')
    parser.add_argument('--show-matches', type=int, default=10,
                       help='Number of top matches to display (default: 10)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating summary plots')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Show 3D visualizations of top matches')
    parser.add_argument('--visualize-count', type=int, default=3,
                       help='Number of top matches to visualize (default: 3)')
    parser.add_argument('--visualize-match', type=int,
                       help='Visualize specific match by index (1-based)')
    parser.add_argument('--visualize-fragments', nargs='*',
                       help='Visualize specific fragments with break surfaces')
    parser.add_argument('--fragment-overview', action='store_true',
                       help='Show overview of all fragments with break surfaces')
    
    args = parser.parse_args()
    
    # Check if results folder exists
    if not os.path.exists(args.results_folder):
        print(f"Error: Results folder {args.results_folder} not found")
        return
    
    # Set up custom weights if provided
    custom_weights = None
    use_normal = not args.no_normal
    
    if any([args.weight_normal is not None, args.weight_size is not None, 
            args.weight_shape is not None, args.weight_moment is not None]):
        custom_weights = {}
        if args.weight_normal is not None:
            custom_weights['normal'] = args.weight_normal
            use_normal = args.weight_normal > 0
        if args.weight_size is not None:
            custom_weights['size'] = args.weight_size
        if args.weight_shape is not None:
            custom_weights['shape'] = args.weight_shape
        if args.weight_moment is not None:
            custom_weights['moment'] = args.weight_moment
    
    # Initialize matcher
    matcher = BreakSurfaceMatcher(
        normal_threshold=args.normal_threshold,
        shape_threshold=args.shape_threshold,
        size_threshold=args.size_threshold,
        use_normal_similarity=use_normal,
        weights=custom_weights
    )
    
    # Handle fragment overview visualization
    if args.fragment_overview:
        print("Loading fragments for overview...")
        if matcher.load_all_fragments(args.results_folder):
            matcher.extract_break_surfaces()
            if args.visualize_fragments:
                matcher.visualize_fragment_overview(args.visualize_fragments)
            else:
                matcher.visualize_fragment_overview()
        return
    
    # Process the results folder
    matches = matcher.process_results_folder(
        results_folder=args.results_folder,
        output_folder=args.output,
        min_similarity=args.min_similarity,
        save_format=args.format,
        show_top_matches=args.show_matches,
        create_plots=not args.no_plots,
        visualize_matches=args.visualize,
        num_visualize=args.visualize_count
    )
    
    # Handle specific match visualization
    if args.visualize_match and matches:
        if 1 <= args.visualize_match <= len(matches):
            match_to_show = matches[args.visualize_match - 1]
            print(f"\n{'='*60}")
            print(f"VISUALIZING SPECIFIC MATCH #{args.visualize_match}")
            print(f"{'='*60}")
            matcher.visualize_match(match_to_show)
        else:
            print(f"Invalid match index. Available matches: 1-{len(matches)}")
    
    # Handle fragment-specific visualization
    if args.visualize_fragments and matches:
        print(f"\n{'='*60}")
        print(f"VISUALIZING SPECIFIC FRAGMENTS")
        print(f"{'='*60}")
        matcher.visualize_fragment_overview(args.visualize_fragments)


if __name__ == "__main__":
    main()


# Example usage:
#
# Basic usage - match break surfaces in a results folder:
# python break_surface_matcher.py ./fragment_results
#
# With custom similarity threshold:
# python break_surface_matcher.py ./fragment_results --min-similarity 0.7
#
# Save as pickle format and show more matches:
# python break_surface_matcher.py ./fragment_results --format pkl --show-matches 20
#
# Custom output location:
# python break_surface_matcher.py ./fragment_results --output ./surface_matches
#
# Visualize the best match:
# python break_surface_matcher.py ./fragment_results --visualize-match 1
#
# Advanced parameters:
# python break_surface_matcher.py ./fragment_results --min-similarity 0.65 --normal-threshold 0.15 --shape-threshold 0.25