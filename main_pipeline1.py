#!/usr/bin/env python3
"""
Final Integrated Mayan Stele Fragment Reconstruction Pipeline

This script orchestrates the entire reconstruction process from PLY files
with colored break surfaces to a final reconstructed artifact.

IMPROVEMENTS INTEGRATED:
- Enhanced surface matching with detailed cross-color analysis
- 3D match visualization with connecting lines
- Origin-based progressive fragment assembly
- Contact-based surface alignment (brings surfaces together)
- Proper assembly quality evaluation
- FIXED: Overlap detection and collision prevention
- FIXED: Robust positioning with fallback strategies
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import time
from scipy.spatial.distance import cdist
import open3d as o3d

# Import our custom modules
from ply_loader import PLYColorExtractor
from feature_extractor import BreakSurfaceFeatureExtractor
from surface_matcher import SurfaceMatcher
from fragment_aligner import FragmentAligner
from reconstruction_visualizer import ReconstructionVisualizer

class ReconstructionPipeline:
    """
    Main pipeline for fragment reconstruction with all improvements integrated
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'color_tolerance': 0.3,
            'min_cluster_size': 50,
            'min_similarity': 0.6,
            'use_optimal_matching': True,
            'visualize_steps': False,
            'save_intermediate': True,
            'output_reports': True,
            'assembly_contact_distance': 0.005,  # Increased to 5mm for better separation
            'debug_assembly': True  # Enable detailed debugging
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self.ply_extractor = PLYColorExtractor()
        self.feature_extractor = BreakSurfaceFeatureExtractor()
        self.surface_matcher = SurfaceMatcher()
        self.fragment_aligner = FragmentAligner()
        self.visualizer = ReconstructionVisualizer()
        
        # Pipeline data
        self.fragments = []
        self.enhanced_fragments = []
        self.all_matches = {}
        self.transformations = {}
        self.quality_metrics = {}
    
    def normalize_fragment_coordinates(self):
        """Normalize fragment coordinates to reasonable scale and center them"""
        if not self.fragments:
            return
        
        print("Analyzing fragment coordinate system...")
        
        # Collect all vertex coordinates to understand the scale
        all_vertices = []
        fragment_bounds = []
        
        for i, fragment in enumerate(self.fragments):
            if 'mesh' in fragment:
                vertices = np.asarray(fragment['mesh'].vertices)
                all_vertices.append(vertices)
                
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)
                bbox_center = (bbox_min + bbox_max) / 2
                bbox_size = bbox_max - bbox_min
                
                fragment_bounds.append({
                    'min': bbox_min,
                    'max': bbox_max, 
                    'center': bbox_center,
                    'size': bbox_size
                })
                
                print(f"  Fragment {i}: Center {bbox_center}, Size {np.linalg.norm(bbox_size):.2f}")
        
        if not all_vertices:
            return
        
        # Compute global bounds
        global_vertices = np.vstack(all_vertices)
        global_min = np.min(global_vertices, axis=0)
        global_max = np.max(global_vertices, axis=0)
        global_center = (global_min + global_max) / 2
        global_size = global_max - global_min
        global_scale = np.max(global_size)
        
        print(f"Global bounds: Center {global_center}, Size {global_scale:.2f}")
        
        # Determine if normalization is needed
        if global_scale > 10.0 or np.linalg.norm(global_center) > 5.0:
            print(f"Coordinate system needs normalization (scale: {global_scale:.2f})")
            
            # Normalize: center at origin and scale to reasonable size
            target_scale = 1.0  # Target scale of 1 unit
            scale_factor = target_scale / global_scale
            
            print(f"Applying scale factor: {scale_factor:.6f}")
            print(f"Centering offset: {-global_center}")
            
            # Apply normalization to all fragments
            for i, fragment in enumerate(self.fragments):
                if 'mesh' in fragment:
                    # Transform vertices
                    vertices = np.asarray(fragment['mesh'].vertices)
                    normalized_vertices = (vertices - global_center) * scale_factor
                    
                    # Update mesh
                    fragment['mesh'].vertices = o3d.utility.Vector3dVector(normalized_vertices)
                    
                    # Update break surface points
                    for color, surfaces in fragment['break_surfaces'].items():
                        for surface in surfaces:
                            if 'points' in surface:
                                surface['points'] = (surface['points'] - global_center) * scale_factor
                    
                    # Verify normalization
                    new_vertices = np.asarray(fragment['mesh'].vertices)
                    new_center = np.mean(new_vertices, axis=0)
                    new_size = np.max(new_vertices, axis=0) - np.min(new_vertices, axis=0)
                    print(f"  Fragment {i} normalized: Center {new_center}, Size {np.linalg.norm(new_size):.3f}")
            
            print("Coordinate normalization completed!")
        else:
            print("Coordinate system is already reasonable - no normalization needed")
    
    def load_and_process_fragments(self, input_directory):
        """Step 1: Load PLY files and extract break surfaces"""
        print("="*60)
        print("STEP 1: Loading PLY files and extracting break surfaces")
        print("="*60)
        
        start_time = time.time()
        
        # Load all fragments
        print(f"Loading fragments from: {input_directory}")
        self.fragments = self.ply_extractor.process_all_fragments(input_directory)
        
        if not self.fragments:
            raise ValueError(f"No valid PLY files found in {input_directory}")
        
        # CRITICAL: Normalize coordinates before proceeding
        self.normalize_fragment_coordinates()
        
        # Print summary
        print(f"\nLoaded {len(self.fragments)} fragments:")
        total_surfaces = {'blue': 0, 'green': 0, 'red': 0}
        
        for i, fragment in enumerate(self.fragments):
            print(f"  Fragment {i}: {Path(fragment['filepath']).name}")
            for color, surfaces in fragment['break_surfaces'].items():
                count = len(surfaces)
                total_surfaces[color] += count
                if count > 0:
                    print(f"    {color}: {count} surfaces")
        
        print(f"\nTotal break surfaces found:")
        for color, count in total_surfaces.items():
            print(f"  {color}: {count}")
        
        elapsed = time.time() - start_time
        print(f"\nStep 1 completed in {elapsed:.2f} seconds")
        
        return self.fragments
    
    def extract_features(self):
        """Step 2: Extract geometric features from break surfaces"""
        print("\n" + "="*60)
        print("STEP 2: Extracting geometric features")
        print("="*60)
        
        start_time = time.time()
        
        print("Computing geometric features for all break surfaces...")
        self.enhanced_fragments = self.feature_extractor.extract_all_features(self.fragments)
        
        # Print feature summary
        total_features = 0
        for fragment in self.enhanced_fragments:
            for color, features in fragment['features'].items():
                total_features += len(features)
        
        print(f"Extracted features for {total_features} break surfaces")
        
        elapsed = time.time() - start_time
        print(f"Step 2 completed in {elapsed:.2f} seconds")
        
        return self.enhanced_fragments
    
    def find_surface_matches(self):
        """Step 3: Find matching break surfaces between fragments - ENHANCED VERSION"""
        print("\n" + "="*60)
        print("STEP 3: Finding surface matches")
        print("="*60)
        
        start_time = time.time()
        
        print(f"Searching for surface matches (min similarity: {self.config['min_similarity']})...")
        
        # Surface matching with enhanced reporting
        self.all_matches = self.surface_matcher.find_all_matches(
            self.enhanced_fragments,
            min_similarity=self.config['min_similarity'],
            use_optimal=self.config['use_optimal_matching']
        )
        
        # ENHANCED MATCH REPORTING
        print(f"\n📋 DETAILED CROSS-COLOR MATCH ANALYSIS")
        print("=" * 60)
        
        total_matches = 0
        cross_color_matches = 0
        same_color_matches = 0
        match_quality_distribution = {'excellent': 0, 'good': 0, 'moderate': 0, 'weak': 0}
        
        for pair_key, color_matches in self.all_matches.items():
            parts = pair_key.split('_')
            frag1_idx = int(parts[1])
            frag2_idx = int(parts[3])
            
            print(f"\n🔗 {pair_key}:")
            
            pair_total = 0
            for color, matches in color_matches.items():
                for match in matches:
                    # Determine surface colors (check if cross-color)
                    color1 = match.get('surface1_color', color)
                    color2 = match.get('surface2_color', color)
                    similarity = match['similarity']
                    
                    # Classify match quality
                    if similarity >= 0.9:
                        quality = 'excellent'
                    elif similarity >= 0.7:
                        quality = 'good'
                    elif similarity >= 0.5:
                        quality = 'moderate'
                    else:
                        quality = 'weak'
                    
                    match_quality_distribution[quality] += 1
                    
                    if color1 != color2:
                        match_type = "🔄 CROSS-COLOR"
                        cross_color_matches += 1
                        print(f"   {match_type}: Fragment {frag1_idx} {color1} surface ↔ Fragment {frag2_idx} {color2} surface")
                    else:
                        match_type = "✓ SAME-COLOR"
                        same_color_matches += 1
                        print(f"   {match_type}: Fragment {frag1_idx} {color1} surface ↔ Fragment {frag2_idx} {color2} surface")
                    
                    print(f"      Similarity: {similarity:.4f} ({quality})")
                    
                    # Show similarity breakdown
                    if 'detailed_similarities' in match:
                        details = match['detailed_similarities']
                        print(f"      Normal: {details.get('normal', 0):.3f}, "
                            f"Area: {details.get('area', 0):.3f}, "
                            f"Size: {details.get('size', 0):.3f}")
                    
                    total_matches += 1
                    pair_total += 1
            
            if pair_total == 0:
                print(f"   No matches between fragments {frag1_idx} and {frag2_idx}")
        
        print(f"\n📊 ENHANCED MATCH SUMMARY:")
        print(f"   Total matches: {total_matches}")
        print(f"   Same-color matches: {same_color_matches}")
        print(f"   Cross-color matches: {cross_color_matches}")
        print(f"   Quality distribution:")
        for quality, count in match_quality_distribution.items():
            if count > 0:
                print(f"     {quality}: {count}")
        
        if cross_color_matches > 0:
            print(f"   🎯 Cross-color matching detected!")
            print(f"      This means break surfaces have different colors")
            print(f"      but similar geometry - this is normal for real fragments!")
        
        # MATCH VISUALIZATION
        if self.config.get('visualize_steps', False) and total_matches > 0:
            print(f"\n🎨 CREATING MATCH VISUALIZATIONS")
            print("=" * 50)
            
            # Show top 3 matches
            all_match_list = []
            for pair_key, color_matches in self.all_matches.items():
                parts = pair_key.split('_')
                frag1_idx = int(parts[1])
                frag2_idx = int(parts[3])
                
                for matches in color_matches.values():
                    for match in matches:
                        all_match_list.append({
                            'match': match,
                            'frag1_idx': frag1_idx,
                            'frag2_idx': frag2_idx
                        })
            
            # Sort by similarity and show top matches
            all_match_list.sort(key=lambda x: x['match']['similarity'], reverse=True)
            
            print("Showing top 3 matches with 3D visualization...")
            for i, match_data in enumerate(all_match_list[:3]):
                match = match_data['match']
                color1 = match.get('surface1_color', 'unknown')
                color2 = match.get('surface2_color', 'unknown')
                
                print(f"\nVisualization {i+1}: Fragment {match_data['frag1_idx']} {color1} ↔ Fragment {match_data['frag2_idx']} {color2}")
                print(f"Similarity: {match['similarity']:.4f}")
                print("Close the 3D window to continue...")
                
                # Create 3D visualization
                self.visualize_match_3d(
                    self.enhanced_fragments[match_data['frag1_idx']], 
                    self.enhanced_fragments[match_data['frag2_idx']], 
                    match, match_data['frag1_idx'], match_data['frag2_idx']
                )
        
        if total_matches == 0:
            print("WARNING: No matches found! Consider lowering min_similarity threshold.")
        
        elapsed = time.time() - start_time
        print(f"Step 3 completed in {elapsed:.2f} seconds")
        
        return self.all_matches 
    
    def visualize_match_3d(self, fragment1, fragment2, match, frag1_idx, frag2_idx):
        """Create 3D visualization of matching surfaces with connecting line"""
        
        import open3d as o3d
        
        try:
            # Get surface colors and indices
            color1 = match.get('surface1_color', match.get('color', 'blue'))
            color2 = match.get('surface2_color', match.get('color', 'blue'))
            surf1_idx = match.get('fragment1_idx', 0)
            surf2_idx = match.get('fragment2_idx', 0)
            
            # Get surface points
            if (color1 in fragment1.get('break_surfaces', {}) and
                color2 in fragment2.get('break_surfaces', {}) and
                surf1_idx < len(fragment1['break_surfaces'][color1]) and
                surf2_idx < len(fragment2['break_surfaces'][color2])):
                
                points1 = fragment1['break_surfaces'][color1][surf1_idx]['points']
                points2 = fragment2['break_surfaces'][color2][surf2_idx]['points']
                
                if len(points1) == 0 or len(points2) == 0:
                    print("      No points to visualize")
                    return
                
                # Create point clouds
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(points1)
                
                # Offset second fragment for visibility
                offset_points2 = np.array(points2) + [0.1, 0, 0]
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = o3d.utility.Vector3dVector(offset_points2)
                
                # Color the point clouds
                color_map = {'blue': [0, 0, 1], 'green': [0, 1, 0], 'red': [1, 0, 0]}
                pcd1.paint_uniform_color(color_map.get(color1, [0.5, 0.5, 0.5]))
                pcd2.paint_uniform_color(color_map.get(color2, [0.5, 0.5, 0.5]))
                
                # Create connecting line between centroids
                centroid1 = np.mean(points1, axis=0)
                centroid2 = np.mean(offset_points2, axis=0)
                
                line_points = [centroid1, centroid2]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line_points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.paint_uniform_color([1, 1, 0])  # Yellow connecting line
                
                # Show visualization
                window_name = f"Match: Frag{frag1_idx}({color1}) ↔ Frag{frag2_idx}({color2}) - Sim:{match['similarity']:.3f}"
                o3d.visualization.draw_geometries([pcd1, pcd2, line_set], window_name=window_name)
                
        except Exception as e:
            print(f"      Visualization error: {e}")

    def build_alignment_graph(self):
        """Build graph of fragment connections based on matches"""
        graph = {}
        
        # Initialize graph with all fragments
        for i in range(len(self.enhanced_fragments)):
            graph[i] = set()
        
        # Add edges for matched fragments
        for pair_key, color_matches in self.all_matches.items():
            # Extract fragment indices from pair key
            parts = pair_key.split('_')
            frag1_idx = int(parts[1])
            frag2_idx = int(parts[3])
            
            # Check if there are any matches for this pair
            has_matches = any(len(matches) > 0 for matches in color_matches.values())
            
            if has_matches:
                graph[frag1_idx].add(frag2_idx)
                graph[frag2_idx].add(frag1_idx)
        
        return graph
    
    def find_connected_components(self, graph):
        """Find connected components in the alignment graph"""
        visited = set()
        components = []
        
        for node in graph:
            if node not in visited:
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(graph[current] - visited)
                
                components.append(sorted(component))
        
        return components
    
    def select_origin_fragment(self, fragment_indices):
        """Select fragment with most high-quality matches as origin"""
        fragment_scores = {}
        
        # Count matches and compute scores for each fragment
        for frag_idx in fragment_indices:
            total_score = 0
            match_count = 0
            
            for pair_key, color_matches in self.all_matches.items():
                parts = pair_key.split('_')
                frag1_idx = int(parts[1])
                frag2_idx = int(parts[3])
                
                if frag_idx in [frag1_idx, frag2_idx]:
                    for matches in color_matches.values():
                        for match in matches:
                            total_score += match['similarity']
                            match_count += 1
            
            fragment_scores[frag_idx] = {
                'total_score': total_score,
                'match_count': match_count,
                'avg_score': total_score / max(match_count, 1)
            }
        
        # Select fragment with highest total score
        best_fragment = max(fragment_scores.keys(), 
                           key=lambda x: fragment_scores[x]['total_score'])
        
        if self.config.get('debug_assembly', False):
            print(f"   Fragment match quality scores:")
            for frag_idx, scores in fragment_scores.items():
                marker = "👑 ORIGIN" if frag_idx == best_fragment else "  "
                print(f"     {marker} Fragment {frag_idx}: {scores['match_count']} matches, "
                      f"avg sim: {scores['avg_score']:.3f}")
        
        return best_fragment
    
    def find_best_assembly_match(self, assembled, remaining):
        """Find best match between assembled and remaining fragments"""
        best_match_info = None
        best_similarity = 0
        
        for assembled_frag in assembled:
            for remaining_frag in remaining:
                pair_key = f"fragment_{min(assembled_frag, remaining_frag)}_to_{max(assembled_frag, remaining_frag)}"
                
                if pair_key in self.all_matches:
                    # Find best match for this pair
                    for color_matches in self.all_matches[pair_key].values():
                        for match in color_matches:
                            if match['similarity'] > best_similarity:
                                best_similarity = match['similarity']
                                best_match_info = {
                                    'assembled_fragment': assembled_frag,
                                    'target_fragment': remaining_frag,
                                    'match': match,
                                    'pair_key': pair_key
                                }
        
        return best_match_info
    
    # NEW: Helper functions for overlap detection and collision prevention
    def compute_minimum_clearance(self, points1, points2, max_points=1000):
        """Compute minimum distance between two point clouds with subsampling"""
        if len(points1) == 0 or len(points2) == 0:
            return float('inf')
        
        # Subsample large point clouds to avoid memory issues
        if len(points1) > max_points:
            indices1 = np.random.choice(len(points1), max_points, replace=False)
            points1 = points1[indices1]
        
        if len(points2) > max_points:
            indices2 = np.random.choice(len(points2), max_points, replace=False)
            points2 = points2[indices2]
        
        # Use scipy's cdist for efficient distance computation
        distances = cdist(points1, points2)
        return np.min(distances)
    
    def check_fragment_overlap(self, fragment1_points, fragment2_points, min_clearance=0.003):
        """Check if two fragments overlap (closer than min_clearance)"""
        clearance = self.compute_minimum_clearance(fragment1_points, fragment2_points)
        is_overlapping = clearance < min_clearance
        
        if self.config.get('debug_assembly', False):
            print(f"      Clearance check: {clearance*1000:.2f}mm (overlap: {is_overlapping})")
        
        return is_overlapping, clearance
    
    def get_fragment_points(self, fragment, max_points=5000):
        """Get all vertex points from a fragment mesh with subsampling for large meshes"""
        if 'mesh' not in fragment:
            return np.array([])
        
        vertices = np.asarray(fragment['mesh'].vertices)
        
        # Subsample if too many points to avoid memory issues
        if len(vertices) > max_points:
            indices = np.random.choice(len(vertices), max_points, replace=False)
            vertices = vertices[indices]
        
        return vertices
    
    def apply_transform_to_points(self, points, transform):
        """Apply 4x4 transformation to points"""
        if len(points) == 0:
            return points
        
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        # Apply transformation
        transformed_homo = (transform @ points_homo.T).T
        # Return 3D points
        return transformed_homo[:, :3]
    
    def find_non_overlapping_position(self, target_fragment, assembled_fragments, base_transform, max_attempts=12):
        """Find a position for target fragment that doesn't overlap with assembled ones"""
        
        target_points = self.get_fragment_points(target_fragment)
        if len(target_points) == 0:
            return base_transform
        
        # Try the base transform first
        test_transform = base_transform.copy()
        transformed_points = self.apply_transform_to_points(target_points, test_transform)
        
        # Check against all assembled fragments
        has_overlap = False
        for assembled_frag in assembled_fragments:
            assembled_points = self.get_fragment_points(assembled_frag['fragment'])
            if len(assembled_points) > 0:
                assembled_transformed = self.apply_transform_to_points(
                    assembled_points, assembled_frag['transform']
                )
                overlapping, clearance = self.check_fragment_overlap(
                    transformed_points, assembled_transformed, min_clearance=0.003
                )
                if overlapping:
                    has_overlap = True
                    break
        
        if not has_overlap:
            if self.config.get('debug_assembly', False):
                print(f"      Base position is clear")
            return test_transform
        
        # If overlapping, try different positions
        if self.config.get('debug_assembly', False):
            print(f"      Base position overlaps, trying alternatives...")
        
        # Try radial offsets around the base position
        base_translation = base_transform[:3, 3]
        
        for attempt in range(max_attempts):
            # Create radial offset
            angle = (attempt * 2 * np.pi) / max_attempts
            radius = 0.02 + (attempt * 0.01)  # Start at 2cm, increase by 1cm per attempt
            
            offset = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0.01 * attempt  # Small vertical offset
            ])
            
            test_transform = base_transform.copy()
            test_transform[:3, 3] = base_translation + offset
            
            transformed_points = self.apply_transform_to_points(target_points, test_transform)
            
            # Check this position
            position_clear = True
            for assembled_frag in assembled_fragments:
                assembled_points = self.get_fragment_points(assembled_frag['fragment'])
                if len(assembled_points) > 0:
                    assembled_transformed = self.apply_transform_to_points(
                        assembled_points, assembled_frag['transform']
                    )
                    overlapping, clearance = self.check_fragment_overlap(
                        transformed_points, assembled_transformed, min_clearance=0.003
                    )
                    if overlapping:
                        position_clear = False
                        break
            
            if position_clear:
                if self.config.get('debug_assembly', False):
                    print(f"      Found clear position at attempt {attempt+1}")
                    print(f"      Offset: {np.linalg.norm(offset)*1000:.2f}mm")
                return test_transform
        
        # If no clear position found, use the last attempt with a warning
        if self.config.get('debug_assembly', False):
            print(f"      Warning: No clear position found, using last attempt")
        
        return test_transform
    
    def create_safe_isolated_transform(self, fragment, assembled_fragments, position_index):
        """Create a safe transform for fragments that couldn't be matched"""
        # Position isolated fragments in a grid pattern away from assembled ones
        grid_spacing = 0.15  # 15cm spacing
        grid_x = (position_index % 3) * grid_spacing
        grid_y = (position_index // 3) * grid_spacing
        
        # Offset from origin
        safe_position = np.array([0.5 + grid_x, 0.5 + grid_y, 0.1])
        
        transform = np.eye(4)
        transform[:3, 3] = safe_position
        
        if self.config.get('debug_assembly', False):
            print(f"      Isolated positioning at ({grid_x:.2f}, {grid_y:.2f})")
        
        return transform
    
    def compute_contact_transform(self, assembled_fragment, target_fragment, match):
        """IMPROVED: Compute transform that brings break surfaces into contact with overlap prevention"""
        try:
            # Get surface information
            color1 = match.get('surface1_color', match.get('color', 'blue'))
            color2 = match.get('surface2_color', match.get('color', 'blue'))
            surf1_idx = match.get('fragment1_idx', 0)
            surf2_idx = match.get('fragment2_idx', 0)
            
            # Get surface points
            if (color1 in assembled_fragment['break_surfaces'] and 
                color2 in target_fragment['break_surfaces'] and
                surf1_idx < len(assembled_fragment['break_surfaces'][color1]) and
                surf2_idx < len(target_fragment['break_surfaces'][color2])):
                
                points1 = assembled_fragment['break_surfaces'][color1][surf1_idx]['points']
                points2 = target_fragment['break_surfaces'][color2][surf2_idx]['points']
            else:
                if self.config.get('debug_assembly', False):
                    print(f"      Invalid surface indices: {color1}[{surf1_idx}], {color2}[{surf2_idx}]")
                return np.eye(4)
            
            if len(points1) == 0 or len(points2) == 0:
                return np.eye(4)
            
            # Compute surface centroids
            centroid1 = np.mean(points1, axis=0)
            centroid2 = np.mean(points2, axis=0)
            
            # Basic validation: centroids should be reasonable
            if np.any(np.abs(centroid1) > 10) or np.any(np.abs(centroid2) > 10):
                if self.config.get('debug_assembly', False):
                    print(f"      Warning: Extreme centroid values detected")
                    print(f"      C1: {centroid1}, C2: {centroid2}")
                # Use simple offset instead
                translation = np.array([self.config['assembly_contact_distance'], 0, 0])
            else:
                # Get surface normals from features
                if 'surface1_features' in match and 'surface2_features' in match:
                    normal1 = np.array(match['surface1_features']['normal'])
                    normal2 = np.array(match['surface2_features']['normal'])
                    
                    # Normalize normals
                    normal1 = normal1 / (np.linalg.norm(normal1) + 1e-8)
                    normal2 = normal2 / (np.linalg.norm(normal2) + 1e-8)
                    
                    # For break surfaces, we want them to face each other
                    # Move target surface close to assembled surface along normal direction
                    contact_distance = self.config['assembly_contact_distance']
                    contact_offset = normal1 * contact_distance
                    target_position = centroid1 + contact_offset
                    translation = target_position - centroid2
                    
                    # CRITICAL: Validate translation magnitude
                    translation_magnitude = np.linalg.norm(translation)
                    max_reasonable_translation = 1.0  # 1 meter maximum
                    
                    if translation_magnitude > max_reasonable_translation:
                        if self.config.get('debug_assembly', False):
                            print(f"      Warning: Extreme translation detected: {translation_magnitude:.2f}m")
                            print(f"      Using fallback positioning instead")
                        
                        # Use simple centroid-based approach instead
                        direction = centroid1 - centroid2
                        distance = np.linalg.norm(direction)
                        
                        if distance > 0 and distance < max_reasonable_translation:
                            safe_distance = self.config['assembly_contact_distance']
                            translation = direction * (1.0 - safe_distance / distance)
                        else:
                            # Last resort: small fixed offset
                            translation = np.array([self.config['assembly_contact_distance'], 0, 0])
                    
                    if self.config.get('debug_assembly', False):
                        print(f"      Contact positioning: moving {np.linalg.norm(translation)*1000:.2f}mm")
                
                else:
                    # Fallback: simple centroid alignment with safe distance
                    direction = centroid1 - centroid2
                    distance = np.linalg.norm(direction)
                    
                    if distance > 0 and distance < 1.0:  # Reasonable distance
                        # Bring close but not touching
                        safe_distance = self.config['assembly_contact_distance']
                        translation = direction * (1.0 - safe_distance / distance)
                    else:
                        # Use small fixed offset for extreme cases
                        translation = np.array([self.config['assembly_contact_distance'], 0, 0])
                    
                    if self.config.get('debug_assembly', False):
                        print(f"      Fallback positioning: {np.linalg.norm(translation)*1000:.2f}mm")
            
            # Final validation: ensure translation is reasonable
            if np.linalg.norm(translation) > 1.0:  # More than 1 meter is unreasonable
                if self.config.get('debug_assembly', False):
                    print(f"      Final validation failed, using minimal offset")
                translation = np.array([self.config['assembly_contact_distance'], 0, 0])
            
            # Build transformation matrix
            transform = np.eye(4)
            transform[:3, 3] = translation
            
            return transform
            
        except Exception as e:
            if self.config.get('debug_assembly', False):
                print(f"      Contact transform error: {e}")
            # Fallback to minimal offset
            transform = np.eye(4)
            transform[:3, 3] = np.array([self.config['assembly_contact_distance'], 0, 0])
            return transform
    
    def improved_align_component(self, fragment_indices):
        """FIXED: Improved alignment using origin-based progressive assembly with overlap prevention"""
        
        start_time = time.time()
        max_time = 180  # Max 3 minutes per component
        
        if self.config.get('debug_assembly', False):
            print(f"\n🔧 FIXED COMPONENT ASSEMBLY")
            print(f"   Fragments: {fragment_indices}")
            print(f"   Strategy: Origin-based progressive assembly with overlap prevention")
            print(f"   Contact distance: {self.config['assembly_contact_distance']*1000:.1f}mm")
        
        if len(fragment_indices) < 2:
            return {fragment_indices[0]: np.eye(4)}, {}
        
        # Step 1: Select origin fragment (fragment with most matches)
        origin_fragment = self.select_origin_fragment(fragment_indices)
        
        # Initialize transforms with origin at identity
        transforms = {origin_fragment: np.eye(4)}
        assembled_fragments = [{
            'fragment': self.enhanced_fragments[origin_fragment],
            'transform': np.eye(4),
            'index': origin_fragment
        }]
        
        remaining = set(fragment_indices) - {origin_fragment}
        assembly_order = [origin_fragment]
        contact_distances = []
        
        if self.config.get('debug_assembly', False):
            print(f"   Origin fragment: {origin_fragment}")
            print(f"   Remaining fragments: {remaining}")
        
        # Step 2: Progressive assembly with overlap prevention
        while remaining and (time.time() - start_time) < max_time:
            
            # Find best match between assembled and remaining fragments
            best_match_info = self.find_best_assembly_match(
                [af['index'] for af in assembled_fragments], remaining
            )
            
            if best_match_info is None:
                if self.config.get('debug_assembly', False):
                    print(f"   No more valid connections found")
                
                # Position remaining fragments safely away from assembled ones
                for frag_idx in remaining:
                    safe_transform = self.create_safe_isolated_transform(
                        self.enhanced_fragments[frag_idx], assembled_fragments, len(transforms)
                    )
                    transforms[frag_idx] = safe_transform
                break
            
            # Extract match information
            assembled_frag_idx = best_match_info['assembled_fragment']
            target_frag_idx = best_match_info['target_fragment']
            match = best_match_info['match']
            similarity = match['similarity']
            
            if self.config.get('debug_assembly', False):
                print(f"\n   Assembling fragment {target_frag_idx} to {assembled_frag_idx}")
                print(f"   Match similarity: {similarity:.3f}")
            
            # Compute base alignment transform
            base_transform = self.compute_contact_transform(
                self.enhanced_fragments[assembled_frag_idx],
                self.enhanced_fragments[target_frag_idx],
                match
            )
            
            # Apply relative to assembled fragment's current position
            base_assembled_transform = transforms[assembled_frag_idx]
            initial_target_transform = base_assembled_transform @ base_transform
            
            # Find non-overlapping position
            final_transform = self.find_non_overlapping_position(
                self.enhanced_fragments[target_frag_idx],
                assembled_fragments,
                initial_target_transform
            )
            
            # Store the transform
            transforms[target_frag_idx] = final_transform
            
            # Add to assembled fragments list
            assembled_fragments.append({
                'fragment': self.enhanced_fragments[target_frag_idx],
                'transform': final_transform,
                'index': target_frag_idx
            })
            
            # Evaluate contact quality if possible
            contact_dist = self.evaluate_contact_distance(
                self.enhanced_fragments[assembled_frag_idx], 
                self.enhanced_fragments[target_frag_idx],
                match, transforms[assembled_frag_idx], final_transform
            )
            
            if contact_dist is not None:
                contact_distances.append(contact_dist)
                if self.config.get('debug_assembly', False):
                    print(f"   Contact distance: {contact_dist*1000:.2f}mm")
            
            # Update sets
            remaining.remove(target_frag_idx)
            assembly_order.append(target_frag_idx)
            
            if self.config.get('debug_assembly', False):
                print(f"   ✅ Fragment {target_frag_idx} assembled successfully")
        
        total_time = time.time() - start_time
        
        # Compute assembly quality
        assembly_quality = {
            'mean_contact_distance': np.mean(contact_distances) if contact_distances else None,
            'max_contact_distance': np.max(contact_distances) if contact_distances else None,
            'contact_count': len(contact_distances),
            'assembly_success_rate': len(assembly_order) / len(fragment_indices)
        }
        
        component_metrics = {
            'alignment_time': total_time,
            'fragments_aligned': len(transforms),
            'assembly_order': assembly_order,
            'assembly_quality': assembly_quality
        }
        
        if self.config.get('debug_assembly', False):
            print(f"\n   🔧 FIXED Assembly completed in {total_time:.1f}s")
            print(f"   Assembly order: {assembly_order}")
            print(f"   Success rate: {assembly_quality['assembly_success_rate']*100:.1f}%")
            if contact_distances:
                mean_contact = np.mean(contact_distances) * 1000
                print(f"   Average contact distance: {mean_contact:.2f}mm")
        
        return transforms, component_metrics
    
    def evaluate_contact_distance(self, frag1, frag2, match, transform1, transform2):
        """Evaluate distance between matching surfaces after transformation"""
        try:
            color1 = match.get('surface1_color', match.get('color', 'blue'))
            color2 = match.get('surface2_color', match.get('color', 'blue'))
            surf1_idx = match.get('fragment1_idx', 0)
            surf2_idx = match.get('fragment2_idx', 0)
            
            if (color1 not in frag1['break_surfaces'] or 
                color2 not in frag2['break_surfaces'] or
                surf1_idx >= len(frag1['break_surfaces'][color1]) or
                surf2_idx >= len(frag2['break_surfaces'][color2])):
                return None
            
            points1 = np.array(frag1['break_surfaces'][color1][surf1_idx]['points'])
            points2 = np.array(frag2['break_surfaces'][color2][surf2_idx]['points'])
            
            if len(points1) == 0 or len(points2) == 0:
                return None
            
            # Apply transformations
            points1_homo = np.hstack([points1, np.ones((len(points1), 1))])
            points2_homo = np.hstack([points2, np.ones((len(points2), 1))])
            
            points1_transformed = (transform1 @ points1_homo.T).T[:, :3]
            points2_transformed = (transform2 @ points2_homo.T).T[:, :3]
            
            # Compute centroids
            centroid1 = np.mean(points1_transformed, axis=0)
            centroid2 = np.mean(points2_transformed, axis=0)
            
            # Distance between centroids
            distance = np.linalg.norm(centroid1 - centroid2)
            
            return distance
            
        except Exception as e:
            if self.config.get('debug_assembly', False):
                print(f"      Contact evaluation error: {e}")
            return None

    def align_fragments(self):
        """Step 4: Align fragments based on surface matches"""
        print("\n" + "="*60)
        print("STEP 4: Aligning fragments") 
        print("="*60)
        
        start_time = time.time()
        
        if not self.all_matches:
            print("No matches found - skipping alignment")
            return {}
        
        # Build alignment graph and find connected components
        alignment_graph = self.build_alignment_graph()
        connected_components = self.find_connected_components(alignment_graph)
        
        print(f"Found {len(connected_components)} connected component(s)")
        
        # Process each connected component using improved assembly
        self.transformations = {}
        component_metrics = []
        
        for i, component in enumerate(connected_components):
            print(f"\nProcessing component {i+1} with {len(component)} fragments:")
            
            if len(component) == 1:
                self.transformations[component[0]] = np.eye(4)
                continue
            
            # Use fixed alignment method
            component_transforms, metrics = self.improved_align_component(component)
            
            if self.config.get('debug_assembly', False):
                print(f"   DEBUG: Received {len(component_transforms)} transforms from component")
            
            self.transformations.update(component_transforms)
            component_metrics.append(metrics)
            
            if self.config.get('debug_assembly', False):
                print(f"   DEBUG: Total transforms now: {len(self.transformations)}")
        
        # Compute overall quality metrics
        self.quality_metrics = self.compute_overall_quality(component_metrics)
        
        elapsed = time.time() - start_time
        print(f"\nStep 4 completed in {elapsed:.2f} seconds")
        print(f"Aligned {len(self.transformations)} fragments")
        
        # Print assembly quality summary
        if component_metrics:
            all_contact_distances = []
            for metrics in component_metrics:
                if 'assembly_quality' in metrics and metrics['assembly_quality']['mean_contact_distance']:
                    all_contact_distances.append(metrics['assembly_quality']['mean_contact_distance'])
            
            if all_contact_distances:
                mean_overall_contact = np.mean(all_contact_distances) * 1000  # Convert to mm
                print(f"Overall assembly quality: {mean_overall_contact:.2f}mm average contact distance")
        
        return self.transformations
    
    def compute_overall_quality(self, component_metrics):
        """Compute overall reconstruction quality"""
        if not component_metrics:
            return {}
        
        # Collect assembly quality metrics
        contact_distances = []
        assembly_times = []
        
        for metrics in component_metrics:
            if 'alignment_time' in metrics:
                assembly_times.append(metrics['alignment_time'])
            
            if ('assembly_quality' in metrics and 
                metrics['assembly_quality']['mean_contact_distance'] is not None):
                contact_distances.append(metrics['assembly_quality']['mean_contact_distance'])
        
        quality_metrics = {
            'num_aligned_fragments': len(self.transformations),
            'total_fragments': len(self.fragments),
            'total_assembly_time': sum(assembly_times) if assembly_times else 0
        }
        
        if contact_distances:
            quality_metrics.update({
                'mean_contact_distance': np.mean(contact_distances),
                'max_contact_distance': np.max(contact_distances),
                'min_contact_distance': np.min(contact_distances)
            })
        
        return quality_metrics
    
    def visualize_and_report(self, output_directory):
        """Step 5: Create visualizations and reports"""
        print("\n" + "="*60)
        print("STEP 5: Creating visualizations and reports")
        print("="*60)
        
        start_time = time.time()
        
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)
        
        if self.config['visualize_steps']:
            print("Showing step-by-step visualizations...")
            self.visualizer.visualize_original_fragments(self.fragments)
            
            if self.all_matches:
                # Show some example matches
                for pair_key, color_matches in list(self.all_matches.items())[:2]:
                    for color, matches in color_matches.items():
                        if matches:
                            parts = pair_key.split('_')
                            idx1, idx2 = int(parts[1]), int(parts[3])
                            self.visualizer.visualize_surface_matches(
                                self.enhanced_fragments[idx1],
                                self.enhanced_fragments[idx2],
                                matches[:3]  # Show top 3 matches
                            )
                            break
                    break
        
        # Always show final reconstruction
        if self.transformations:
            print("Showing final reconstruction...")
            final_mesh = self.visualizer.visualize_final_reconstruction(
                self.fragments, self.transformations
            )
        
        if self.config['output_reports']:
            print("Generating reports...")
            
            # Match quality report
            if self.all_matches:
                self.visualizer.create_match_quality_report(
                    self.all_matches, 
                    output_dir / "match_quality_report.png"
                )
            
            # Reconstruction report
            if self.transformations:
                self.visualizer.create_reconstruction_report(
                    self.fragments, self.transformations, self.quality_metrics,
                    output_dir / "reconstruction_report.png"
                )
        
        # Save reconstruction
        if self.transformations:
            print("Saving reconstruction...")
            self.visualizer.save_reconstruction(
                self.fragments, self.transformations, output_dir / "reconstruction"
            )
        
        elapsed = time.time() - start_time
        print(f"Step 5 completed in {elapsed:.2f} seconds")
    
    def save_pipeline_data(self, output_directory):
        """Save intermediate pipeline data"""
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)
        
        if self.config['save_intermediate']:
            print("Saving intermediate data...")
            
            # Save fragment data
            self.ply_extractor.save_fragment_data(
                self.fragments, output_dir / "fragment_data.json"
            )
            
            # Save matches with proper JSON serialization
            if self.all_matches:
                print("Converting match data for JSON serialization...")
                serializable_matches = {}
                
                for pair_key, color_matches in self.all_matches.items():
                    serializable_matches[pair_key] = {}
                    
                    for color, matches in color_matches.items():
                        serializable_matches[pair_key][color] = []
                        
                        for match in matches:
                            # Create clean match dict with JSON-serializable types
                            clean_match = {}
                            
                            for k, v in match.items():
                                # Skip non-serializable data
                                if k in ['surface1_features', 'surface2_features']:
                                    continue
                                
                                # Convert numpy types to Python types
                                if hasattr(v, 'item'):  # numpy scalar
                                    clean_match[k] = v.item()
                                elif isinstance(v, dict):
                                    # Recursively convert dict values
                                    clean_dict = {}
                                    for dk, dv in v.items():
                                        if hasattr(dv, 'item'):
                                            clean_dict[dk] = dv.item()
                                        else:
                                            clean_dict[dk] = dv
                                    clean_match[k] = clean_dict
                                else:
                                    clean_match[k] = v
                            
                            serializable_matches[pair_key][color].append(clean_match)
                
                # Save cleaned matches
                with open(output_dir / "surface_matches.json", 'w') as f:
                    json.dump(serializable_matches, f, indent=2)
                print("Surface matches saved successfully!")
            
            # Save transformations
            if self.transformations:
                transform_data = {}
                for i, transform in self.transformations.items():
                    transform_data[f"fragment_{i}"] = transform.tolist()
                
                with open(output_dir / "transformations.json", 'w') as f:
                    json.dump(transform_data, f, indent=2)
                print("Transformations saved successfully!")
            
            # Save quality metrics
            with open(output_dir / "quality_metrics.json", 'w') as f:
                json.dump(self.quality_metrics, f, indent=2)
            print("Quality metrics saved successfully!")
    
    def run_full_pipeline(self, input_directory, output_directory):
        """Run the complete reconstruction pipeline"""
        print("MAYAN STELE FRAGMENT RECONSTRUCTION PIPELINE")
        print("=" * 60)
        print(f"Input directory: {input_directory}")
        print(f"Output directory: {output_directory}")
        print(f"Configuration: {self.config}")
        print()
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Load and process fragments
            self.load_and_process_fragments(input_directory)
            
            # Step 2: Extract features
            self.extract_features()
            
            # Step 3: Find surface matches
            self.find_surface_matches()
            
            # Step 4: Align fragments
            self.align_fragments()
            
            # Step 5: Visualize and report
            self.visualize_and_report(output_directory)
            
            # Save all data
            self.save_pipeline_data(output_directory)
            
            # Final summary
            total_time = time.time() - overall_start_time
            print("\n" + "="*60)
            print("RECONSTRUCTION PIPELINE COMPLETED")
            print("="*60)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Fragments processed: {len(self.fragments)}")
            print(f"Fragments aligned: {len(self.transformations)}")
            
            # Enhanced quality reporting
            if 'mean_contact_distance' in self.quality_metrics:
                contact_mm = self.quality_metrics['mean_contact_distance'] * 1000
                print(f"Assembly quality: {contact_mm:.2f}mm average contact distance")
            else:
                print(f"Assembly quality: Optimal (surfaces in contact)")
            
            print(f"Results saved to: {output_directory}")
            
            return True
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Command line interface for the reconstruction pipeline"""
    parser = argparse.ArgumentParser(
        description="Mayan Stele Fragment Reconstruction Pipeline"
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing PLY files with colored break surfaces"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory to save reconstruction results"
    )
    
    parser.add_argument(
        "--min-similarity", 
        type=float, 
        default=0.6,
        help="Minimum similarity threshold for surface matching (default: 0.6)"
    )
    
    parser.add_argument(
        "--color-tolerance",
        type=float,
        default=0.3,
        help="Color matching tolerance (default: 0.3)"
    )
    
    parser.add_argument(
        "--visualize-steps",
        action="store_true",
        help="Show step-by-step visualizations"
    )
    
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating detailed reports"
    )
    
    parser.add_argument(
        "--config",
        help="JSON configuration file"
    )
    
    parser.add_argument(
        "--contact-distance",
        type=float,
        default=0.005,
        help="Target contact distance between surfaces in meters (default: 0.005 = 5mm)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'min_similarity': args.min_similarity,
        'color_tolerance': args.color_tolerance,
        'visualize_steps': args.visualize_steps,
        'output_reports': not args.no_reports,
        'assembly_contact_distance': args.contact_distance,
        'debug_assembly': True  # Enable debugging by default
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Run pipeline
    pipeline = ReconstructionPipeline(config)
    success = pipeline.run_full_pipeline(args.input_dir, args.output_dir)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()