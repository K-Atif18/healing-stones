#!/usr/bin/env python3
"""
Direct Mesh Color Detector
Works directly with mesh vertices - no sampling, no clustering
Simply finds colored vertices and uses them as break surfaces
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import json

class DirectMeshColorDetector:
    def __init__(self):
        """Initialize direct mesh color detector"""
        pass
        
    def load_mesh(self, filepath):
        """Load PLY file as mesh"""
        try:
            mesh = o3d.io.read_triangle_mesh(str(filepath))
            if len(mesh.vertices) > 0 and len(mesh.vertex_colors) > 0:
                print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
                print(f"Has vertex colors: {len(mesh.vertex_colors)} colors")
                return mesh
            else:
                print(f"Warning: {filepath} has no vertices or colors")
                return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_colored_vertices(self, mesh, color_name):
        """
        Get vertices of a specific color directly from mesh
        
        Args:
            mesh: Open3D triangle mesh
            color_name: 'red', 'green', or 'blue'
        
        Returns:
            numpy array of colored vertex indices
        """
        if not mesh.has_vertex_colors():
            return np.array([])
        
        colors = np.asarray(mesh.vertex_colors)  # Already in 0-1 range
        
        # Apply color thresholds (same as your suggestion)
        if color_name == 'green':
            mask = (
                (colors[:, 1] > 0.6) &  # Green channel is high  
                (colors[:, 0] < 0.4) &  # Red channel is low
                (colors[:, 2] < 0.4)    # Blue channel is low
            )
        elif color_name == 'blue':
            mask = (
                (colors[:, 2] > 0.6) &  # Blue channel is high
                (colors[:, 0] < 0.4) &  # Red channel is low
                (colors[:, 1] < 0.4)    # Green channel is low
            )
        elif color_name == 'red':
            mask = (
                (colors[:, 0] > 0.6) &  # Red channel is high
                (colors[:, 1] < 0.4) &  # Green channel is low
                (colors[:, 2] < 0.4)    # Blue channel is low
            )
        else:
            return np.array([])
        
        colored_indices = np.where(mask)[0]
        return colored_indices
    
    def create_break_surface(self, mesh, color_name, colored_indices):
        """
        Create break surface data structure from colored vertices
        
        Args:
            mesh: Open3D triangle mesh
            color_name: 'red', 'green', or 'blue'  
            colored_indices: Indices of colored vertices
        
        Returns:
            Break surface dictionary compatible with reconstruction pipeline
        """
        if len(colored_indices) == 0:
            return None
        
        vertices = np.asarray(mesh.vertices)
        colored_points = vertices[colored_indices]
        
        # Create point cloud for this surface
        surface_pcd = o3d.geometry.PointCloud()
        surface_pcd.points = o3d.utility.Vector3dVector(colored_points)
        
        # Estimate normals
        surface_pcd.estimate_normals()
        
        # Create surface data structure compatible with reconstruction pipeline
        surface_data = {
            'points': colored_points,
            'point_cloud': surface_pcd,
            'color': color_name,
            'size': len(colored_points)
        }
        
        return surface_data
    
    def process_fragment(self, filepath):
        """
        Process a single fragment file and extract colored break surfaces directly
        
        Returns:
            Dictionary containing fragment data and break surfaces
        """
        print(f"Processing {Path(filepath).name}...")
        
        # Load mesh directly
        mesh = self.load_mesh(str(filepath))
        if mesh is None:
            return None
        
        fragment_data = {
            'filepath': str(filepath),
            'mesh': mesh,
            'break_surfaces': {}
        }
        
        # Extract break surfaces for each color
        for color in ['blue', 'green', 'red']:
            print(f"  Extracting {color} break surface...")
            
            # Get colored vertices directly
            colored_indices = self.get_colored_vertices(mesh, color)
            
            if len(colored_indices) > 0:
                # Create break surface directly from these vertices
                surface_data = self.create_break_surface(mesh, color, colored_indices)
                
                if surface_data:
                    fragment_data['break_surfaces'][color] = [surface_data]  # List with single surface
                    print(f"    Found {color} surface: {len(colored_indices)} vertices")
                else:
                    fragment_data['break_surfaces'][color] = []
                    print(f"    Failed to create {color} surface")
            else:
                fragment_data['break_surfaces'][color] = []
                print(f"    No {color} vertices found")
        
        return fragment_data
    
    def test_color_detection(self, filepath):
        """Test color detection and show detailed results"""
        print(f"Testing color detection on {Path(filepath).name}...")
        print("=" * 50)
        
        mesh = self.load_mesh(str(filepath))
        if mesh is None:
            return
        
        colors = np.asarray(mesh.vertex_colors)
        total_vertices = len(colors)
        
        print(f"Total vertices: {total_vertices}")
        print(f"Color range: {colors.min():.3f} to {colors.max():.3f}")
        print()
        
        # Test each color
        for color_name in ['blue', 'green', 'red']:
            colored_indices = self.get_colored_vertices(mesh, color_name)
            count = len(colored_indices)
            percentage = (count / total_vertices) * 100
            
            print(f"{color_name.upper()} DETECTION:")
            print(f"  Vertices found: {count:,} ({percentage:.2f}%)")
            
            if count > 0:
                # Show some sample colors
                sample_colors = colors[colored_indices[:5]]
                print(f"  Sample colors:")
                for i, (r, g, b) in enumerate(sample_colors):
                    print(f"    {i+1}: R={r:.3f}, G={g:.3f}, B={b:.3f}")
                
                # Show color statistics
                color_stats = colors[colored_indices]
                print(f"  Color stats:")
                print(f"    Red   - Min: {color_stats[:, 0].min():.3f}, Max: {color_stats[:, 0].max():.3f}, Mean: {color_stats[:, 0].mean():.3f}")
                print(f"    Green - Min: {color_stats[:, 1].min():.3f}, Max: {color_stats[:, 1].max():.3f}, Mean: {color_stats[:, 1].mean():.3f}")
                print(f"    Blue  - Min: {color_stats[:, 2].min():.3f}, Max: {color_stats[:, 2].max():.3f}, Mean: {color_stats[:, 2].mean():.3f}")
            print()
    
    def create_direct_config(self):
        """Create configuration for direct mesh detection"""
        config = {
            "color_extraction": {
                "method": "direct_mesh_thresholds",
                "color_tolerance": 0.0,  # Not used with direct method
                "min_cluster_size": 1,   # Not used with direct method
                "use_direct_detection": True,
                
                # Direct threshold ranges (converted to 0-255 for compatibility)
                "blue_range": {
                    "min": [0, 0, 153],      # 0.6 * 255
                    "max": [102, 102, 255]   # 0.4 * 255 for R,G limits
                },
                "green_range": {
                    "min": [0, 153, 0],      # 0.6 * 255
                    "max": [102, 255, 102]   # 0.4 * 255 for R,B limits
                },
                "red_range": {
                    "min": [153, 0, 0],      # 0.6 * 255
                    "max": [255, 102, 102]   # 0.4 * 255 for G,B limits
                }
            },
            "matching": {
                "min_similarity": 0.9,   # High selectivity - 90%
                "use_optimal_matching": True,
                "similarity_weights": {
                    "normal_similarity": 0.50,  # Emphasize normals heavily
                    "area_similarity": 0.30,    # Size matters a lot
                    "shape_similarity": 0.10,
                    "curvature_similarity": 0.05,
                    "boundary_similarity": 0.03,
                    "size_similarity": 0.02
                }
            },
            "alignment": {
                "icp_max_iterations": 50,   # Fewer iterations for speed
                "icp_threshold": 0.03,
                "optimization_method": "BFGS"
            },
            "visualization": {
                "show_step_by_step": True,
                "save_images": True
            }
        }
        
        return config

def main():
    """Test the direct mesh color detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct mesh color detection")
    parser.add_argument('input', help='PLY file or directory to test')
    parser.add_argument('--test-only', action='store_true', help='Only test detection, don\'t process')
    parser.add_argument('--config-output', help='Save configuration to file')
    
    args = parser.parse_args()
    
    detector = DirectMeshColorDetector()
    
    input_path = Path(args.input)
    
    if input_path.is_file() and input_path.suffix.lower() == '.ply':
        # Test single file
        if args.test_only:
            detector.test_color_detection(str(input_path))
        else:
            fragment_data = detector.process_fragment(str(input_path))
            if fragment_data:
                total_surfaces = sum(len(surfaces) for surfaces in fragment_data['break_surfaces'].values())
                print(f"\nResult: Found {total_surfaces}/3 expected break surfaces")
                for color, surfaces in fragment_data['break_surfaces'].items():
                    if surfaces:
                        print(f"  {color}: {surfaces[0]['size']} vertices")
                    else:
                        print(f"  {color}: not found")
    
    elif input_path.is_dir():
        # Test directory
        ply_files = list(input_path.glob("*.ply"))
        if not ply_files:
            print(f"No PLY files found in {input_path}")
            return
        
        print(f"Testing direct mesh detection on {len(ply_files)} files...")
        print("=" * 60)
        
        total_surfaces = {'blue': 0, 'green': 0, 'red': 0}
        perfect_fragments = 0
        
        for ply_file in ply_files:
            if args.test_only:
                detector.test_color_detection(str(ply_file))
                print("=" * 50)
            else:
                fragment_data = detector.process_fragment(str(ply_file))
                
                if fragment_data:
                    fragment_surfaces = 0
                    for color, surfaces in fragment_data['break_surfaces'].items():
                        if surfaces:
                            total_surfaces[color] += 1
                            fragment_surfaces += 1
                    
                    if fragment_surfaces == 3:
                        perfect_fragments += 1
        
        if not args.test_only:
            print("\n" + "=" * 60)
            print("DIRECT DETECTION SUMMARY")
            print("=" * 60)
            print(f"Total fragments: {len(ply_files)}")
            print(f"Perfect detection (3/3): {perfect_fragments}")
            print(f"Break surfaces found:")
            for color, count in total_surfaces.items():
                print(f"  {color}: {count}")
    
    else:
        print(f"Invalid input: {args.input}")
        return
    
    # Save configuration if requested
    if args.config_output:
        config = detector.create_direct_config()
        with open(args.config_output, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nDirect detection configuration saved to: {args.config_output}")

if __name__ == "__main__":
    main()