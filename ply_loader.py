import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pathlib import Path
import json

class PLYColorExtractor:
    """
    Class for loading PLY files and extracting colored break surfaces
    """
    
    def __init__(self):
        self.color_ranges = {
            'blue': {'min': [0, 0, 100], 'max': [100, 100, 255]},
            'green': {'min': [0, 100, 0], 'max': [100, 255, 100]},
            'red': {'min': [100, 0, 0], 'max': [255, 100, 100]}
        }
    
    def load_ply(self, filepath):
        """Load PLY file and return mesh with colors"""
        try:
            mesh = o3d.io.read_triangle_mesh(str(filepath))
            if not mesh.has_vertex_colors():
                print(f"Warning: {filepath} has no vertex colors")
                return None
            return mesh
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_colored_vertices(self, mesh, color_name, tolerance=0.3):
        """
        Extract vertices of a specific color using direct mesh approach
        
        Args:
            mesh: Open3D triangle mesh
            color_name: 'blue', 'green', or 'red'
            tolerance: Color matching tolerance (not used with direct method)
        
        Returns:
            numpy array of colored vertex indices
        """
        if not mesh.has_vertex_colors():
            return np.array([])
        
        colors = np.asarray(mesh.vertex_colors)  # Already in 0-1 range
        
        print(f"    Using direct mesh detection for {color_name}...")
        
        # Use direct color thresholds (same as direct_mesh_detector)
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
        print(f"    Found {len(colored_indices)} {color_name} vertices using direct detection")
        
        return colored_indices
    
    def extract_break_surface_points(self, mesh, color_name, min_cluster_size=50):
        """
        Extract break surface point clouds for a specific color - direct approach
        
        Args:
            mesh: Open3D triangle mesh
            color_name: 'blue', 'green', or 'red'
            min_cluster_size: Minimum points in a cluster (reduced for direct method)
        
        Returns:
            List of point clouds representing break surfaces
        """
        colored_indices = self.extract_colored_vertices(mesh, color_name)
        
        if len(colored_indices) == 0:
            return []
        
        vertices = np.asarray(mesh.vertices)
        colored_points = vertices[colored_indices]
        
        # Since we know each fragment has exactly 1 surface of each color,
        # skip clustering and create a single surface directly
        if len(colored_points) >= min_cluster_size:
            # Create point cloud directly from all colored points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(colored_points)
            
            # Estimate normals
            pcd.estimate_normals()
            
            break_surface = {
                'points': colored_points,
                'point_cloud': pcd,
                'color': color_name,
                'size': len(colored_points)
            }
            
            print(f"    Created {color_name} break surface with {len(colored_points)} points")
            return [break_surface]
        else:
            print(f"    {color_name} surface too small: {len(colored_points)} < {min_cluster_size}")
            return []
    
    def process_fragment(self, filepath):
        """
        Process a single fragment file and extract all break surfaces
        
        Returns:
            Dictionary containing fragment data and break surfaces
        """
        mesh = self.load_ply(filepath)
        if mesh is None:
            return None
        
        fragment_data = {
            'filepath': str(filepath),
            'mesh': mesh,
            'break_surfaces': {}
        }
        
        # Extract break surfaces for each color
        for color in ['blue', 'green', 'red']:
            break_surfaces = self.extract_break_surface_points(mesh, color)
            fragment_data['break_surfaces'][color] = break_surfaces
            print(f"Found {len(break_surfaces)} {color} break surfaces in {filepath.name}")
        
        return fragment_data
    
    def process_all_fragments(self, directory_path):
        """
        Process all PLY files in a directory
        
        Returns:
            List of fragment data dictionaries
        """
        directory = Path(directory_path)
        ply_files = list(directory.glob("*.ply"))
        
        if not ply_files:
            print(f"No PLY files found in {directory_path}")
            return []
        
        fragments = []
        for ply_file in ply_files:
            print(f"Processing {ply_file.name}...")
            fragment_data = self.process_fragment(ply_file)
            if fragment_data:
                fragments.append(fragment_data)
        
        return fragments
    
    def save_fragment_data(self, fragments, output_path):
        """Save fragment data to JSON (excluding mesh objects)"""
        serializable_data = []
        
        for fragment in fragments:
            frag_data = {
                'filepath': fragment['filepath'],
                'break_surfaces': {}
            }
            
            for color, surfaces in fragment['break_surfaces'].items():
                frag_data['break_surfaces'][color] = []
                for surface in surfaces:
                    frag_data['break_surfaces'][color].append({
                        'points': surface['points'].tolist(),
                        'color': surface['color'],
                        'size': surface['size']
                    })
            
            serializable_data.append(frag_data)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Fragment data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    extractor = PLYColorExtractor()
    
    # Process all fragments in a directory
    fragments = extractor.process_all_fragments("path/to/your/ply/files")
    
    # Save data for later use
    extractor.save_fragment_data(fragments, "fragment_data.json")
    
    # Print summary
    for i, fragment in enumerate(fragments):
        print(f"\nFragment {i+1}: {Path(fragment['filepath']).name}")
        for color, surfaces in fragment['break_surfaces'].items():
            if surfaces:
                print(f"  {color}: {len(surfaces)} surfaces")