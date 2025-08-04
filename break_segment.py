import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import argparse
import os

class PLYFaceSegmenter:
    def __init__(self, angle_threshold=30.0, min_face_size=50):
        """
        Initialize the face segmenter.
        
        Args:
            angle_threshold: Threshold angle in degrees for detecting sharp edges
            min_face_size: Minimum number of triangles to consider as a face
        """
        self.angle_threshold = np.deg2rad(angle_threshold)
        self.min_face_size = min_face_size
        
    def load_ply_file(self, filepath):
        """Load PLY file and return mesh."""
        try:
            mesh = o3d.io.read_triangle_mesh(filepath)
            if len(mesh.vertices) == 0:
                raise ValueError("Empty mesh loaded")
            return mesh
        except Exception as e:
            print(f"Error loading PLY file: {e}")
            return None
    
    def compute_face_normals(self, mesh):
        """Compute face normals for the mesh."""
        mesh.compute_triangle_normals()
        return np.asarray(mesh.triangle_normals)
    
    def compute_adjacency_matrix(self, mesh):
        """Compute adjacency matrix for triangular faces."""
        triangles = np.asarray(mesh.triangles)
        num_faces = len(triangles)
        adjacency = np.zeros((num_faces, num_faces), dtype=bool)
        
        # Create edge to face mapping
        edge_to_faces = {}
        
        for face_idx, triangle in enumerate(triangles):
            # Get edges of the triangle (sorted for consistency)
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]
            
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Set adjacency for faces sharing edges
        for faces in edge_to_faces.values():
            if len(faces) == 2:  # Edge shared by exactly 2 faces
                face1, face2 = faces
                adjacency[face1, face2] = True
                adjacency[face2, face1] = True
        
        return adjacency
    
    def compute_dihedral_angles(self, mesh, face_normals, adjacency_matrix):
        """Compute dihedral angles between adjacent faces."""
        num_faces = len(face_normals)
        dihedral_angles = np.zeros((num_faces, num_faces))
        
        for i in range(num_faces):
            for j in range(i + 1, num_faces):
                if adjacency_matrix[i, j]:
                    # Compute dihedral angle
                    normal1 = face_normals[i]
                    normal2 = face_normals[j]
                    
                    # Clamp dot product to avoid numerical errors
                    dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                    angle = np.arccos(np.abs(dot_product))
                    
                    # Dihedral angle is π - angle between normals
                    dihedral_angle = np.pi - angle
                    
                    dihedral_angles[i, j] = dihedral_angle
                    dihedral_angles[j, i] = dihedral_angle
        
        return dihedral_angles
    
    def identify_sharp_edges(self, dihedral_angles, adjacency_matrix):
        """Identify sharp edges based on dihedral angle threshold."""
        sharp_edge_matrix = np.zeros_like(adjacency_matrix, dtype=bool)
        
        for i in range(len(dihedral_angles)):
            for j in range(len(dihedral_angles)):
                if adjacency_matrix[i, j] and dihedral_angles[i, j] < self.angle_threshold:
                    sharp_edge_matrix[i, j] = True
        
        return sharp_edge_matrix
    
    def segment_faces_by_connectivity(self, adjacency_matrix, sharp_edge_matrix):
        """Segment faces into regions based on connectivity without sharp edges."""
        num_faces = adjacency_matrix.shape[0]
        
        # Create connectivity matrix (adjacent but not sharp)
        connectivity_matrix = adjacency_matrix & ~sharp_edge_matrix
        
        # Create graph and find connected components
        G = nx.Graph()
        G.add_nodes_from(range(num_faces))
        
        for i in range(num_faces):
            for j in range(i + 1, num_faces):
                if connectivity_matrix[i, j]:
                    G.add_edge(i, j)
        
        # Find connected components (face segments)
        components = list(nx.connected_components(G))
        
        # Filter out small components
        large_components = [comp for comp in components if len(comp) >= self.min_face_size]
        
        # Create face labels
        face_labels = np.full(num_faces, -1, dtype=int)
        for label, component in enumerate(large_components):
            for face_idx in component:
                face_labels[face_idx] = label
        
        return face_labels, large_components
    
    def segment_mesh(self, mesh):
        """Main function to segment mesh faces."""
        print("Computing face normals...")
        face_normals = self.compute_face_normals(mesh)
        
        print("Computing face adjacency...")
        adjacency_matrix = self.compute_adjacency_matrix(mesh)
        
        print("Computing dihedral angles...")
        dihedral_angles = self.compute_dihedral_angles(mesh, face_normals, adjacency_matrix)
        
        print("Identifying sharp edges...")
        sharp_edge_matrix = self.identify_sharp_edges(dihedral_angles, adjacency_matrix)
        
        print("Segmenting faces...")
        face_labels, components = self.segment_faces_by_connectivity(adjacency_matrix, sharp_edge_matrix)
        
        return face_labels, components

class PLYFaceVisualizer:
    def __init__(self):
        # Generate more distinct and vibrant colors for better visualization
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
    
    def visualize_segmented_mesh(self, mesh, face_labels, save_path=None):
        """Visualize mesh with different colors for each face segment using Open3D."""
        
        # Create color array for each triangle
        num_faces = len(face_labels)
        face_colors = np.zeros((num_faces, 3))
        
        unique_labels = np.unique(face_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        print(f"Found {len(valid_labels)} face segments")
        
        # Assign colors to each face segment
        for i, label in enumerate(valid_labels):
            color_idx = i % len(self.colors)
            face_colors[face_labels == label] = self.colors[color_idx]
            print(f"Segment {label}: {np.sum(face_labels == label)} triangles - Color: {self.colors[color_idx]}")
        
        # Gray color for unassigned faces
        unassigned_mask = face_labels == -1
        face_colors[unassigned_mask] = [0.5, 0.5, 0.5]
        if np.sum(unassigned_mask) > 0:
            print(f"Unassigned: {np.sum(unassigned_mask)} triangles - Color: Gray")
        
        # Create visualization mesh
        vis_mesh = o3d.geometry.TriangleMesh()
        vis_mesh.vertices = mesh.vertices
        vis_mesh.triangles = mesh.triangles
        vis_mesh.triangle_colors = o3d.utility.Vector3dVector(face_colors)
        
        # Compute vertex normals for better lighting
        vis_mesh.compute_vertex_normals()
        
        # Create visualizer
        print("\nDisplaying segmented mesh in Open3D...")
        print("Controls:")
        print("- Mouse: Rotate view")
        print("- Mouse wheel: Zoom")
        print("- Ctrl + Mouse: Pan")
        print("- Press 'H' for help")
        print("- Press 'Q' or close window to exit")
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PLY Face Segmentation - Different Colors per Face", 
                         width=1400, height=900)
        vis.add_geometry(vis_mesh)
        
        # Configure render options for better visualization
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.mesh_show_wireframe = False
        render_option.light_on = True
        render_option.point_size = 1.0
        render_option.line_width = 1.0
        
        # Set background color to white for better contrast
        render_option.background_color = np.array([1.0, 1.0, 1.0])
        
        # Run visualization
        vis.run()
        
        # Save screenshot if requested
        if save_path:
            vis.capture_screen_image(save_path)
            print(f"Screenshot saved to {save_path}")
        
        vis.destroy_window()
        
        return vis_mesh
    
    def visualize_with_wireframe(self, mesh, face_labels):
        """Alternative visualization showing wireframe overlay."""
        
        # Create colored mesh
        num_faces = len(face_labels)
        face_colors = np.zeros((num_faces, 3))
        
        unique_labels = np.unique(face_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        for i, label in enumerate(valid_labels):
            color_idx = i % len(self.colors)
            face_colors[face_labels == label] = self.colors[color_idx]
        
        face_colors[face_labels == -1] = [0.5, 0.5, 0.5]
        
        # Create main mesh
        vis_mesh = o3d.geometry.TriangleMesh()
        vis_mesh.vertices = mesh.vertices
        vis_mesh.triangles = mesh.triangles
        vis_mesh.triangle_colors = o3d.utility.Vector3dVector(face_colors)
        vis_mesh.compute_vertex_normals()
        
        # Create wireframe
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(vis_mesh)
        wireframe.paint_uniform_color([0.3, 0.3, 0.3])  # Dark gray wireframe
        
        # Visualize both
        print("\nDisplaying mesh with wireframe overlay...")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="PLY Face Segmentation - With Wireframe", 
                         width=1400, height=900)
        vis.add_geometry(vis_mesh)
        vis.add_geometry(wireframe)
        
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = True
        render_option.background_color = np.array([0.9, 0.9, 0.9])
        
        vis.run()
        vis.destroy_window()
        
        return vis_mesh
    
    def create_face_statistics_plot(self, face_labels, components, save_path=None):
        """Create a plot showing statistics about the segmented faces."""
        unique_labels = np.unique(face_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        # Count faces per segment
        face_counts = []
        for label in valid_labels:
            count = np.sum(face_labels == label)
            face_counts.append(count)
        
        # Create statistics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of face counts per segment
        ax1.bar(range(len(face_counts)), face_counts, 
                color=self.colors[:len(face_counts)][:, :3])
        ax1.set_xlabel('Face Segment')
        ax1.set_ylabel('Number of Triangles')
        ax1.set_title('Triangles per Face Segment')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of face segment sizes
        ax2.hist(face_counts, bins=min(10, len(face_counts)), 
                 color='skyblue', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Number of Triangles')
        ax2.set_ylabel('Number of Segments')
        ax2.set_title('Distribution of Segment Sizes')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics plot saved to {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Segment and visualize 3D PLY fragments by faces')
    parser.add_argument('ply_file', help='Path to PLY file')
    parser.add_argument('--angle_threshold', type=float, default=30.0, 
                       help='Angle threshold in degrees for sharp edge detection (default: 30.0)')
    parser.add_argument('--min_face_size', type=int, default=50,
                       help='Minimum number of triangles per face segment (default: 50)')
    parser.add_argument('--save_screenshot', type=str, default=None,
                       help='Path to save screenshot of visualization')
    parser.add_argument('--save_statistics', type=str, default=None,
                       help='Path to save statistics plot')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.ply_file):
        print(f"Error: File {args.ply_file} not found")
        return
    
    # Initialize segmenter and visualizer
    segmenter = PLYFaceSegmenter(
        angle_threshold=args.angle_threshold,
        min_face_size=args.min_face_size
    )
    visualizer = PLYFaceVisualizer()
    
    # Load mesh
    print(f"Loading PLY file: {args.ply_file}")
    mesh = segmenter.load_ply_file(args.ply_file)
    if mesh is None:
        return
    
    print(f"Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Segment mesh
    face_labels, components = segmenter.segment_mesh(mesh)
    
    # Print results
    num_segments = len([label for label in np.unique(face_labels) if label >= 0])
    unassigned_faces = np.sum(face_labels == -1)
    
    print(f"\nSegmentation Results:")
    print(f"Number of face segments: {num_segments}")
    print(f"Unassigned triangles: {unassigned_faces}")
    print(f"Largest segment: {max([len(comp) for comp in components]) if components else 0} triangles")
    
    # Visualize results
    vis_mesh = visualizer.visualize_segmented_mesh(
        mesh, face_labels, save_path=args.save_screenshot
    )
    
    # Ask user if they want to see wireframe version
    try:
        user_input = input("\nWould you like to see the wireframe overlay version? (y/n): ").lower().strip()
        if user_input == 'y' or user_input == 'yes':
            visualizer.visualize_with_wireframe(mesh, face_labels)
    except KeyboardInterrupt:
        print("\nSkipping wireframe visualization...")
    
    # Create statistics plot
    visualizer.create_face_statistics_plot(
        face_labels, components, save_path=args.save_statistics
    )

if __name__ == "__main__":
    main()

