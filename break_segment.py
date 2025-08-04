import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN
from collections import defaultdict
import argparse
import os
import gc

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
    
    def compute_edge_to_faces_mapping(self, mesh):
        """Create efficient edge to faces mapping using dictionary."""
        triangles = np.asarray(mesh.triangles)
        edge_to_faces = defaultdict(list)
        
        print(f"Processing {len(triangles)} triangles for edge mapping...")
        
        for face_idx, triangle in enumerate(triangles):
            if face_idx % 100000 == 0:
                print(f"Processed {face_idx}/{len(triangles)} triangles")
            
            # Get edges of the triangle (sorted for consistency)
            edges = [
                tuple(sorted([triangle[0], triangle[1]])),
                tuple(sorted([triangle[1], triangle[2]])),
                tuple(sorted([triangle[2], triangle[0]]))
            ]
            
            for edge in edges:
                edge_to_faces[edge].append(face_idx)
        
        return edge_to_faces
    
    def compute_sparse_adjacency_and_angles(self, mesh, face_normals, edge_to_faces):
        """Compute sparse adjacency matrix and dihedral angles efficiently."""
        num_faces = len(face_normals)
        
        # Use lists to build sparse matrix data
        row_indices = []
        col_indices = []
        angle_data = []
        
        print(f"Computing dihedral angles for edges...")
        edge_count = 0
        
        for edge, faces in edge_to_faces.items():
            edge_count += 1
            if edge_count % 50000 == 0:
                print(f"Processed {edge_count}/{len(edge_to_faces)} edges")
            
            if len(faces) == 2:  # Edge shared by exactly 2 faces
                face1, face2 = faces
                
                # Compute dihedral angle
                normal1 = face_normals[face1]
                normal2 = face_normals[face2]
                
                # Clamp dot product to avoid numerical errors
                dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
                angle = np.arccos(np.abs(dot_product))
                
                # Dihedral angle is π - angle between normals
                dihedral_angle = np.pi - angle
                
                # Add to sparse matrix data
                row_indices.extend([face1, face2])
                col_indices.extend([face2, face1])
                angle_data.extend([dihedral_angle, dihedral_angle])
        
        # Create sparse matrices
        adjacency_matrix = csr_matrix((np.ones(len(row_indices)), (row_indices, col_indices)), 
                                    shape=(num_faces, num_faces), dtype=bool)
        
        angle_matrix = csr_matrix((angle_data, (row_indices, col_indices)), 
                                shape=(num_faces, num_faces), dtype=np.float32)
        
        return adjacency_matrix, angle_matrix
    
    def identify_sharp_edges_sparse(self, adjacency_matrix, angle_matrix):
        """Identify sharp edges using sparse matrices."""
        # Create boolean mask for sharp edges
        sharp_mask = angle_matrix.data < self.angle_threshold
        
        # Create new sparse matrix with only non-sharp edges
        smooth_data = adjacency_matrix.data.copy()
        smooth_data[sharp_mask] = False
        
        smooth_adjacency = csr_matrix((smooth_data, adjacency_matrix.indices, adjacency_matrix.indptr),
                                    shape=adjacency_matrix.shape, dtype=bool)
        
        return smooth_adjacency
    
    def segment_faces_sparse(self, smooth_adjacency):
        """Segment faces using sparse connected components."""
        print("Finding connected components...")
        
        # Find connected components in the smooth adjacency graph
        n_components, labels = connected_components(smooth_adjacency, directed=False)
        
        print(f"Found {n_components} initial components")
        
        # Filter components by size
        unique_labels, counts = np.unique(labels, return_counts=True)
        large_components = []
        final_labels = np.full(len(labels), -1, dtype=int)
        
        new_label = 0
        for label, count in zip(unique_labels, counts):
            if count >= self.min_face_size:
                mask = labels == label
                final_labels[mask] = new_label
                large_components.append(np.where(mask)[0])
                new_label += 1
        
        print(f"Kept {len(large_components)} components with >= {self.min_face_size} faces")
        
        return final_labels, large_components
    
    def segment_mesh(self, mesh):
        """Main function to segment mesh faces using memory-efficient approach."""
        print("Computing face normals...")
        face_normals = self.compute_face_normals(mesh)
        
        print("Building edge to faces mapping...")
        edge_to_faces = self.compute_edge_to_faces_mapping(mesh)
        
        print("Computing sparse adjacency and angles...")
        adjacency_matrix, angle_matrix = self.compute_sparse_adjacency_and_angles(
            mesh, face_normals, edge_to_faces)
        
        # Free memory
        del edge_to_faces
        gc.collect()
        
        print("Identifying smooth connectivity...")
        smooth_adjacency = self.identify_sharp_edges_sparse(adjacency_matrix, angle_matrix)
        
        # Free more memory
        del adjacency_matrix, angle_matrix
        gc.collect()
        
        print("Segmenting faces...")
        face_labels, components = self.segment_faces_sparse(smooth_adjacency)
        
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
    
    def downsample_for_visualization(self, mesh, face_labels, max_faces=100000):
        """Downsample mesh if too large for smooth visualization."""
        if len(face_labels) <= max_faces:
            return mesh, face_labels
        
        print(f"Mesh has {len(face_labels)} faces, downsampling for visualization...")
        
        # Simple downsampling by taking every nth face
        step = len(face_labels) // max_faces
        indices = np.arange(0, len(face_labels), step)
        
        # Create new mesh with subset of triangles
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        new_triangles = triangles[indices]
        new_face_labels = face_labels[indices]
        
        # Create vertex mapping and remove unused vertices
        used_vertices = np.unique(new_triangles.flatten())
        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        new_vertices = vertices[used_vertices]
        remapped_triangles = np.array([[vertex_map[v] for v in triangle] for triangle in new_triangles])
        
        # Create new mesh
        downsampled_mesh = o3d.geometry.TriangleMesh()
        downsampled_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
        downsampled_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
        
        print(f"Downsampled to {len(new_face_labels)} faces for visualization")
        return downsampled_mesh, new_face_labels
    
    def assign_vertex_colors_from_faces(self, mesh, face_labels):
        """Assign vertex colors based on face segment labels."""
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        # Initialize vertex colors (default gray)
        vertex_colors = np.full((len(vertices), 3), 0.5)
        vertex_votes = np.zeros(len(vertices))  # Track how many faces influence each vertex
        
        unique_labels = np.unique(face_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        # For each face, contribute its color to its vertices
        for label in valid_labels:
            face_mask = face_labels == label
            face_indices = np.where(face_mask)[0]
            
            color_idx = label % len(self.colors)
            color = self.colors[color_idx]
            
            # Get all vertices belonging to faces with this label
            face_triangles = triangles[face_mask]
            face_vertices = np.unique(face_triangles.flatten())
            
            # Add color contribution to these vertices
            vertex_colors[face_vertices] += color
            vertex_votes[face_vertices] += 1
        
        # Average colors for vertices influenced by multiple faces
        valid_votes = vertex_votes > 0
        vertex_colors[valid_votes] /= vertex_votes[valid_votes, np.newaxis]
        
        return vertex_colors
    
    def visualize_segmented_mesh(self, mesh, face_labels, save_path=None):
        """Visualize mesh with different colors for each face segment using Open3D."""
        
        # Check if only one segment found - suggest different parameters
        unique_labels = np.unique(face_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        if len(valid_labels) <= 1:
            print(f"\nWARNING: Only {len(valid_labels)} face segment(s) found!")
            print("This suggests the mesh is very smooth or angle threshold is too high.")
            print("Try running with a smaller angle threshold, e.g.:")
            print("  --angle_threshold 15.0   (for more sensitive edge detection)")
            print("  --angle_threshold 10.0   (for very sensitive edge detection)")
            print("  --min_face_size 500      (to allow smaller segments)")
            
            if len(valid_labels) == 0:
                print("No segments to visualize. Exiting.")
                return None
        
        # Downsample if necessary
        vis_mesh, vis_face_labels = self.downsample_for_visualization(mesh, face_labels)
        print(f"Found {len(np.unique(vis_face_labels[vis_face_labels >= 0]))} face segments in visualization")
        
        # Create vertex colors from face labels
        vertex_colors = self.assign_vertex_colors_from_faces(vis_mesh, vis_face_labels)
        
        # Create visualization mesh with vertex colors
        final_mesh = o3d.geometry.TriangleMesh()
        final_mesh.vertices = vis_mesh.vertices
        final_mesh.triangles = vis_mesh.triangles
        final_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        
        # Compute vertex normals for better lighting
        final_mesh.compute_vertex_normals()
        
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
        vis.add_geometry(final_mesh)
        
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
        
        return final_mesh
    
    def create_face_statistics_plot(self, face_labels, components, save_path=None):
        """Create a plot showing statistics about the segmented faces."""
        unique_labels = np.unique(face_labels)
        valid_labels = unique_labels[unique_labels >= 0]
        
        # Count faces per segment
        face_counts = []
        for label in valid_labels:
            count = np.sum(face_labels == label)
            face_counts.append(count)
        
        if len(face_counts) == 0:
            print("No face segments found to plot statistics")
            return
        
        # Create statistics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar plot of face counts per segment (limit to top 20 for readability)
        top_n = min(20, len(face_counts))
        top_indices = np.argsort(face_counts)[-top_n:]
        top_counts = [face_counts[i] for i in top_indices]
        top_colors = [self.colors[i % len(self.colors)] for i in range(top_n)]
        
        ax1.bar(range(len(top_counts)), top_counts, color=top_colors)
        ax1.set_xlabel('Face Segment (Top 20)')
        ax1.set_ylabel('Number of Triangles')
        ax1.set_title('Triangles per Face Segment')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of face segment sizes
        ax2.hist(face_counts, bins=min(20, len(face_counts)), 
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
    parser.add_argument('--angle_threshold', type=float, default=15.0, 
                       help='Angle threshold in degrees for sharp edge detection (default: 15.0)')
    parser.add_argument('--min_face_size', type=int, default=500,
                       help='Minimum number of triangles per face segment (default: 500)')
    parser.add_argument('--save_screenshot', type=str, default=None,
                       help='Path to save screenshot of visualization')
    parser.add_argument('--save_statistics', type=str, default=None,
                       help='Path to save statistics plot')
    parser.add_argument('--max_vis_faces', type=int, default=300000,
                       help='Maximum faces for visualization (default: 300000)')
    
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
    
    # Warn about large meshes
    if len(mesh.triangles) > 500000:
        print(f"WARNING: Large mesh detected ({len(mesh.triangles)} triangles)")
        print("This may take several minutes to process...")
    
    # Segment mesh
    face_labels, components = segmenter.segment_mesh(mesh)
    
    # Print results
    num_segments = len([label for label in np.unique(face_labels) if label >= 0])
    unassigned_faces = np.sum(face_labels == -1)
    
    print(f"\nSegmentation Results:")
    print(f"Number of face segments: {num_segments}")
    print(f"Unassigned triangles: {unassigned_faces}")
    print(f"Largest segment: {max([len(comp) for comp in components]) if components else 0} triangles")
    
    if num_segments == 0:
        print("No face segments found. Try adjusting --angle_threshold or --min_face_size parameters.")
        return
    
    # Visualize results
    vis_mesh = visualizer.visualize_segmented_mesh(
        mesh, face_labels, save_path=args.save_screenshot
    )
    
    # Create statistics plot
    visualizer.create_face_statistics_plot(
        face_labels, components, save_path=args.save_statistics
    )

if __name__ == "__main__":
    main()
