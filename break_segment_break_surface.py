import open3d as o3d
import numpy as np
import pickle
import os
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import SpectralClustering
import networkx as nx
from scipy.ndimage import gaussian_filter1d

class IntegralInvariants:
    """Compute integral invariant descriptors for surface analysis"""
    
    def __init__(self, mesh, radii=[0.005, 0.020]):  # 5mm and 20mm in meters
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.radii = radii
        self.kdtree = cKDTree(self.vertices)
        
    def compute_volume_invariant(self, point_idx, radius):
        """Compute volume under radius r - Vr(p)"""
        # Find neighbors within radius
        neighbors = self.kdtree.query_ball_point(self.vertices[point_idx], radius)
        
        if len(neighbors) < 4:
            return 0.0
            
        neighbor_points = self.vertices[neighbors]
        center = self.vertices[point_idx]
        normal = self.normals[point_idx]
        
        # Project neighbors onto plane defined by normal
        relative_points = neighbor_points - center
        projections = relative_points - np.outer(np.dot(relative_points, normal), normal)
        
        # Compute volume using spherical cap approximation
        distances = np.linalg.norm(relative_points, axis=1)
        heights = np.abs(np.dot(relative_points, normal))
        
        # Volume approximation
        valid_mask = distances < radius
        if np.sum(valid_mask) < 3:
            return 0.0
            
        volumes = np.pi * heights[valid_mask] * (3 * radius - heights[valid_mask]) / 3
        return np.sum(volumes) / len(neighbors)
    
    def compute_variation_descriptor(self, point_idx):
        """Compute variation across scales - VDr(p)"""
        volumes = []
        for radius in self.radii:
            vol = self.compute_volume_invariant(point_idx, radius)
            volumes.append(vol)
        
        if len(volumes) < 2:
            return 0.0
            
        return np.std(volumes) / (np.mean(volumes) + 1e-8)
    
    def compute_sharpness(self, point_idx, radius):
        """Compute local sharpness - svol(p)"""
        neighbors = self.kdtree.query_ball_point(self.vertices[point_idx], radius)
        
        if len(neighbors) < 4:
            return 0.0
            
        neighbor_normals = self.normals[neighbors]
        current_normal = self.normals[point_idx]
        
        # Measure normal variation
        dot_products = np.dot(neighbor_normals, current_normal)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(np.abs(dot_products))
        
        return np.mean(angles)
    
    def compute_surface_roughness(self, point_idx, radius):
        """Compute surface roughness - ek,r(p)"""
        neighbors = self.kdtree.query_ball_point(self.vertices[point_idx], radius)
        
        if len(neighbors) < 4:
            return 0.0
            
        neighbor_points = self.vertices[neighbors]
        center = self.vertices[point_idx]
        normal = self.normals[point_idx]
        
        # Project onto tangent plane and measure deviation
        relative_points = neighbor_points - center
        plane_projections = relative_points - np.outer(np.dot(relative_points, normal), normal)
        heights = np.abs(np.dot(relative_points, normal))
        
        # Roughness as RMS height variation
        return np.sqrt(np.mean(heights**2))

class MultiScaleEdgeExtractor:
    """Extract geometric edges using multi-scale analysis"""
    
    def __init__(self, mesh, radii=[0.005, 0.020]):
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.normals = np.asarray(mesh.vertex_normals)
        self.faces = np.asarray(mesh.triangles)
        self.radii = radii
        self.kdtree = cKDTree(self.vertices)
        
    def compute_principal_curvatures(self):
        """Compute principal curvatures for each vertex"""
        curvatures = []
        
        for i, vertex in enumerate(self.vertices):
            # Find local neighborhood
            neighbors = self.kdtree.query_ball_point(vertex, self.radii[0])
            
            if len(neighbors) < 6:
                curvatures.append((0.0, 0.0))
                continue
                
            # Fit local quadric surface
            neighbor_points = self.vertices[neighbors]
            normal = self.normals[i]
            
            # Create local coordinate system
            center = vertex
            z_axis = normal
            x_axis = np.cross(z_axis, [0, 0, 1])
            if np.linalg.norm(x_axis) < 0.1:
                x_axis = np.cross(z_axis, [1, 0, 0])
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # Project to local coordinates
            relative_points = neighbor_points - center
            local_x = np.dot(relative_points, x_axis)
            local_y = np.dot(relative_points, y_axis)
            local_z = np.dot(relative_points, z_axis)
            
            # Fit quadric: z = ax² + bxy + cy²
            if len(local_x) >= 6:
                A = np.column_stack([local_x**2, local_x * local_y, local_y**2, local_x, local_y, np.ones(len(local_x))])
                try:
                    coeffs = np.linalg.lstsq(A, local_z, rcond=None)[0]
                    a, b, c = coeffs[0], coeffs[1], coeffs[2]
                    
                    # Principal curvatures from quadric coefficients
                    H = (a + c) / 2  # Mean curvature
                    K = a * c - (b/2)**2  # Gaussian curvature
                    
                    discriminant = H**2 - K
                    if discriminant >= 0:
                        k1 = H + np.sqrt(discriminant)
                        k2 = H - np.sqrt(discriminant)
                    else:
                        k1 = k2 = H
                        
                    curvatures.append((k1, k2))
                except:
                    curvatures.append((0.0, 0.0))
            else:
                curvatures.append((0.0, 0.0))
                
        return np.array(curvatures)
    
    def extract_edge_points(self, curvature_threshold=0.1):
        """Extract high-curvature edge points"""
        curvatures = self.compute_principal_curvatures()
        
        # Edge strength based on maximum absolute curvature
        edge_strength = np.max(np.abs(curvatures), axis=1)
        
        # Debug: Print curvature statistics
        print(f"   Curvature stats: min={edge_strength.min():.6f}, max={edge_strength.max():.6f}, mean={edge_strength.mean():.6f}")
        print(f"   Using threshold: {curvature_threshold:.6f}")
        
        # If threshold is too high, use adaptive threshold
        if np.sum(edge_strength > curvature_threshold) == 0:
            # Use percentile-based threshold
            adaptive_threshold = np.percentile(edge_strength, 90)  # Top 10%
            print(f"   No edges found with threshold {curvature_threshold:.6f}")
            print(f"   Using adaptive threshold (90th percentile): {adaptive_threshold:.6f}")
            edge_mask = edge_strength > adaptive_threshold
        else:
            edge_mask = edge_strength > curvature_threshold
        
        edge_indices = np.where(edge_mask)[0]
        
        return edge_indices, edge_strength
    
    def build_connectivity_graph(self, edge_indices, edge_strength):
        """Build connectivity graph of edge points"""
        if len(edge_indices) < 2:
            return nx.Graph()
            
        edge_points = self.vertices[edge_indices]
        
        # Build graph connecting nearby edge points
        graph = nx.Graph()
        
        # Add nodes with their properties
        for i, idx in enumerate(edge_indices):
            graph.add_node(i, vertex_idx=idx, strength=edge_strength[idx], 
                         position=edge_points[i])
        
        # Connect nearby edge points
        edge_kdtree = cKDTree(edge_points)
        connection_radius = self.radii[0] * 2  # Twice the smallest radius
        
        for i, point in enumerate(edge_points):
            neighbors = edge_kdtree.query_ball_point(point, connection_radius)
            for j in neighbors:
                if i != j:
                    distance = np.linalg.norm(edge_points[i] - edge_points[j])
                    # Weight by inverse distance and edge strength
                    weight = 1.0 / (distance + 1e-6) * (edge_strength[edge_indices[i]] + edge_strength[edge_indices[j]])
                    graph.add_edge(i, j, weight=weight, distance=distance)
        
        return graph
    
    def extract_boundary_loops(self, graph):
        """Extract boundary loops using MST + cycle detection"""
        if len(graph.nodes()) < 3:
            return []
            
        # Create MST
        mst = nx.minimum_spanning_tree(graph, weight='weight')
        
        # Find cycles by adding back edges not in MST
        loops = []
        
        for edge in graph.edges():
            if not mst.has_edge(*edge):
                # This edge creates a cycle
                try:
                    path = nx.shortest_path(mst, edge[0], edge[1])
                    if len(path) > 3:  # Valid loop
                        loops.append(path)
                except:
                    continue
        
        # Filter and merge nearby loops
        filtered_loops = []
        for loop in loops:
            if len(loop) >= 4:  # Minimum loop size
                filtered_loops.append(loop)
        
        return filtered_loops

class SurfaceSegmenter:
    """Segment mesh into surface patches using boundary loops"""
    
    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.triangles)
        
    def segment_faces(self, boundary_loops, edge_indices):
        """Segment mesh into discrete surface regions"""
        # Create face labels
        face_labels = np.zeros(len(self.faces), dtype=int)
        
        if not boundary_loops:
            # Fallback: use simple region growing based on curvature
            print("   No boundary loops found, using curvature-based segmentation...")
            return self.curvature_based_segmentation()
        
        # For each boundary loop, identify faces it encloses
        for loop_id, loop in enumerate(boundary_loops):
            if len(loop) < 3:
                continue
                
            # Get actual vertex indices from edge indices
            loop_vertices = [edge_indices[i] for i in loop if i < len(edge_indices)]
            
            if len(loop_vertices) < 3:
                continue
            
            # Find faces that contain vertices from this loop
            for face_idx, face in enumerate(self.faces):
                # Check if face vertices are "inside" the loop region
                face_center = np.mean(self.vertices[face], axis=0)
                
                # Simple containment check based on distance to loop vertices
                loop_positions = self.vertices[loop_vertices]
                distances = np.linalg.norm(loop_positions - face_center, axis=1)
                
                if np.min(distances) < 0.1:  # Threshold for containment
                    face_labels[face_idx] = loop_id + 1
        
        return face_labels
    
    def curvature_based_segmentation(self):
        """Fallback segmentation based on face curvatures"""
        face_curvatures = self.compute_face_curvatures()
        
        # Simple thresholding - smooth faces vs rough faces
        smooth_threshold = np.percentile(face_curvatures, 30)  # Bottom 30% as smooth
        
        face_labels = np.zeros(len(self.faces), dtype=int)
        face_labels[face_curvatures <= smooth_threshold] = 1  # Smooth faces (potential breaks)
        
        print(f"   Curvature-based segmentation: {np.sum(face_labels == 1)} smooth faces, {np.sum(face_labels == 0)} rough faces")
        
        return face_labels
    
    def merge_oversegmented_regions(self, face_labels, similarity_threshold=0.8):
        """Merge over-segmented regions using curvature similarity"""
        # Compute face curvatures
        face_curvatures = self.compute_face_curvatures()
        
        unique_labels = np.unique(face_labels)
        merged_labels = face_labels.copy()
        
        # Merge similar adjacent regions
        for label1 in unique_labels:
            if label1 == 0:  # Skip background
                continue
                
            mask1 = face_labels == label1
            curvature1 = np.mean(face_curvatures[mask1])
            
            for label2 in unique_labels:
                if label2 <= label1 or label2 == 0:
                    continue
                    
                mask2 = face_labels == label2
                curvature2 = np.mean(face_curvatures[mask2])
                
                # Check similarity
                similarity = 1.0 / (1.0 + abs(curvature1 - curvature2))
                
                if similarity > similarity_threshold:
                    # Check adjacency
                    if self.are_regions_adjacent(mask1, mask2):
                        merged_labels[mask2] = label1
        
        return merged_labels
    
    def compute_face_curvatures(self):
        """Compute curvature for each face"""
        face_curvatures = np.zeros(len(self.faces))
        
        for i, face in enumerate(self.faces):
            # Get face vertices and normals
            face_vertices = self.vertices[face]
            face_normals = self.mesh.vertex_normals
            face_normal_vectors = np.array([face_normals[j] for j in face])
            
            # Estimate curvature from normal variation
            normal_variations = []
            for j in range(3):
                for k in range(j+1, 3):
                    dot_product = np.dot(face_normal_vectors[j], face_normal_vectors[k])
                    dot_product = np.clip(dot_product, -1, 1)
                    angle = np.arccos(abs(dot_product))
                    normal_variations.append(angle)
            
            face_curvatures[i] = np.mean(normal_variations)
        
        return face_curvatures
    
    def are_regions_adjacent(self, mask1, mask2):
        """Check if two face regions are adjacent"""
        faces1 = np.where(mask1)[0]
        faces2 = np.where(mask2)[0]
        
        # Check if any faces share vertices
        for f1 in faces1:
            vertices1 = set(self.faces[f1])
            for f2 in faces2:
                vertices2 = set(self.faces[f2])
                if len(vertices1.intersection(vertices2)) > 0:
                    return True
        
        return False

class FragmentClassifier:
    """Classify surface patches as original or break surfaces"""
    
    def __init__(self, mesh, radii=[0.005, 0.020]):
        self.mesh = mesh
        self.vertices = np.asarray(mesh.vertices)
        self.radii = radii
        self.integral_invariants = IntegralInvariants(mesh, radii)
        
    def compute_surface_descriptors(self, face_labels):
        """Compute integral invariant descriptors for each surface patch"""
        unique_labels = np.unique(face_labels)
        descriptors = {}
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            # Get faces belonging to this patch
            patch_faces = np.where(face_labels == label)[0]
            
            if len(patch_faces) == 0:
                continue
            
            # Get vertices from these faces
            patch_vertices = set()
            for face_idx in patch_faces:
                patch_vertices.update(self.mesh.triangles[face_idx])
            
            patch_vertices = list(patch_vertices)
            
            if len(patch_vertices) < 10:  # Skip very small patches
                continue
            
            # Compute descriptors for patch vertices
            vr_values = []
            vdr_values = []
            svol_values = []
            roughness_values = []
            
            for vertex_idx in patch_vertices:
                # Volume invariant at largest radius
                vr = self.integral_invariants.compute_volume_invariant(vertex_idx, self.radii[-1])
                vr_values.append(vr)
                
                # Variation across scales
                vdr = self.integral_invariants.compute_variation_descriptor(vertex_idx)
                vdr_values.append(vdr)
                
                # Sharpness
                svol = self.integral_invariants.compute_sharpness(vertex_idx, self.radii[0])
                svol_values.append(svol)
                
                # Surface roughness
                roughness = self.integral_invariants.compute_surface_roughness(vertex_idx, self.radii[0])
                roughness_values.append(roughness)
            
            # Aggregate descriptors for the patch
            descriptors[label] = {
                'vr_mean': np.mean(vr_values),
                'vr_std': np.std(vr_values),
                'vdr_mean': np.mean(vdr_values),
                'vdr_std': np.std(vdr_values),
                'svol_mean': np.mean(svol_values),
                'svol_std': np.std(svol_values),
                'roughness_mean': np.mean(roughness_values),
                'roughness_std': np.std(roughness_values),
                'patch_size': len(patch_vertices)
            }
        
        return descriptors
    
    def classify_surfaces(self, descriptors, roughness_threshold=0.01, sharpness_threshold=0.5):
        """Classify surfaces as original (0) or break (1)"""
        classifications = {}
        
        for label, desc in descriptors.items():
            # Classification rules based on surface characteristics
            # Break surfaces: low roughness, low sharpness, more uniform
            # Original surfaces: high roughness, high sharpness, more variation
            
            is_break = (
                desc['roughness_mean'] < roughness_threshold and
                desc['svol_mean'] < sharpness_threshold and
                desc['vdr_std'] < 0.1  # Low variation across scales
            )
            
            classifications[label] = 1 if is_break else 0
        
        return classifications

class MultiScaleFragmentSegmenter:
    """Main class combining all segmentation components"""
    
    def __init__(self, radii=[0.005, 0.020], curvature_threshold=0.01):  # Lower default threshold
        self.radii = radii
        self.curvature_threshold = curvature_threshold
        
    def process_fragment(self, ply_path, save_pkl=True, output_dir=None, visualize=True):
        """Process fragment using multi-scale edge-based segmentation"""
        print(f"\nProcessing: {ply_path}")
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        
        if len(mesh.vertices) == 0:
            print(f"Error: Could not load mesh from {ply_path}")
            return None
        
        # Ensure normals are computed
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        print("1. Extracting multi-scale edges...")
        edge_extractor = MultiScaleEdgeExtractor(mesh, self.radii)
        edge_indices, edge_strength = edge_extractor.extract_edge_points(self.curvature_threshold)
        
        print(f"   Found {len(edge_indices)} edge points")
        
        print("2. Building connectivity graph...")
        connectivity_graph = edge_extractor.build_connectivity_graph(edge_indices, edge_strength)
        
        print(f"   Graph has {len(connectivity_graph.nodes())} nodes and {len(connectivity_graph.edges())} edges")
        
        print("3. Extracting boundary loops...")
        boundary_loops = edge_extractor.extract_boundary_loops(connectivity_graph)
        
        print(f"   Found {len(boundary_loops)} boundary loops")
        
        print("4. Segmenting faces...")
        segmenter = SurfaceSegmenter(mesh)
        face_labels = segmenter.segment_faces(boundary_loops, edge_indices)
        merged_labels = segmenter.merge_oversegmented_regions(face_labels)
        
        unique_patches = len(np.unique(merged_labels)) - 1  # Exclude background
        print(f"   Created {unique_patches} surface patches")
        
        print("5. Computing integral invariants...")
        classifier = FragmentClassifier(mesh, self.radii)
        descriptors = classifier.compute_surface_descriptors(merged_labels)
        
        print("6. Classifying surfaces...")
        classifications = classifier.classify_surfaces(descriptors)
        
        # Create classification mask for vertices
        vertex_classifications = np.zeros(len(mesh.vertices))
        for face_idx, label in enumerate(merged_labels):
            if label in classifications:
                face_vertices = mesh.triangles[face_idx]
                vertex_classifications[face_vertices] = classifications[label]
        
        break_mask = vertex_classifications == 1
        original_mask = vertex_classifications == 0
        
        print(f"   Classified {np.sum(break_mask)} vertices as break surfaces")
        print(f"   Classified {np.sum(original_mask)} vertices as original surfaces")
        
        # Create colored mesh
        colored_mesh = self.create_colored_mesh(mesh, break_mask)
        
        # Save results
        if save_pkl and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            pkl_path = output_dir / f"{Path(ply_path).stem}_multiscale_segmented.pkl"
            self.save_segmented_data(mesh, break_mask, original_mask, descriptors, 
                                   classifications, merged_labels, pkl_path)
        
        # Visualize
        if visualize:
            self.visualize_segmentation(colored_mesh, break_mask, descriptors)
        
        return {
            'mesh': mesh,
            'colored_mesh': colored_mesh,
            'break_mask': break_mask,
            'original_mask': original_mask,
            'descriptors': descriptors,
            'classifications': classifications,
            'face_labels': merged_labels
        }
    
    def create_colored_mesh(self, mesh, break_mask):
        """Create colored mesh highlighting break surfaces"""
        colored_mesh = o3d.geometry.TriangleMesh()
        colored_mesh.vertices = mesh.vertices
        colored_mesh.triangles = mesh.triangles
        colored_mesh.vertex_normals = mesh.vertex_normals
        
        # Only color break surfaces in red
        colors = np.zeros((len(mesh.vertices), 3))
        colors[break_mask] = [1.0, 0.2, 0.2]  # Red for break surfaces
        
        if np.any(break_mask):
            colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        return colored_mesh
    
    def save_segmented_data(self, mesh, break_mask, original_mask, descriptors, 
                          classifications, face_labels, output_path):
        """Save comprehensive segmentation data"""
        segmentation_data = {
            'vertices': np.asarray(mesh.vertices),
            'faces': np.asarray(mesh.triangles),
            'vertex_normals': np.asarray(mesh.vertex_normals),
            'break_mask': break_mask,
            'original_mask': original_mask,
            'descriptors': descriptors,
            'classifications': classifications,
            'face_labels': face_labels,
            'radii_used': self.radii,
            'curvature_threshold': self.curvature_threshold
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(segmentation_data, f)
        
        print(f"Multi-scale segmentation data saved to: {output_path}")
    
    def visualize_segmentation(self, colored_mesh, break_mask, descriptors):
        """Visualize segmentation results"""
        print("\nVisualization Instructions:")
        print("- Original color: Original fragment surfaces")
        print("- Red surfaces: Break/smooth surfaces (joining surfaces)")
        print("- Press 'Q' to close the visualization")
        print(f"\nSurface Analysis Results:")
        
        for label, desc in descriptors.items():
            surface_type = "BREAK" if desc['roughness_mean'] < 0.1 else "ORIGINAL"
            print(f"  Patch {label}: {surface_type}")
            print(f"    - Roughness: {desc['roughness_mean']:.4f}")
            print(f"    - Sharpness: {desc['svol_mean']:.4f}")
            print(f"    - Size: {desc['patch_size']} vertices")
        
        # Create visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Multi-Scale Fragment Segmentation - Red=Break Surfaces")
        vis.add_geometry(colored_mesh)
        
        # Set visualization options
        render_option = vis.get_render_option()
        render_option.show_coordinate_frame = True
        render_option.background_color = [0.1, 0.1, 0.1]
        
        vis.run()
        vis.destroy_window()

def process_fragment_folder(folder_path, save_pkl=True, output_dir=None, 
                          radii=[0.005, 0.020], curvature_threshold=0.1):  # Lower default threshold
    """Process all PLY files in a folder using multi-scale segmentation"""
    folder_path = Path(folder_path)
    segmenter = MultiScaleFragmentSegmenter(radii=radii, curvature_threshold=curvature_threshold)
    
    if not folder_path.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return
    
    ply_files = list(folder_path.glob("*.ply"))
    
    if not ply_files:
        print(f"No PLY files found in {folder_path}")
        return
    
    print(f"Found {len(ply_files)} PLY files")
    
    for ply_file in ply_files:
        try:
            result = segmenter.process_fragment(ply_file, save_pkl=save_pkl, 
                                              output_dir=output_dir, visualize=True)
            if result:
                print(f"Successfully processed {ply_file.name}")
        except Exception as e:
            print(f"Error processing {ply_file.name}: {e}")

# Example usage
if __name__ == "__main__":
    # Process your fragment folder with multi-scale edge-based segmentation
    fragment_folder = "manual_fragments/frag_3"
    
    # Multi-scale radii: 5mm and 20mm (converted to meters)
    radii = [0.005, 0.020]
    
    # Lower curvature threshold for edge detection
    curvature_threshold = 0.1  # Much more sensitive
    
    # Process fragments
    process_fragment_folder(fragment_folder, save_pkl=False, 
                          output_dir="results", 
                          radii=radii, curvature_threshold=curvature_threshold)