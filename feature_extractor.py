import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    warnings.warn("open3d not available. Feature extraction will not work.")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Using basic nearest neighbor search.")

class FeatureDict(TypedDict):
    """Type for feature dictionary"""
    points: np.ndarray
    point_features: np.ndarray
    global_features: np.ndarray
    num_points: int
    feature_dim: int
    file_path: str
    fragment_name: str

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    target_points: int = 30000  # Target number of points after downsampling
    voxel_size: Optional[float] = None  # Auto-computed if None
    
    # Normalization
    normalize_scale: bool = True
    normalize_position: bool = True
    
    # Normal computation
    compute_normals: bool = True
    normal_radius: float = 0.05
    normal_max_nn: int = 30
    
    # Geometric descriptors
    compute_fpfh: bool = True
    fpfh_radius: float = 0.1
    fpfh_max_nn: int = 100
    
    compute_curvature: bool = True
    curvature_radius: float = 0.05
    curvature_max_nn: int = 30
    
    # Global features
    compute_global_features: bool = True
    
    # Color features
    compute_color_features: bool = True
    
    # Point-wise features
    include_position: bool = True
    include_normals: bool = True
    include_descriptors: bool = True

class FeatureExtractor:
    """
    Comprehensive feature extractor for point cloud fragments.
    Handles adaptive downsampling, normalization, and various feature computations.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """Initialize feature extractor with configuration."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for feature extraction")
        self.config = FeatureConfig() if config is None else config
        
    def load_point_cloud(self, file_path: str) -> Optional[o3d.geometry.PointCloud]:
        """Load point cloud from file with error handling."""
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                print(f"Warning: Empty point cloud in {file_path}")
                return None
            return pcd
        except Exception as e:
            print(f"Error loading point cloud {file_path}: {e}")
            return None
    
    def adaptive_downsample(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Adaptively downsample point cloud to target density."""
        if len(pcd.points) <= self.config.target_points:
            return pcd
            
        if self.config.voxel_size is not None:
            return pcd.voxel_down_sample(self.config.voxel_size)
        
        # Compute adaptive voxel size
        points = np.asarray(pcd.points)
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        volume = np.prod(bbox_size)
        point_density = len(points) / volume
        target_density = self.config.target_points / volume
        voxel_size = (point_density / target_density) ** (1/3)
        
        # Downsample with computed voxel size
        downsampled = pcd.voxel_down_sample(voxel_size)
        
        # If still too many points, use uniform sampling
        if len(downsampled.points) > self.config.target_points:
            downsampled = downsampled.uniform_down_sample(
                int(len(downsampled.points) / self.config.target_points)
            )
        
        return downsampled
    
    def normalize_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Normalize point cloud scale and position."""
        points = np.asarray(pcd.points)
        
        if self.config.normalize_position:
            # Center at origin
            centroid = np.mean(points, axis=0)
            points = points - centroid
        
        if self.config.normalize_scale:
            # Scale to unit sphere
            scale = np.max(np.linalg.norm(points, axis=1))
            if scale > 0:
                points = points / scale
        
        # Update point cloud
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd
    
    def compute_normals(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Compute point cloud normals."""
        if not self.config.compute_normals:
            return pcd
            
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.config.normal_radius,
                    max_nn=self.config.normal_max_nn
                )
            )
        return pcd
    
    def compute_geometric_descriptors(self, pcd: o3d.geometry.PointCloud) -> Dict[str, np.ndarray]:
        """Compute geometric descriptors with proper normalization and error handling."""
        descriptors = {}
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        # Compute local curvature if enabled
        if self.config.compute_curvature and normals is not None:
            try:
                curvature = self._compute_curvature(points, normals)
                # Normalize curvature to [0, 1] range
                if len(curvature) > 0:
                    curvature = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature) + 1e-6)
                descriptors['curvature'] = curvature
            except Exception as e:
                print(f"Warning: Curvature computation failed: {e}")
                descriptors['curvature'] = np.zeros(len(points))
        
        # Compute FPFH if enabled
        if self.config.compute_fpfh:
            try:
                # Ensure normals are computed
                if not pcd.has_normals():
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=self.config.fpfh_radius,
                            max_nn=self.config.fpfh_max_nn
                        )
                    )
                
                # Compute FPFH features
                fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd,
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=self.config.fpfh_radius,
                        max_nn=self.config.fpfh_max_nn
                    )
                )
                fpfh_data = np.asarray(fpfh.data).T  # [N, 33]
                
                # Normalize FPFH features
                fpfh_data = np.nan_to_num(fpfh_data)  # Replace NaN with 0
                if fpfh_data.shape[0] > 0:
                    # Normalize each feature dimension independently
                    fpfh_data = (fpfh_data - np.mean(fpfh_data, axis=0, keepdims=True)) / (np.std(fpfh_data, axis=0, keepdims=True) + 1e-6)
                
                descriptors['fpfh'] = fpfh_data
                
            except Exception as e:
                print(f"Warning: FPFH computation failed: {e}")
                descriptors['fpfh'] = np.zeros((len(points), 33))
        
        # Compute local geometric features
        try:
            # Local point density
            if SKLEARN_AVAILABLE:
                nbrs = NearestNeighbors(n_neighbors=min(30, len(points)), algorithm='auto')
                nbrs.fit(points)
                distances, _ = nbrs.kneighbors(points)
                local_density = 1.0 / (np.mean(distances, axis=1) + 1e-6)
                # Normalize density
                local_density = (local_density - np.min(local_density)) / (np.max(local_density) - np.min(local_density) + 1e-6)
                descriptors['local_density'] = local_density
            else:
                descriptors['local_density'] = np.ones(len(points))
            
            # Height features
            if points.shape[0] > 0:
                height = points[:, 2] - np.min(points[:, 2])  # Assuming Z is up
                height = height / (np.max(height) + 1e-6)  # Normalize to [0, 1]
                descriptors['height'] = height
            else:
                descriptors['height'] = np.zeros(len(points))
                
        except Exception as e:
            print(f"Warning: Local feature computation failed: {e}")
            descriptors['local_density'] = np.ones(len(points))
            descriptors['height'] = np.zeros(len(points))
        
        return descriptors
    
    def _compute_curvature(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Compute local curvature for each point."""
        if not SKLEARN_AVAILABLE:
            return np.zeros(len(points))
            
        nbrs = NearestNeighbors(n_neighbors=min(self.config.curvature_max_nn, len(points)), algorithm='auto')
        nbrs.fit(points)
        _, indices = nbrs.kneighbors(points)
        
        curvatures = []
        for i, neighbors in enumerate(indices):
            neighbor_normals = normals[neighbors]
            normal_variation = np.mean(np.abs(1 - np.dot(neighbor_normals, normals[i])))
            curvatures.append(normal_variation)
        
        return np.array(curvatures)
    
    def compute_global_features(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Compute global features for the point cloud."""
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        features = []
        
        # Basic shape features
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_points = np.asarray(bbox.get_box_points())
        bbox_size = np.max(bbox_points, axis=0) - np.min(bbox_points, axis=0)
        
        # Volume and surface area approximation
        volume = np.prod(bbox_size)
        surface_area = 2 * (bbox_size[0]*bbox_size[1] + bbox_size[1]*bbox_size[2] + bbox_size[0]*bbox_size[2])
        
        # Point distribution features
        centroid = np.mean(points, axis=0)
        distances_to_centroid = np.linalg.norm(points - centroid, axis=1)
        avg_distance = np.mean(distances_to_centroid)
        std_distance = np.std(distances_to_centroid)
        
        # Combine all global features
        features.extend([
            volume,
            surface_area,
            avg_distance,
            std_distance,
            bbox_size[0], bbox_size[1], bbox_size[2],  # Width, height, depth
            bbox_size[0]/bbox_size[1],  # Aspect ratio 1
            bbox_size[1]/bbox_size[2],  # Aspect ratio 2
            bbox_size[0]/bbox_size[2],  # Aspect ratio 3
        ])
        
        if normals is not None:
            # Normal distribution features
            normal_angles = np.arccos(np.clip(normals @ [0, 0, 1], -1, 1))
            features.extend([
                np.mean(normal_angles),
                np.std(normal_angles),
                np.percentile(normal_angles, 25),
                np.percentile(normal_angles, 75),
            ])
        
        return np.array(features)
    
    def compute_color_features(self, pcd: o3d.geometry.PointCloud) -> Optional[np.ndarray]:
        """Compute color features if available."""
        if not self.config.compute_color_features:
            return None
            
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        if colors is None or len(colors) == 0:
            return None
            
        return colors  # Shape: (N, 3)
    
    def extract_mesh_boundary(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract boundary vertices and their features from a mesh."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for mesh boundary extraction")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            if not mesh.has_vertices():
                print(f"Warning: Empty mesh in {file_path}")
                return None
            mesh.compute_vertex_normals()
            
            # First try: use Open3D's built-in boundary detection
            mesh.compute_adjacency_list()
            vertex_adj = np.asarray(mesh.adjacency_list)
            boundary_vertex_indices = []
            for i, adj in enumerate(vertex_adj):
                if len(adj) > 0 and len(adj) != 6:  # Non-manifold vertex
                    boundary_vertex_indices.append(i)
            boundary_vertex_indices = np.array(boundary_vertex_indices, dtype=int)
            print(f"[DEBUG] {file_path}: Found {len(boundary_vertex_indices)} initial boundary vertices")
            
            if len(boundary_vertex_indices) > 0:
                # Filter vertices by checking their neighborhood
                filtered_boundary_indices = []
                for idx in boundary_vertex_indices:
                    adj = vertex_adj[idx]
                    if len(adj) > 0:
                        # Count how many neighbors are also boundary vertices
                        boundary_neighbors = sum(1 for n in adj if n in boundary_vertex_indices)
                        # Keep vertices that have 1-3 boundary neighbors
                        if 1 <= boundary_neighbors <= 3:
                            filtered_boundary_indices.append(idx)
                
                boundary_vertex_indices = np.array(filtered_boundary_indices, dtype=int)
                print(f"[DEBUG] {file_path}: Found {len(boundary_vertex_indices)} boundary vertices after filtering")
            
            if len(boundary_vertex_indices) == 0:
                # Second try: use triangle adjacency
                triangles = np.asarray(mesh.triangles)
                vertices = np.asarray(mesh.vertices)
                print(f"[DEBUG] {file_path}: Found {len(triangles)} triangles")
                
                # Create edge to triangle mapping
                edge_to_triangles = {}
                for tri_idx, tri in enumerate(triangles):
                    for i in range(3):
                        edge = tuple(sorted((tri[i], tri[(i+1)%3])))
                        if edge not in edge_to_triangles:
                            edge_to_triangles[edge] = []
                        edge_to_triangles[edge].append(tri_idx)
                
                # Find edges with exactly one adjacent triangle
                boundary_edges = [edge for edge, tris in edge_to_triangles.items() if len(tris) == 1]
                print(f"[DEBUG] {file_path}: Found {len(boundary_edges)} boundary edges")
                
                # Get unique boundary vertices from edges
                boundary_vertex_indices = np.array(sorted(list(set([idx for edge in boundary_edges for idx in edge]))), dtype=int)
            
            # Get coordinates and other attributes for boundary vertices
            vertices = np.asarray(mesh.vertices)
            boundary_coords = vertices[boundary_vertex_indices]
            if mesh.has_vertex_normals():
                boundary_normals = np.asarray(mesh.vertex_normals)[boundary_vertex_indices]
            if mesh.has_vertex_colors():
                boundary_colors = np.asarray(mesh.vertex_colors)[boundary_vertex_indices]
            
            # Create sequential indices for the boundary vertices
            sequential_indices = np.arange(len(boundary_vertex_indices))
            
            # Downsample boundary vertices if too many
            MAX_BOUNDARY_VERTICES = 500  # Reduced from 750 to get cleaner boundaries
            if len(boundary_vertex_indices) > MAX_BOUNDARY_VERTICES:
                # Use uniform sampling to select indices
                step = len(boundary_vertex_indices) // MAX_BOUNDARY_VERTICES
                # Keep the original vertex indices but sample them uniformly
                sampled_indices = np.arange(0, len(boundary_vertex_indices), step)
                # Update all arrays using the sampled indices
                boundary_vertex_indices = boundary_vertex_indices[sampled_indices]
                boundary_coords = boundary_coords[sampled_indices]
                sequential_indices = sequential_indices[sampled_indices]
                if mesh.has_vertex_normals():
                    boundary_normals = boundary_normals[sampled_indices]
                if mesh.has_vertex_colors():
                    boundary_colors = boundary_colors[sampled_indices]
                print(f"[DEBUG] {file_path}: Downsampled to {len(boundary_vertex_indices)} boundary vertices")
            
            return {
                'boundary_indices': sequential_indices,  # Sequential indices for edge extraction
                'original_indices': boundary_vertex_indices,  # Original mesh vertex indices
                'boundary_coords': boundary_coords,
                'boundary_normals': boundary_normals if mesh.has_vertex_normals() else None,
                'boundary_colors': boundary_colors if mesh.has_vertex_colors() else None,
                'boundary_edges': np.array(boundary_edges, dtype=int) if 'boundary_edges' in locals() else np.zeros((0, 2), dtype=int)
            }
        except Exception as e:
            print(f"Error extracting mesh boundary from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_point_cloud_boundary(self, file_path: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract boundary points from a point cloud using local density and curvature analysis."""
        if not OPEN3D_AVAILABLE:
            raise ImportError("Open3D is required for point cloud boundary extraction")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist")
            return None
        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                print(f"Warning: Empty point cloud in {file_path}")
                return None
            
            # Compute normals with adaptive radius
            points = np.asarray(pcd.points)
            bbox = pcd.get_axis_aligned_bounding_box()
            bbox_size = bbox.get_extent()
            adaptive_radius = min(bbox_size) * 0.05
            
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=adaptive_radius,
                    max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=30)
            
            # Compute local features for boundary detection
            normals = np.asarray(pcd.normals)
            
            # 1. Local density analysis
            nbrs = NearestNeighbors(n_neighbors=min(30, len(points)), algorithm='auto')
            nbrs.fit(points)
            distances, indices = nbrs.kneighbors(points)
            local_density = 1.0 / (np.mean(distances, axis=1) + 1e-6)
            
            # 2. Normal variation (curvature)
            curvatures = []
            for i, neighbors in enumerate(indices):
                neighbor_normals = normals[neighbors]
                normal_variation = np.mean(np.abs(1 - np.dot(neighbor_normals, normals[i])))
                curvatures.append(normal_variation)
            curvatures = np.array(curvatures)
            
            # 3. Identify boundary points using density and curvature
            density_threshold = np.percentile(local_density, 25)  # Lower density indicates boundary
            curvature_threshold = np.percentile(curvatures, 75)  # Higher curvature indicates boundary
            
            boundary_mask = (local_density < density_threshold) & (curvatures > curvature_threshold)
            boundary_indices = np.where(boundary_mask)[0]
            
            # 4. Filter isolated boundary points
            filtered_boundary_indices = []
            for idx in boundary_indices:
                neighbors = indices[idx]
                boundary_neighbors = sum(1 for n in neighbors if n in boundary_indices)
                if boundary_neighbors >= 2:  # Keep points with at least 2 boundary neighbors
                    filtered_boundary_indices.append(idx)
            
            boundary_indices = np.array(filtered_boundary_indices)
            
            # Downsample if too many boundary points
            MAX_BOUNDARY_POINTS = 500
            if len(boundary_indices) > MAX_BOUNDARY_POINTS:
                step = len(boundary_indices) // MAX_BOUNDARY_POINTS
                boundary_indices = boundary_indices[::step]
            
            # Sort boundary points to maintain a sequential order
            boundary_coords = points[boundary_indices]
            # Project points onto their principal components to get a rough ordering
            pca = PCA(n_components=2)
            projected = pca.fit_transform(boundary_coords)
            angles = np.arctan2(projected[:, 1], projected[:, 0])
            sort_idx = np.argsort(angles)
            
            return {
                'boundary_indices': boundary_indices[sort_idx],
                'boundary_coords': boundary_coords[sort_idx],
                'boundary_normals': normals[boundary_indices[sort_idx]] if pcd.has_normals() else None,
                'boundary_colors': np.asarray(pcd.colors)[boundary_indices[sort_idx]] if pcd.has_colors() else None,
                'local_density': local_density[boundary_indices[sort_idx]],
                'curvature': curvatures[boundary_indices[sort_idx]]
            }
            
        except Exception as e:
            print(f"Error extracting point cloud boundary from {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def extract_features(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Main feature extraction pipeline.
        
        Returns:
            Dictionary containing all extracted features or None if failed
        """
        print(f"Extracting features from: {os.path.basename(file_path)}")
        
        # Load point cloud
        pcd = self.load_point_cloud(file_path)
        if pcd is None:
            return None
        
        # Adaptive downsampling
        pcd = self.adaptive_downsample(pcd)
        
        # Normalize scale and position
        pcd = self.normalize_point_cloud(pcd)
        
        # Compute normals
        pcd = self.compute_normals(pcd)
        
        # Extract all features
        features: Dict[str, Any] = {
            'file_path': file_path,
            'fragment_name': os.path.basename(file_path)
        }
        
        # Point coordinates
        points = np.asarray(pcd.points)
        features['points'] = points  # Ensure 'points' is always in the features dict
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # Point-wise features
        pointwise_features = []
        
        if self.config.include_position:
            pointwise_features.append(points)  # (N, 3)
        
        if self.config.include_normals and normals is not None:
            pointwise_features.append(normals)  # (N, 3)
        
        # Geometric descriptors
        descriptors = self.compute_geometric_descriptors(pcd)
        if self.config.include_descriptors:
            for desc_name, desc_values in descriptors.items():
                # Ensure all descriptors are 2D (N, F)
                if desc_values.ndim == 1:
                    desc_values = desc_values[:, None]
                pointwise_features.append(desc_values)
        
        # Color features
        if self.config.compute_color_features and colors is not None:
            pointwise_features.append(colors)  # (N, 3)
        
        # Combine all point-wise features
        if pointwise_features:
            # Debug: print shapes
            for i, arr in enumerate(pointwise_features):
                if arr.shape[0] != points.shape[0]:
                    print(f"Warning: Feature {i} has {arr.shape[0]} points, expected {points.shape[0]}")
                if arr.ndim != 2:
                    print(f"Warning: Feature {i} is not 2D, shape: {arr.shape}")
            # Fix: ensure all are 2D
            pointwise_features = [arr if arr.ndim == 2 else arr[:, None] for arr in pointwise_features]
            features['point_features'] = np.concatenate(pointwise_features, axis=1)
        else:
            features['point_features'] = points  # Fallback to just coordinates
        
        # Global features
        features['global_features'] = self.compute_global_features(pcd)
        
        # Metadata
        features['num_points'] = len(points)
        features['feature_dim'] = features['point_features'].shape[1]
        
        print(f"  ✓ Extracted {features['num_points']} points with {features['feature_dim']} features each")
        
        # Add mesh boundary info if available
        if file_path.endswith('.ply') or file_path.endswith('.obj') or file_path.endswith('.stl'):
            boundary_info = self.extract_mesh_boundary(file_path)
            if boundary_info is not None:
                features['boundary_indices'] = boundary_info['boundary_indices']
                features['boundary_coords'] = boundary_info['boundary_coords']
                features['boundary_normals'] = boundary_info['boundary_normals']
                features['boundary_colors'] = boundary_info['boundary_colors']
        
        return features

def extract_features_batch(file_paths: List[str], 
                          config: Optional[FeatureConfig] = None,
                          max_workers: int = 4) -> Dict[str, Dict]:
    """
    Extract features from multiple fragments in parallel.
    
    Args:
        file_paths: List of .ply file paths
        config: Feature extraction configuration
        max_workers: Number of parallel workers
    
    Returns:
        Dictionary mapping fragment names to feature dictionaries
    """
    extractor = FeatureExtractor(config)
    
    def extract_single(file_path):
        return os.path.basename(file_path), extractor.extract_features(file_path)
    
    print(f"Extracting features from {len(file_paths)} fragments using {max_workers} workers...")
    start_time = time.time()
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for fragment_name, features in executor.map(extract_single, file_paths):
            if features is not None:
                results[fragment_name] = features
            else:
                print(f"Warning: Failed to extract features from {fragment_name}")
    
    end_time = time.time()
    print(f"Feature extraction completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(results)}/{len(file_paths)} fragments")
    
    return results

# Batch feature extraction and saving

def extract_and_save_features_batch(file_paths: List[str], out_dir: str, config: Optional[FeatureConfig] = None, max_workers: int = 4):
    """
    Extract features from multiple fragments and save each as a .npz file in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    extractor = FeatureExtractor(config)
    def process(file_path):
        features = extractor.extract_features(file_path)
        if features is not None:
            out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(file_path))[0] + '.npz')
            # Use allow_pickle=True to save objects like None or arrays
            np.savez_compressed(out_path, **{k: v for k, v in features.items() if isinstance(v, np.ndarray)})  # type: ignore
            print(f"Saved features for {file_path} -> {out_path}")
        else:
            print(f"Failed to extract features for {file_path}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process, file_paths)

# Visualization utility

def plot_mesh_with_boundary(file_path: str, boundary_indices: np.ndarray):
    """
    Plot mesh and highlight boundary vertices.
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required for mesh visualization")
    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertices():
        print(f"Warning: Empty mesh in {file_path}")
        return
    vertices = np.asarray(mesh.vertices)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], s=1, c='gray', alpha=0.3, label='Vertices')
    if boundary_indices is not None and len(boundary_indices) > 0:
        boundary_coords = vertices[boundary_indices]
        ax.scatter(boundary_coords[:,0], boundary_coords[:,1], boundary_coords[:,2], s=10, c='red', label='Boundary')
    ax.set_title(os.path.basename(file_path))
    ax.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Configuration
    config = FeatureConfig(
        target_points=30000,
        normalize_scale=True,
        normalize_position=True,
        compute_normals=True,
        compute_fpfh=True,
        compute_curvature=True,
        compute_global_features=True,
        compute_color_features=True
    )
    
    # Extract features from a single fragment
    extractor = FeatureExtractor(config)
    features = extractor.extract_features("path/to/fragment.ply")
    
    if features:
        print(f"Point features shape: {features['point_features'].shape}")
        print(f"Global features: {list(features['global_features'].keys())}")
    
    # Batch processing
    # file_paths = ["frag1.ply", "frag2.ply", "frag3.ply"]
    # all_features = extract_features_batch(file_paths, config, max_workers=4)