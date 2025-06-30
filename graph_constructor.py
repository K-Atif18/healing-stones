import os
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, TypedDict
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

try:
    import torch_geometric
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_undirected
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not available. Please install with: pip install torch_geometric")

try:
    from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Using basic neighbor search.")

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
class GraphConfig:
    """Configuration for graph construction"""
    # Node features
    use_global_features: bool = True
    global_feature_repeat: bool = True  # Repeat global features for each node
    
    # Intra-fragment edges (within each fragment)
    build_intra_edges: bool = True
    intra_edge_method: str = "knn"  # "knn", "radius", "both"
    intra_k: int = 8  # Number of nearest neighbors
    intra_radius: float = 0.1  # Radius for neighborhood
    
    # Inter-fragment edges (between fragments) 
    build_inter_edges: bool = True
    inter_edge_method: str = "knn"  # "knn", "radius", "fully_connected", "spatial"
    inter_k: int = 5  # K nearest neighbors across fragments
    inter_radius: float = 0.15  # Radius for inter-fragment connections
    
    # Edge attributes
    compute_edge_attributes: bool = True
    edge_attr_features: List[str] = field(default_factory=lambda: ["distance", "feature_diff"])
    
    # Graph properties
    undirected: bool = True  # Convert to undirected graph
    self_loops: bool = False  # Add self loops
    
    # Performance
    max_nodes: int = 60000

class GraphConstructor:
    """
    Constructs graphs from point cloud fragment pairs for GNN training.
    """
    
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for graph construction")
    
    def _build_knn_edges(self, points: np.ndarray, k: int, 
                        node_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Build k-nearest neighbor edges."""
        if not SKLEARN_AVAILABLE:
            return self._build_knn_edges_fallback(points, k, node_offset)
            
        if len(points) <= k:
            k = len(points) - 1
            
        if k <= 0:
            return np.array([[], []]), np.array([])
            
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Remove self-connections (first column)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        
        # Create edge list
        source_nodes = np.repeat(np.arange(len(points)), k)
        target_nodes = indices.flatten()
        
        # Add node offset
        source_nodes += node_offset
        target_nodes += node_offset
        
        # Create edge index
        edge_index = np.vstack([source_nodes, target_nodes])
        edge_distances = distances.flatten()
        
        return edge_index, edge_distances
    
    def _build_knn_edges_fallback(self, points: np.ndarray, k: int, 
                                 node_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback KNN without sklearn."""
        n_points = len(points)
        if n_points <= k:
            k = n_points - 1
            
        if k <= 0:
            return np.array([[], []]), np.array([])
        
        # Compute pairwise distances
        distances_matrix = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        
        # Find k nearest neighbors for each point
        edges = []
        edge_distances = []
        
        for i in range(n_points):
            # Get indices of k+1 nearest neighbors (including self)
            nearest_indices = np.argpartition(distances_matrix[i], k+1)[:k+1]
            # Remove self (distance 0)
            nearest_indices = nearest_indices[nearest_indices != i][:k]
            
            for j in nearest_indices:
                edges.append([i + node_offset, j + node_offset])
                edge_distances.append(distances_matrix[i, j])
        
        if not edges:
            return np.array([[], []]), np.array([])
            
        edge_index = np.array(edges).T
        edge_distances = np.array(edge_distances)
        
        return edge_index, edge_distances
    
    def _build_radius_edges(self, points: np.ndarray, radius: float,
                           node_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Build radius-based edges."""
        if not SKLEARN_AVAILABLE:
            return self._build_radius_edges_fallback(points, radius, node_offset)
            
        # Build radius neighbor graph
        adj_matrix = radius_neighbors_graph(
            points, radius=radius, mode='distance', include_self=False
        )
        
        # Convert to edge list
        rows, cols = adj_matrix.nonzero()
        distances = adj_matrix.data
        
        # Add node offset
        rows = rows + node_offset
        cols = cols + node_offset
        
        # Create edge index
        edge_index = np.vstack([rows, cols])
        
        return edge_index, distances
    
    def _build_radius_edges_fallback(self, points: np.ndarray, radius: float,
                                    node_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback radius edges without sklearn."""
        n_points = len(points)
        edges = []
        edge_distances = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(points[i] - points[j])
                if dist <= radius:
                    edges.append([i + node_offset, j + node_offset])
                    edges.append([j + node_offset, i + node_offset])  # Bidirectional
                    edge_distances.extend([dist, dist])
        
        if not edges:
            return np.array([[], []]), np.array([])
            
        edge_index = np.array(edges).T
        edge_distances = np.array(edge_distances, dtype=np.float32)
        
        return edge_index, edge_distances
    
    def _build_intra_edges(self, points: np.ndarray, node_offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges within a fragment."""
        edges = []
        edge_distances = []
        
        if self.config.intra_edge_method in ["knn", "both"]:
            knn_edges, knn_dists = self._build_knn_edges(points, self.config.intra_k, node_offset)
            if len(knn_edges[0]) > 0:
                edges.append(knn_edges)
                edge_distances.append(knn_dists)
        
        if self.config.intra_edge_method in ["radius", "both"]:
            radius_edges, radius_dists = self._build_radius_edges(points, self.config.intra_radius, node_offset)
            if len(radius_edges[0]) > 0:
                edges.append(radius_edges)
                edge_distances.append(radius_dists)
        
        if not edges:
            return np.array([[], []]), np.array([])
        
        return np.concatenate(edges, axis=1), np.concatenate(edge_distances)
    
    def _build_inter_fragment_edges(self, points_a: np.ndarray, points_b: np.ndarray,
                                   offset_a: int, offset_b: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build edges between two fragments."""
        if self.config.inter_edge_method == "knn":
            # Compute all pairwise distances
            n_a, n_b = len(points_a), len(points_b)
            if n_a == 0 or n_b == 0:
                return np.array([[], []]), np.array([], dtype=np.float32)
            
            # Use broadcasting for memory-efficient distance computation
            distances = np.linalg.norm(points_a[:, None] - points_b[None, :], axis=2)
            
            # Find k nearest neighbors for each point in fragment A
            k = min(self.config.inter_k, n_b)
            if k == 0:
                return np.array([[], []]), np.array([], dtype=np.float32)
            
            nearest_indices = np.argpartition(distances, k, axis=1)[:, :k]
            
            # Create edge list
            source_nodes = np.repeat(np.arange(n_a), k) + offset_a
            target_nodes = nearest_indices.flatten() + offset_b
            edge_distances = np.take_along_axis(distances, nearest_indices, axis=1).flatten()
            
            # Create edge index
            edge_index = np.vstack([source_nodes, target_nodes])
            
            # Add reverse edges
            edge_index_rev = np.vstack([target_nodes, source_nodes])
            
            return np.concatenate([edge_index, edge_index_rev], axis=1), \
                   np.concatenate([edge_distances, edge_distances])
        
        elif self.config.inter_edge_method == "radius":
            try:
                # Process in batches for memory efficiency
                batch_size = min(10000, len(points_a))
                source_nodes = []
                target_nodes = []
                edge_distances = []
                
                # Build radius neighbors index for points_b
                nbrs = NearestNeighbors(radius=self.config.inter_radius, algorithm='auto')
                nbrs.fit(points_b)
                
                # Process points_a in batches
                for i in range(0, len(points_a), batch_size):
                    batch_points = points_a[i:i + batch_size]
                    distances, indices = nbrs.radius_neighbors(batch_points)
                    
                    # Create edge list for this batch
                    for j, (dists, idx) in enumerate(zip(distances, indices)):
                        if len(idx) > 0:  # Only add edges if neighbors found
                            source_nodes.extend([i + j] * len(idx))
                            target_nodes.extend(idx)
                            edge_distances.extend(dists)
                
                if not source_nodes:  # No edges found
                    return np.array([[], []]), np.array([])
                
                # Add offsets
                source_nodes = np.array(source_nodes) + offset_a
                target_nodes = np.array(target_nodes) + offset_b
                edge_distances = np.array(edge_distances)
                
                # Create edge index
                edge_index = np.vstack([source_nodes, target_nodes])
                
                # Add reverse edges
                edge_index_rev = np.vstack([target_nodes, source_nodes])
                
                return np.concatenate([edge_index, edge_index_rev], axis=1), \
                       np.concatenate([edge_distances, edge_distances])
                   
            except Exception as e:
                print(f"Warning: Radius edge construction failed: {e}")
                return np.array([[], []]), np.array([])
            
        elif self.config.inter_edge_method == "fully_connected":
            # Create all possible pairs with memory-efficient broadcasting
            n_a, n_b = len(points_a), len(points_b)
            max_edges = min(n_a * n_b, self.config.max_nodes)  # Limit total edges
            
            if max_edges == 0:
                return np.array([[], []]), np.array([])
            
            # Sample edges if needed
            if n_a * n_b > max_edges:
                # Random sampling of edges
                total_edges = n_a * n_b
                sample_prob = max_edges / total_edges
                mask = np.random.rand(n_a, n_b) < sample_prob
                rows, cols = np.where(mask)
            else:
                # All pairs if under limit
                rows = np.repeat(np.arange(n_a), n_b)
                cols = np.tile(np.arange(n_b), n_a)
            
            # Add offsets
            source_nodes = rows + offset_a
            target_nodes = cols + offset_b
            
            # Compute distances
            edge_distances = np.linalg.norm(points_a[rows] - points_b[cols], axis=1)
            
            # Create edge index
            edge_index = np.vstack([source_nodes, target_nodes])
            
            # Add reverse edges
            edge_index_rev = np.vstack([target_nodes, source_nodes])
            
            return np.concatenate([edge_index, edge_index_rev], axis=1), \
                   np.concatenate([edge_distances, edge_distances])
        
        else:
            raise ValueError(f"Unknown inter-edge method: {self.config.inter_edge_method}")
    
    def _compute_edge_attributes(self, points_a: np.ndarray, points_b: np.ndarray,
                               edge_index: np.ndarray, features_a: Optional[np.ndarray] = None,
                               features_b: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute edge attributes."""
        if not self.config.compute_edge_attributes:
            return np.array([])
            
        edge_attrs = []
        source_nodes = edge_index[0]
        target_nodes = edge_index[1]
        
        # Get source and target points
        source_points = points_a if source_nodes[0] < len(points_a) else points_b
        target_points = points_b if target_nodes[0] >= len(points_a) else points_a
        
        # Adjust indices for second fragment
        source_idx = source_nodes % len(points_a)
        target_idx = target_nodes % len(points_b)
        
        # Distance between points
        if "distance" in self.config.edge_attr_features:
            distances = np.linalg.norm(
                source_points[source_idx] - target_points[target_idx],
                axis=1, keepdims=True
            )
            edge_attrs.append(distances)
        
        # Feature difference
        if "feature_diff" in self.config.edge_attr_features and features_a is not None and features_b is not None:
            source_features = features_a if source_nodes[0] < len(points_a) else features_b
            target_features = features_b if target_nodes[0] >= len(points_a) else features_a
            
            feature_diffs = np.linalg.norm(
                source_features[source_idx] - target_features[target_idx],
                axis=1, keepdims=True
            )
            edge_attrs.append(feature_diffs)
        
        # Angle between points (if using normals)
        if "angle" in self.config.edge_attr_features and features_a is not None and features_b is not None:
            source_normals = features_a[:, :3] if source_nodes[0] < len(points_a) else features_b[:, :3]
            target_normals = features_b[:, :3] if target_nodes[0] >= len(points_a) else features_a[:, :3]
            
            dot_products = np.sum(
                source_normals[source_idx] * target_normals[target_idx],
                axis=1, keepdims=True
            )
            angles = np.arccos(np.clip(dot_products, -1, 1))
            edge_attrs.append(angles)
        
        if not edge_attrs:
            return np.array([])
            
        return np.concatenate(edge_attrs, axis=1)
    
    def build_graph_from_features(self, features_a: Dict[str, Any], features_b: Dict[str, Any],
                                label: int, metadata: Optional[Dict[str, Any]] = None) -> Optional[Data]:
        """Build PyG graph from extracted features."""
        # Extract points and features
        points_a = features_a['points']
        points_b = features_b['points']
        
        if len(points_a) + len(points_b) > self.config.max_nodes:
            print(f"Warning: Total nodes ({len(points_a) + len(points_b)}) exceeds max_nodes ({self.config.max_nodes})")
            return None
        
        # Node features
        x = torch.from_numpy(np.concatenate([
            features_a['point_features'],
            features_b['point_features']
        ], axis=0)).float()
        
        # Build edges
        edge_index = []
        edge_attrs = []
        
        # Intra-fragment edges
        if self.config.build_intra_edges:
            # Fragment A
            intra_edges_a, intra_attrs_a = self._build_intra_edges(points_a)
            edge_index.append(intra_edges_a)
            if self.config.compute_edge_attributes:
                edge_attrs.append(intra_attrs_a)
            
            # Fragment B (with offset)
            intra_edges_b, intra_attrs_b = self._build_intra_edges(
                points_b, node_offset=len(points_a)
            )
            edge_index.append(intra_edges_b)
            if self.config.compute_edge_attributes:
                edge_attrs.append(intra_attrs_b)
        
        # Inter-fragment edges
        if self.config.build_inter_edges:
            inter_edges, inter_attrs = self._build_inter_fragment_edges(
                points_a, points_b, 0, len(points_a)
            )
            edge_index.append(inter_edges)
            if self.config.compute_edge_attributes:
                edge_attrs.append(inter_attrs)
        
        # Combine edges and attributes
        if edge_index:
            edge_index = torch.from_numpy(np.concatenate(edge_index, axis=1)).long()
            if edge_attrs:
                edge_attr = torch.from_numpy(np.concatenate(edge_attrs, axis=0)).float()
            else:
                edge_attr = None
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = None
        
        # Convert to undirected if specified
        if self.config.undirected:
            edge_index = to_undirected(edge_index)
            if edge_attr is not None:
                edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Add self loops if specified
        if self.config.self_loops:
            self_loops = torch.arange(len(x))
            self_loops = torch.stack([self_loops, self_loops], dim=0)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
            if edge_attr is not None:
                self_loop_attrs = torch.zeros((len(x), edge_attr.shape[1]))
                edge_attr = torch.cat([edge_attr, self_loop_attrs], dim=0)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=len(x)
        )
        
        # Add metadata if provided
        if metadata:
            for key, value in metadata.items():
                data[key] = value
        
        return data
    
    def build_graph_from_pair_folder(self, pair_folder_path: str, 
                                    feature_extractor: Any, label: int) -> Optional[Data]:
        """
        Build graph from a pair folder containing two .ply files.
        
        Args:
            pair_folder_path: Path to folder containing exactly 2 .ply files
            feature_extractor: FeatureExtractor instance
            label: Graph label (1 for positive, 0 for negative)
            
        Returns:
            PyTorch Geometric Data object or None if failed
        """
        # Get .ply files
        ply_files = [f for f in os.listdir(pair_folder_path) if f.endswith('.ply')]
        
        if len(ply_files) != 2:
            print(f"Warning: {pair_folder_path} does not contain exactly 2 .ply files")
            return None
        
        # Extract features
        features_a = feature_extractor.extract_features(os.path.join(pair_folder_path, ply_files[0]))
        features_b = feature_extractor.extract_features(os.path.join(pair_folder_path, ply_files[1]))
        
        if features_a is None or features_b is None:
            print(f"Warning: Failed to extract features from {pair_folder_path}")
            return None
        
        # Build graph
        metadata = {
            'pair_folder': os.path.basename(pair_folder_path),
            'fragment_files': ply_files
        }
        
        return self.build_graph_from_features(features_a, features_b, label, metadata)

def build_dataset_graphs(pairs_root: str, 
                        feature_extractor: Any,
                        graph_config: Optional[GraphConfig] = None,
                        max_workers: int = 4,
                        save_path: Optional[str] = None,
                        batch_size: int = 10) -> None:
    """
    Build graphs for entire dataset, saving each graph (or batch) to disk immediately to avoid high RAM usage.
    Args:
        pairs_root: Root directory containing pos_pair_* and neg_pair_* folders
        feature_extractor: FeatureExtractor instance
        graph_config: Graph construction configuration
        max_workers: Number of parallel workers
        save_path: Optional path to save processed graphs (ignored, see below)
        batch_size: Number of graphs to save in a single file (default: 10)
    """
    constructor = GraphConstructor(graph_config)
    
    # Get all pair folders
    pos_folders = [f for f in os.listdir(pairs_root) 
                   if f.startswith('pos_pair_') and os.path.isdir(os.path.join(pairs_root, f))]
    neg_folders = [f for f in os.listdir(pairs_root) 
                   if f.startswith('neg_pair_') and os.path.isdir(os.path.join(pairs_root, f))]
    
    print(f"Found {len(pos_folders)} positive and {len(neg_folders)} negative pairs")
    
    def process_folder(folder_info):
        folder_name, label = folder_info
        folder_path = os.path.join(pairs_root, folder_name)
        try:
            graph = constructor.build_graph_from_pair_folder(folder_path, feature_extractor, label)
            return (folder_name, graph, label, None)
        except Exception as e:
            return (folder_name, None, label, str(e))
    
    # Prepare processing list
    folder_list = [(f, 1) for f in pos_folders] + [(f, 0) for f in neg_folders]
    print(f"Processing {len(folder_list)} pairs with {max_workers} workers...")
    start_time = time.time()
    
    # Prepare output directory
    out_dir = 'dataset_graphs'
    os.makedirs(out_dir, exist_ok=True)
    
    # Process in parallel, but save as we go
    from concurrent.futures import ThreadPoolExecutor
    batch = []
    batch_meta = []
    failed_pairs = []
    pos_count = 0
    neg_count = 0
    total_count = 0
    batch_idx = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for folder_name, graph, label, error in executor.map(process_folder, folder_list):
            total_count += 1
            if graph is not None:
                batch.append(graph)
                batch_meta.append((folder_name, label))
                if label == 1:
                    pos_count += 1
                else:
                    neg_count += 1
            else:
                failed_pairs.append((folder_name, label, error))
            # Save batch if full
            if len(batch) >= batch_size:
                torch.save(batch, os.path.join(out_dir, f'batch_{batch_idx}.pt'))
                with open(os.path.join(out_dir, f'batch_{batch_idx}_meta.txt'), 'w') as f:
                    for meta in batch_meta:
                        f.write(f'{meta[0]},{meta[1]}\n')
                print(f"Saved batch {batch_idx} with {len(batch)} graphs.")
                batch = []
                batch_meta = []
                batch_idx += 1
    # Save any remaining graphs
    if batch:
        torch.save(batch, os.path.join(out_dir, f'batch_{batch_idx}.pt'))
        with open(os.path.join(out_dir, f'batch_{batch_idx}_meta.txt'), 'w') as f:
            for meta in batch_meta:
                f.write(f'{meta[0]},{meta[1]}\n')
        print(f"Saved batch {batch_idx} with {len(batch)} graphs.")
    end_time = time.time()
    print(f"Graph construction completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully created {pos_count + neg_count} graphs: {pos_count} positive, {neg_count} negative")
    if failed_pairs:
        print(f"Failed to process {len(failed_pairs)} pairs:")
        for folder_name, label, error in failed_pairs:
            print(f"  {folder_name} (label={label}): {error}")
    else:
        print("All pairs processed successfully!")

# Example usage and testing
if __name__ == "__main__":
    from feature_extractor import FeatureExtractor, FeatureConfig
    
    # Configure feature extraction
    feature_config = FeatureConfig(
        target_points=15000,  # Reduced to 20k
        compute_fpfh=True,
        compute_curvature=True,
        compute_global_features=True
    )
    
    # Configure graph construction
    graph_config = GraphConfig(
        build_intra_edges=True,
        intra_edge_method="knn",
        intra_k=6,
        build_inter_edges=True,
        inter_edge_method="knn",
        inter_k=3,
        compute_edge_attributes=True,
        edge_attr_features=["distance", "feature_diff"],
        max_nodes=120000  # Increased max_nodes
    )
    
    # Create instances
    feature_extractor = FeatureExtractor(feature_config)
    graph_constructor = GraphConstructor(graph_config)
    
    # Example: Build dataset (now saves in batches)
    build_dataset_graphs(
        "dataset_root/pairs", feature_extractor, graph_config, 
        max_workers=1, batch_size=10)
    
    print("Graph constructor ready!")