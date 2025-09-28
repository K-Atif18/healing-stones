#!/usr/bin/env python3
"""
Phase 2.1: Cluster-Aware Fragment Encoder
Converts a fragment into a 1280-dimensional embedding using break-surface clusters
and their spatial/structural features.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import pickle
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ClusterFeatureBranch(nn.Module):
    """Encodes cluster graph topology and PCA descriptors."""
    
    def __init__(self, 
                 pca_input_dim: int = 9,  # barycenter(3) + size(1) + anisotropy(1) + eigenvalues(3) + scale(1)
                 hidden_dim: int = 128,
                 topological_dim: int = 512,
                 pca_dim: int = 256,
                 positional_dim: int = 128):
        super().__init__()
        
        # GCN/GAT for topological features
        self.conv1 = GATConv(pca_input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GCNConv(hidden_dim * 4, topological_dim)
        
        # MLP for PCA descriptors
        self.pca_encoder = nn.Sequential(
            nn.Linear(pca_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, pca_dim)
        )
        
        # Multiscale hierarchy encoder
        self.scale_encoder = nn.Sequential(
            nn.Linear(1, 32),  # scale value
            nn.ReLU(),
            nn.Linear(32, positional_dim)
        )
        
        self.output_dim = topological_dim + pca_dim + positional_dim
        
    def forward(self, x, edge_index, batch, scales):
        """
        Args:
            x: Node features [num_nodes, pca_input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes
            scales: Scale values for each node [num_nodes, 1]
        """
        # Topological features via GNN
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h_topo = self.conv3(h, edge_index)  # [num_nodes, 512]
        
        # PCA descriptor features
        h_pca = self.pca_encoder(x)  # [num_nodes, 256]
        
        # Multiscale positional context
        h_scale = self.scale_encoder(scales)  # [num_nodes, 128]
        
        # Combine all features
        h_combined = torch.cat([h_topo, h_pca, h_scale], dim=-1)  # [num_nodes, 896]
        
        # Pool to fragment level
        fragment_features = global_mean_pool(h_combined, batch)  # [batch_size, 896]
        
        return fragment_features, h_combined


class GeometricBranch(nn.Module):
    """Encodes point cloud geometry using PointNet++ style architecture."""
    
    def __init__(self,
                 input_dim: int = 3,  # xyz
                 integral_dim: int = 4,  # Vr, VDr, svol, ek,r
                 pointnet_dim: int = 256,
                 integral_mlp_dim: int = 128):
        super().__init__()
        
        # Simplified PointNet++ for cluster points
        self.pointnet = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, pointnet_dim, 1)
        )
        
        # MLP for integral invariants
        self.integral_encoder = nn.Sequential(
            nn.Linear(integral_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, integral_mlp_dim)
        )
        
        self.output_dim = pointnet_dim + integral_mlp_dim
        
    def forward(self, points, integral_features):
        """
        Args:
            points: Cluster point clouds [batch_size, num_points, 3]
            integral_features: Integral invariants [batch_size, integral_dim]
        """
        # PointNet++ encoding
        points_t = points.transpose(1, 2)  # [batch, 3, num_points]
        point_features = self.pointnet(points_t)  # [batch, 256, num_points]
        
        # Global max pooling
        point_features = torch.max(point_features, dim=2)[0]  # [batch, 256]
        
        # Integral invariant encoding
        integral_features = self.integral_encoder(integral_features)  # [batch, 128]
        
        # Combine
        geometric_features = torch.cat([point_features, integral_features], dim=-1)  # [batch, 384]
        
        return geometric_features


class AttentionFusion(nn.Module):
    """Attention-based fusion of cluster and geometric features."""
    
    def __init__(self, 
                 cluster_dim: int = 896,
                 geometric_dim: int = 384,
                 hidden_dim: int = 256,
                 output_dim: int = 1280):
        super().__init__()
        
        self.cluster_dim = cluster_dim
        self.geometric_dim = geometric_dim
        
        # Attention computation
        self.cluster_query = nn.Linear(cluster_dim, hidden_dim)
        self.geometric_key = nn.Linear(geometric_dim, hidden_dim)
        self.geometric_value = nn.Linear(geometric_dim, hidden_dim)
        
        # Feature projection
        self.cluster_proj = nn.Linear(cluster_dim, output_dim // 2)
        self.geometric_proj = nn.Linear(hidden_dim, output_dim // 2)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, cluster_features, geometric_features, cluster_weights=None):
        """
        Args:
            cluster_features: [batch_size, num_clusters, cluster_dim]
            geometric_features: [batch_size, num_clusters, geometric_dim]
            cluster_weights: Optional attention weights based on GT priors
        """
        batch_size = cluster_features.shape[0]
        
        # Compute attention
        Q = self.cluster_query(cluster_features)  # [batch, num_clusters, hidden]
        K = self.geometric_key(geometric_features)  # [batch, num_clusters, hidden]
        V = self.geometric_value(geometric_features)  # [batch, num_clusters, hidden]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.shape[-1])
        
        # Apply cluster weights if provided (e.g., based on GT match priors)
        if cluster_weights is not None:
            attention_scores = attention_scores + torch.log(cluster_weights + 1e-8)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to geometric features
        attended_geometric = torch.matmul(attention_weights, V)  # [batch, num_clusters, hidden]
        
        # Pool to fragment level
        cluster_pooled = torch.mean(cluster_features, dim=1)  # [batch, cluster_dim]
        geometric_pooled = torch.mean(attended_geometric, dim=1)  # [batch, hidden]
        
        # Project features
        cluster_proj = self.cluster_proj(cluster_pooled)  # [batch, 640]
        geometric_proj = self.geometric_proj(geometric_pooled)  # [batch, 640]
        
        # Concatenate and fuse
        combined = torch.cat([cluster_proj, geometric_proj], dim=-1)  # [batch, 1280]
        output = self.fusion(combined)  # [batch, 1280]
        
        return output, attention_weights


class ClusterAwareFragmentEncoder(nn.Module):
    """Main encoder that combines all branches to produce 1280-D fragment embedding."""
    
    def __init__(self,
                 pca_input_dim: int = 9,
                 integral_dim: int = 4,
                 output_dim: int = 1280):
        super().__init__()
        
        # Feature branches
        self.cluster_branch = ClusterFeatureBranch(pca_input_dim=pca_input_dim)
        self.geometric_branch = GeometricBranch(integral_dim=integral_dim)
        
        # Attention fusion
        self.fusion = AttentionFusion(
            cluster_dim=self.cluster_branch.output_dim,
            geometric_dim=self.geometric_branch.output_dim,
            output_dim=output_dim
        )
        
        self.output_dim = output_dim
        
    def forward(self, cluster_graph, point_clouds, integral_features, cluster_weights=None):
        """
        Args:
            cluster_graph: PyG Data object with node features and edges
            point_clouds: List of point clouds for each cluster [num_clusters, ~1000, 3]
            integral_features: Integral invariants for each cluster [num_clusters, 4]
            cluster_weights: Optional weights based on GT match priors [num_clusters]
        
        Returns:
            fragment_embedding: 1280-dimensional fragment descriptor
            attention_weights: Attention weights for interpretability
        """
        # Extract cluster features
        cluster_features, node_features = self.cluster_branch(
            cluster_graph.x,
            cluster_graph.edge_index,
            cluster_graph.batch,
            cluster_graph.scales
        )
        
        # Process each cluster's point cloud
        geometric_features = []
        for i, (points, integral) in enumerate(zip(point_clouds, integral_features)):
            geom_feat = self.geometric_branch(
                points.unsqueeze(0),  # Add batch dimension
                integral.unsqueeze(0)
            )
            geometric_features.append(geom_feat)
        
        geometric_features = torch.cat(geometric_features, dim=0)  # [num_clusters, 384]
        
        # Reshape for fusion (add batch dimension)
        cluster_features_expanded = node_features.unsqueeze(0)  # [1, num_clusters, 896]
        geometric_features_expanded = geometric_features.unsqueeze(0)  # [1, num_clusters, 384]
        
        if cluster_weights is not None:
            cluster_weights = cluster_weights.unsqueeze(0)  # [1, num_clusters]
        
        # Fuse features with attention
        fragment_embedding, attention = self.fusion(
            cluster_features_expanded,
            geometric_features_expanded,
            cluster_weights
        )
        
        return fragment_embedding.squeeze(0), attention.squeeze(0)


class FragmentEncoderDataLoader:
    """Loads and prepares data for the fragment encoder."""
    
    def __init__(self,
                 clusters_file: str = "output/feature_clusters_fixed.pkl",
                 segments_file: str = "output/segmented_fragments_with_indices.pkl",
                 assembly_file: str = "output/cluster_assembly_with_gt.h5",
                 ply_dir: str = "Ground_Truth/artifact_1"):
        
        self.ply_dir = Path(ply_dir)
        
        # Load data
        with open(clusters_file, 'rb') as f:
            self.cluster_data = pickle.load(f)
        
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Load assembly knowledge for GT priors
        self.gt_priors = self._load_gt_priors(assembly_file)
        
    def _load_gt_priors(self, assembly_file):
        """Load ground truth priors for cluster importance."""
        gt_priors = {}
        
        if Path(assembly_file).exists():
            with h5py.File(assembly_file, 'r') as f:
                if 'cluster_matches' in f:
                    matches = f['cluster_matches']
                    fragment_1 = [f.decode('utf8') for f in matches['fragment_1'][:]]
                    fragment_2 = [f.decode('utf8') for f in matches['fragment_2'][:]]
                    cluster_id_1 = matches['cluster_id_1'][:]
                    cluster_id_2 = matches['cluster_id_2'][:]
                    is_gt = matches['is_ground_truth'][:]
                    confidences = matches['confidences'][:]
                    
                    # Count GT matches per cluster
                    for i in range(len(fragment_1)):
                        if is_gt[i]:
                            key1 = (fragment_1[i], cluster_id_1[i])
                            key2 = (fragment_2[i], cluster_id_2[i])
                            
                            gt_priors[key1] = gt_priors.get(key1, 0) + confidences[i]
                            gt_priors[key2] = gt_priors.get(key2, 0) + confidences[i]
        
        return gt_priors
    
    def prepare_fragment_data(self, fragment_name: str):
        """Prepare all data needed to encode a fragment."""
        logger.info(f"Preparing data for {fragment_name}")
        
        # Get clusters for this fragment
        fragment_clusters = []
        cluster_idx = 0
        
        # Find clusters belonging to this fragment
        for frag in sorted(self.segment_data.keys()):
            n_clusters = self.segment_data[frag].get('n_clusters', 0)
            
            if frag == fragment_name:
                fragment_clusters = self.cluster_data['clusters'][cluster_idx:cluster_idx + n_clusters]
                break
            
            cluster_idx += n_clusters
        
        if not fragment_clusters:
            raise ValueError(f"No clusters found for fragment {fragment_name}")
        
        # Build cluster graph
        cluster_graph = self._build_cluster_graph(fragment_clusters, fragment_name)
        
        # Extract point clouds and integral features
        point_clouds, integral_features = self._extract_geometric_features(
            fragment_name, fragment_clusters
        )
        
        # Get cluster weights based on GT priors
        cluster_weights = self._get_cluster_weights(fragment_name, fragment_clusters)
        
        return cluster_graph, point_clouds, integral_features, cluster_weights
    
    def _build_cluster_graph(self, clusters, fragment_name):
        """Build PyTorch Geometric graph from clusters."""
        # Node features: barycenter(3) + size(1) + anisotropy(1) + eigenvalues(3) + scale(1)
        node_features = []
        scales = []
        
        for cluster in clusters:
            features = np.concatenate([
                cluster['barycenter'],
                [cluster['size_signature']],
                [cluster['anisotropy_signature']],
                cluster['eigenvalues'],
                [cluster['scale']]
            ])
            node_features.append(features)
            scales.append([cluster['scale']])
        
        # Convert list to numpy array first to avoid warning
        node_features = np.array(node_features)
        scales = np.array(scales)
        
        node_features = torch.FloatTensor(node_features)
        scales = torch.FloatTensor(scales)
        
        # Build edges from overlap graph
        edges = []
        cluster_ids = [c['cluster_id'] for c in clusters]
        cluster_id_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
        
        for edge in self.cluster_data.get('overlap_graph_edges', []):
            if edge[0] in cluster_id_to_idx and edge[1] in cluster_id_to_idx:
                edges.append([cluster_id_to_idx[edge[0]], cluster_id_to_idx[edge[1]]])
                edges.append([cluster_id_to_idx[edge[1]], cluster_id_to_idx[edge[0]]])  # Undirected
        
        if not edges:
            # Create fully connected graph if no overlap edges
            n = len(clusters)
            edges = [[i, j] for i in range(n) for j in range(n) if i != j]
        
        edge_index = torch.LongTensor(edges).t()
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            scales=scales,
            batch=torch.zeros(len(clusters), dtype=torch.long)  # Single graph
        )
        
        return data
    
    def _extract_geometric_features(self, fragment_name, clusters):
        """Extract point clouds and compute integral invariants."""
        # Load fragment point cloud
        ply_file = self.ply_dir / f"{fragment_name}.ply"
        if not ply_file.exists():
            raise FileNotFoundError(f"PLY file not found: {ply_file}")
        
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_file))
        all_points = np.asarray(pcd.points)
        
        # Get segment indices
        segment_indices = self.segment_data[fragment_name].get('segment_indices', [])
        
        point_clouds = []
        integral_features = []
        
        for i, cluster in enumerate(clusters):
            # Get points for this cluster
            cluster_mask = (segment_indices == i)
            cluster_points = all_points[cluster_mask]
            
            if len(cluster_points) < 100:
                # Use points near cluster center if too few assigned points
                from scipy.spatial import cKDTree
                tree = cKDTree(all_points)
                indices = tree.query_ball_point(cluster['barycenter'], cluster['scale'] * 1.5)
                cluster_points = all_points[indices]
            
            # Sample ~1000 points
            if len(cluster_points) > 1000:
                indices = np.random.choice(len(cluster_points), 1000, replace=False)
                cluster_points = cluster_points[indices]
            
            # Center points
            cluster_points = cluster_points - cluster['barycenter']
            
            # Compute simple integral invariants (placeholder - implement full computation)
            integral = self._compute_integral_invariants(cluster_points, cluster['scale'])
            
            point_clouds.append(torch.FloatTensor(cluster_points))
            integral_features.append(torch.FloatTensor(integral))
        
        return point_clouds, integral_features
    
    def _compute_integral_invariants(self, points, scale):
        """Compute integral invariants for a point cloud."""
        # Placeholder implementation - replace with actual computation
        # Should compute: Vr(p), VDr(p), svol(p), ek,r(p)
        
        # Simple features for now
        vr = np.mean(np.linalg.norm(points, axis=1))  # Average distance from center
        vdr = np.std(np.linalg.norm(points, axis=1))  # Variation
        svol = len(points) / (scale ** 3)  # Point density
        ekr = np.mean(np.abs(points[:, 2])) / scale  # Vertical variation
        
        return [vr, vdr, svol, ekr]
    
    def _get_cluster_weights(self, fragment_name, clusters):
        """Get importance weights for clusters based on GT priors."""
        weights = []
        
        for i, cluster in enumerate(clusters):
            key = (fragment_name, i)  # Use local cluster ID
            weight = self.gt_priors.get(key, 0.1)  # Base weight of 0.1
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        
        return torch.FloatTensor(weights)


def test_encoder():
    """Test the fragment encoder on sample data."""
    logger.info("Testing Cluster-Aware Fragment Encoder...")
    
    # Initialize data loader
    data_loader = FragmentEncoderDataLoader()
    
    # Initialize encoder
    encoder = ClusterAwareFragmentEncoder()
    encoder.eval()
    
    # Test on first fragment
    fragment_names = sorted(data_loader.segment_data.keys())
    test_fragment = fragment_names[0]
    
    logger.info(f"Testing on fragment: {test_fragment}")
    
    # Prepare data
    cluster_graph, point_clouds, integral_features, cluster_weights = \
        data_loader.prepare_fragment_data(test_fragment)
    
    # Forward pass
    with torch.no_grad():
        embedding, attention = encoder(
            cluster_graph, 
            point_clouds, 
            integral_features, 
            cluster_weights
        )
    
    logger.info(f"Fragment embedding shape: {embedding.shape}")
    logger.info(f"Embedding L2 norm: {torch.norm(embedding).item():.3f}")
    logger.info(f"Attention weights shape: {attention.shape}")
    
    # Save test results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    torch.save({
        'embedding': embedding,
        'attention': attention,
        'fragment': test_fragment
    }, output_dir / "test_fragment_embedding.pt")
    
    logger.info("Test complete! Saved results to output/test_fragment_embedding.pt")


def encode_all_fragments(output_file: str = "output/fragment_embeddings.h5"):
    """Encode all fragments and save embeddings for Phase 2.2."""
    logger.info("Encoding all fragments...")
    
    # Initialize data loader and encoder
    data_loader = FragmentEncoderDataLoader()
    encoder = ClusterAwareFragmentEncoder()
    encoder.eval()
    
    fragment_names = sorted(data_loader.segment_data.keys())
    embeddings_dict = {}
    attention_dict = {}
    
    # Process each fragment
    for fragment_name in tqdm(fragment_names, desc="Encoding fragments"):
        try:
            # Prepare data
            cluster_graph, point_clouds, integral_features, cluster_weights = \
                data_loader.prepare_fragment_data(fragment_name)
            
            # Encode
            with torch.no_grad():
                embedding, attention = encoder(
                    cluster_graph,
                    point_clouds,
                    integral_features,
                    cluster_weights
                )
            
            embeddings_dict[fragment_name] = embedding.numpy()
            attention_dict[fragment_name] = attention.numpy()
            
        except Exception as e:
            logger.error(f"Failed to encode {fragment_name}: {e}")
            continue
    
    # Save all embeddings
    with h5py.File(output_file, 'w') as f:
        # Save embeddings
        embeddings_group = f.create_group('embeddings')
        for frag_name, embedding in embeddings_dict.items():
            embeddings_group.create_dataset(frag_name, data=embedding)
        
        # Save metadata
        metadata = f.create_group('metadata')
        metadata.attrs['embedding_dim'] = encoder.output_dim
        metadata.attrs['num_fragments'] = len(embeddings_dict)
        metadata.attrs['fragments'] = [f.encode('utf8') for f in fragment_names]
        
        # Save attention weights (optional, can be large)
        attention_group = f.create_group('attention_weights')
        for frag_name, attention in attention_dict.items():
            attention_group.create_dataset(frag_name, data=attention, compression='gzip')
    
    logger.info(f"Saved {len(embeddings_dict)} fragment embeddings to {output_file}")
    
    # Print summary statistics
    logger.info("\nEmbedding Statistics:")
    all_embeddings = np.array(list(embeddings_dict.values()))
    logger.info(f"  Shape: {all_embeddings.shape}")
    logger.info(f"  Mean L2 norm: {np.mean(np.linalg.norm(all_embeddings, axis=1)):.3f}")
    logger.info(f"  Std L2 norm: {np.std(np.linalg.norm(all_embeddings, axis=1)):.3f}")
    
    return embeddings_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cluster-Aware Fragment Encoder")
    parser.add_argument("--test", action="store_true", help="Run test on sample data")
    parser.add_argument("--encode-all", action="store_true", help="Encode all fragments")
    parser.add_argument("--output", default="output_2/fragment_embeddings.h5", 
                       help="Output file for embeddings")
    parser.add_argument("--clusters", default="output/feature_clusters_fixed.pkl")
    parser.add_argument("--segments", default="output/segmented_fragments_with_indices.pkl")
    parser.add_argument("--assembly", default="output/cluster_assembly_with_gt.h5")
    parser.add_argument("--ply_dir", default="Ground_Truth/artifact_1")
    
    args = parser.parse_args()
    
    if args.test:
        test_encoder()
    elif args.encode_all:
        encode_all_fragments(args.output)
    else:
        logger.info("Run with --test flag to test the encoder")
        logger.info("Run with --encode-all flag to encode all fragments")