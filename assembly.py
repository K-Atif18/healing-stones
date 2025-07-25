#!/usr/bin/env python3
"""
Unified Multi-Scale Assembly Extractor
Aligned with unified clustering pipeline and multi-scale ground truth
"""

import numpy as np
import pickle
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree
import logging
from tqdm import tqdm
import networkx as nx
import open3d as o3d

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Copy for pickle compatibility
@dataclass
class SurfaceCluster:
    """Cluster representation - copy for pickle compatibility."""
    cluster_id: str
    fragment_name: str
    point_count_scale: int
    scale_name: str
    local_id: int
    barycenter: np.ndarray
    principal_axes: np.ndarray
    eigenvalues: np.ndarray
    size_signature: float
    anisotropy_signature: float
    point_count: int
    point_indices: List[int]
    original_point_indices: List[int]
    neighbors: List[str] = None

@dataclass
class MultiScaleClusterMatch:
    """Represents a potential match between clusters at a specific scale."""
    scale: str  # "1k", "5k", "10k"
    fragment_1: str
    fragment_2: str
    cluster_id_1: str  # Full cluster ID like "frag_1_c1k_001"
    cluster_id_2: str  # Full cluster ID like "frag_2_c1k_003"
    local_id_1: int
    local_id_2: int
    
    # Similarity metrics
    normal_similarity: float
    size_similarity: float
    shape_similarity: float
    spatial_proximity: float
    
    # Combined confidence
    match_confidence: float
    
    # Ground truth information (for labeling only)
    is_ground_truth: bool = False
    gt_confidence: float = 0.0
    gt_overlap_ratio_1: float = 0.0
    gt_overlap_ratio_2: float = 0.0
    gt_contact_type: str = ""

class UnifiedAssemblyExtractor:
    """Extract assembly knowledge using unified clustering pipeline."""
    
    def __init__(self, 
                 clusters_file: str = "output/feature_clusters.pkl",
                 segments_file: str = "output/segmented_fragments.pkl", 
                 ground_truth_file: str = "Ground_Truth/multi_scale_cluster_ground_truth.json",
                 ply_dir: str = "Ground_Truth/reconstructed/artifact_1",
                 output_dir: str = "output",
                 scales: List[str] = ["1k", "5k", "10k"],
                 contact_threshold: float = 2.0):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ply_dir = Path(ply_dir)
        self.scales = scales
        self.contact_threshold = contact_threshold
        
        # Load all data
        self._load_unified_data(clusters_file, segments_file, ground_truth_file)
        
        # Build ground truth lookup for labeling
        self._build_gt_lookup()
        
        # Load fragment point clouds and find contact pairs
        self._load_fragments_and_find_contacts()
        
    def _load_unified_data(self, clusters_file, segments_file, ground_truth_file):
        """Load data from unified pipeline."""
        
        # Load hierarchical clusters
        try:
            with open(clusters_file, 'rb') as f:
                self.hierarchical_clusters = pickle.load(f)
        except (AttributeError, ImportError) as e:
            # Try flat format
            flat_file = Path(clusters_file).parent / "feature_clusters_flat.pkl"
            if flat_file.exists():
                self._load_flat_clusters(flat_file)
            else:
                raise FileNotFoundError("No compatible cluster file found!")
        
        # Load segmentation data
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Load multi-scale ground truth
        if Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
        else:
            self.ground_truth = {'cluster_ground_truth_matches_by_scale': {}}
    
    def _load_flat_clusters(self, flat_file: Path):
        """Convert flat cluster data to hierarchical."""
        with open(flat_file, 'rb') as f:
            flat_data = pickle.load(f)
        
        clusters = flat_data.get('clusters', flat_data) if isinstance(flat_data, dict) else flat_data
        
        self.hierarchical_clusters = {}
        for cluster_data in clusters:
            if hasattr(cluster_data, 'fragment_name'):
                fragment = cluster_data.fragment_name
                scale = cluster_data.scale_name
                cluster_dict = {
                    'cluster_id': cluster_data.cluster_id,
                    'barycenter': cluster_data.barycenter,
                    'principal_axes': cluster_data.principal_axes,
                    'eigenvalues': cluster_data.eigenvalues,
                    'size_signature': cluster_data.size_signature,
                    'anisotropy_signature': cluster_data.anisotropy_signature,
                    'point_count': cluster_data.point_count,
                    'local_id': cluster_data.local_id
                }
            else:
                fragment = cluster_data.get('fragment', 'unknown')
                scale = cluster_data.get('scale', '1k')
                cluster_dict = cluster_data
            
            if fragment not in self.hierarchical_clusters:
                self.hierarchical_clusters[fragment] = {}
            if scale not in self.hierarchical_clusters[fragment]:
                self.hierarchical_clusters[fragment][scale] = []
            
            self.hierarchical_clusters[fragment][scale].append(cluster_dict)
        
        logger.info("Converted flat clusters to hierarchical structure")
    
    def _build_gt_lookup(self):
        """Build lookup table for ground truth labeling only."""
        
        self.gt_matches_by_scale = {}
        
        # Get GT matches by scale
        gt_matches_by_scale = self.ground_truth.get('cluster_ground_truth_matches_by_scale', {})
        
        for scale in self.scales:
            self.gt_matches_by_scale[scale] = {}
            
            scale_matches = gt_matches_by_scale.get(scale, [])
            for match in scale_matches:
                # Create lookup key
                frag1 = match['fragment_1']
                frag2 = match['fragment_2']
                local_id1 = match['local_id_1']
                local_id2 = match['local_id_2']
                
                # Bidirectional lookup
                key1 = (frag1, local_id1, frag2, local_id2)
                key2 = (frag2, local_id2, frag1, local_id1)
                
                self.gt_matches_by_scale[scale][key1] = match
                self.gt_matches_by_scale[scale][key2] = match
    
    def _load_fragments_and_find_contacts(self):
        """Load fragment point clouds and find contact pairs using same method as GT script."""
        
        self.fragments = {}
        
        # Load all PLY files
        ply_files = sorted(self.ply_dir.glob("*.ply"))
        
        for ply_file in ply_files:
            fragment_name = ply_file.stem
            
            # Skip if not in our cluster data
            if fragment_name not in self.hierarchical_clusters:
                continue
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_file))
            if not pcd.has_points():
                continue
            
            points = np.asarray(pcd.points)
            
            self.fragments[fragment_name] = {
                'points': points,
                'point_cloud': pcd,
                'n_points': len(points)
            }
        
        # Find contact pairs using same method as GT script
        self.contact_pairs = self._find_fragment_contacts()
    
    def _find_fragment_contacts(self):
        """Find which fragments are in contact using same method as GT script."""
        
        fragment_names = sorted(self.fragments.keys())
        contact_pairs = []
        
        # Check all pairs
        for i in range(len(fragment_names)):
            for j in range(i + 1, len(fragment_names)):
                frag1 = fragment_names[i]
                frag2 = fragment_names[j]
                
                if self._check_fragment_contact(frag1, frag2):
                    contact_pairs.append((frag1, frag2))
        
        return contact_pairs
    
    def _check_fragment_contact(self, frag1: str, frag2: str) -> bool:
        """Quick check if two fragments are in contact using same method as GT script."""
        if frag1 not in self.fragments or frag2 not in self.fragments:
            return False
            
        points1 = self.fragments[frag1]['points']
        points2 = self.fragments[frag2]['points']
        
        # Quick bounding box check
        min1, max1 = np.min(points1, axis=0), np.max(points1, axis=0)
        min2, max2 = np.min(points2, axis=0), np.max(points2, axis=0)
        
        for dim in range(3):
            if min1[dim] - self.contact_threshold > max2[dim]:
                return False
            if min2[dim] - self.contact_threshold > max1[dim]:
                return False
        
        # Sample check for efficiency
        sample_size = min(1000, len(points1))
        sample_indices = np.random.choice(len(points1), sample_size, replace=False)
        sample_points = points1[sample_indices]
        
        tree2 = cKDTree(points2)
        distances, _ = tree2.query(sample_points)
        
        return np.min(distances) < self.contact_threshold
    
    def extract_multi_scale_assembly_knowledge(self):
        """Extract assembly knowledge across all scales."""
        
        print(f"Processing {len(self.contact_pairs)} contact pairs:")
        for i, (frag1, frag2) in enumerate(self.contact_pairs):
            print(f"  {i+1}. {frag1} ↔ {frag2}")
        
        all_matches_by_scale = {}
        
        # Process each scale separately
        for scale in self.scales:
            scale_matches = self._extract_scale_matches(scale)
            all_matches_by_scale[scale] = scale_matches
        
        # Print precision analysis
        self._print_precision_analysis(all_matches_by_scale)
        
        # Build multi-scale assembly graph
        assembly_graph = self._build_multi_scale_graph(all_matches_by_scale)
        
        # Extract multi-scale topology features
        topology_features = self._extract_topology_features(all_matches_by_scale)
        
        # Save all results
        self._save_multi_scale_results(all_matches_by_scale, assembly_graph, topology_features)
        
        # Generate comprehensive report
        self._generate_comprehensive_report(all_matches_by_scale)
        
        return all_matches_by_scale, assembly_graph
    
    def _extract_scale_matches(self, scale: str) -> List[MultiScaleClusterMatch]:
        """Extract matches for a specific scale across contact pairs only."""
        scale_matches = []
        
        # Get all fragments that have clusters at this scale
        fragments_with_scale = [
            frag for frag, scales_dict in self.hierarchical_clusters.items()
            if scale in scales_dict and len(scales_dict[scale]) > 0
        ]
        
        # Only compare contact pairs instead of all possible pairs
        for frag1, frag2 in self.contact_pairs:
            # Skip if either fragment doesn't have clusters at this scale
            if frag1 not in fragments_with_scale or frag2 not in fragments_with_scale:
                continue
                
            pair_matches = self._compare_fragment_pair_at_scale(frag1, frag2, scale)
            scale_matches.extend(pair_matches)
        
        return scale_matches
    
    def _compare_fragment_pair_at_scale(self, frag1: str, frag2: str, scale: str) -> List[MultiScaleClusterMatch]:
        """Compare all clusters between two fragments at a specific scale."""
        
        clusters1 = self.hierarchical_clusters[frag1].get(scale, [])
        clusters2 = self.hierarchical_clusters[frag2].get(scale, [])
        
        if not clusters1 or not clusters2:
            return []
        
        pair_matches = []
        
        # Compare EVERY cluster from frag1 with EVERY cluster from frag2
        # BUT only in one direction to avoid duplicates
        for i, cluster1 in enumerate(clusters1):
            for j, cluster2 in enumerate(clusters2):
                
                # Calculate similarity metrics (NO distance threshold)
                match = self._evaluate_cluster_similarity(
                    cluster1, cluster2, frag1, frag2, scale, i, j
                )
                
                # Label with ground truth if available
                self._label_with_ground_truth(match, scale)
                
                # Store ALL matches (no filtering by distance/confidence)
                pair_matches.append(match)
        
        return pair_matches
    
    def _get_cluster_attribute(self, cluster, attr_name, default_value):
        """Helper method to get attribute from either dict or dataclass."""
        if hasattr(cluster, attr_name):
            # It's a dataclass
            return getattr(cluster, attr_name)
        elif isinstance(cluster, dict) and attr_name in cluster:
            # It's a dictionary
            return cluster[attr_name]
        else:
            # Return default
            return default_value
    
    def _evaluate_cluster_similarity(self, cluster1, cluster2, 
                                   frag1: str, frag2: str, scale: str, 
                                   local_id1: int, local_id2: int) -> MultiScaleClusterMatch:
        """Calculate similarity between two clusters WITHOUT distance threshold."""
        
        # Extract cluster properties using helper method
        barycenter1 = np.array(self._get_cluster_attribute(cluster1, 'barycenter', [0, 0, 0]))
        barycenter2 = np.array(self._get_cluster_attribute(cluster2, 'barycenter', [0, 0, 0]))
        
        axes1 = np.array(self._get_cluster_attribute(cluster1, 'principal_axes', np.eye(3)))
        axes2 = np.array(self._get_cluster_attribute(cluster2, 'principal_axes', np.eye(3)))
        
        eigenvals1 = np.array(self._get_cluster_attribute(cluster1, 'eigenvalues', [1, 1, 1]))
        eigenvals2 = np.array(self._get_cluster_attribute(cluster2, 'eigenvalues', [1, 1, 1]))
        
        size_sig1 = self._get_cluster_attribute(cluster1, 'size_signature', 1.0)
        size_sig2 = self._get_cluster_attribute(cluster2, 'size_signature', 1.0)
        
        aniso_sig1 = self._get_cluster_attribute(cluster1, 'anisotropy_signature', 1.0)
        aniso_sig2 = self._get_cluster_attribute(cluster2, 'anisotropy_signature', 1.0)
        
        # 1. Normal Similarity (most important for assembly)
        normal1 = axes1[:, 0]  # First principal axis
        normal2 = axes2[:, 0]
        
        # For assembly, surfaces should have opposing normals
        normal_dot = np.abs(np.dot(normal1, -normal2))
        normal_similarity = normal_dot  # 0 to 1, higher is better
        
        # 2. Size Similarity
        size_ratio = min(size_sig1, size_sig2) / max(size_sig1, size_sig2)
        size_similarity = size_ratio  # 0 to 1, higher is better
        
        # 3. Shape Similarity (anisotropy)
        aniso_ratio = min(aniso_sig1, aniso_sig2) / max(aniso_sig1, aniso_sig2)
        shape_similarity = aniso_ratio  # 0 to 1, higher is better
        
        # 4. Spatial Proximity (for reference, but no threshold)
        distance = np.linalg.norm(barycenter1 - barycenter2)
        # Convert distance to a similarity score (closer = higher similarity)
        max_reasonable_distance = 100.0  # 100mm - adjust based on your data
        spatial_proximity = np.exp(-distance / max_reasonable_distance)  # 0 to 1
        
        # 5. Combined Confidence (weighted combination)
        # Emphasize normal similarity for assembly
        weights = {
            'normal': 0.5,      # Most important for assembly
            'size': 0.2,        # Somewhat important
            'shape': 0.2,       # Somewhat important  
            'spatial': 0.1      # Least important (no threshold)
        }
        
        match_confidence = (
            weights['normal'] * normal_similarity +
            weights['size'] * size_similarity +
            weights['shape'] * shape_similarity +
            weights['spatial'] * spatial_proximity
        )
        
        # Generate cluster IDs
        cluster_id1 = self._get_cluster_attribute(cluster1, 'cluster_id', f"{frag1}_c{scale}_{local_id1+1:03d}")
        cluster_id2 = self._get_cluster_attribute(cluster2, 'cluster_id', f"{frag2}_c{scale}_{local_id2+1:03d}")
        
        return MultiScaleClusterMatch(
            scale=scale,
            fragment_1=frag1,
            fragment_2=frag2,
            cluster_id_1=cluster_id1,
            cluster_id_2=cluster_id2,
            local_id_1=local_id1,
            local_id_2=local_id2,
            normal_similarity=float(normal_similarity),
            size_similarity=float(size_similarity),
            shape_similarity=float(shape_similarity),
            spatial_proximity=float(spatial_proximity),
            match_confidence=float(match_confidence)
        )
    
    def _label_with_ground_truth(self, match: MultiScaleClusterMatch, scale: str):
        """Label match with ground truth information (if available)."""
        
        # Check if this match exists in ground truth
        key = (match.fragment_1, match.local_id_1, match.fragment_2, match.local_id_2)
        
        gt_lookup = self.gt_matches_by_scale.get(scale, {})
        
        if key in gt_lookup:
            gt_match = gt_lookup[key]
            
            # Label as ground truth
            match.is_ground_truth = True
            match.gt_confidence = gt_match.get('confidence', 0.0)
            match.gt_overlap_ratio_1 = gt_match.get('overlap_ratio_1', 0.0)
            match.gt_overlap_ratio_2 = gt_match.get('overlap_ratio_2', 0.0)
            match.gt_contact_type = gt_match.get('contact_type', 'unknown')
            
            # Optionally boost confidence for GT matches (but not required)
            # match.match_confidence = max(match.match_confidence, 0.7)
    
    def _find_missing_gt_matches(self, frag1: str, frag2: str, scale: str, 
                               existing_matches: List[MultiScaleClusterMatch]) -> List[MultiScaleClusterMatch]:
        """Find GT matches that weren't found in similarity comparison."""
        
        missing_matches = []
        gt_lookup = self.gt_matches_by_scale.get(scale, {})
        
        # Get existing match keys
        existing_keys = set()
        for match in existing_matches:
            key = (match.fragment_1, match.local_id_1, match.fragment_2, match.local_id_2)
            existing_keys.add(key)
        
        # Check all GT matches for this pair
        for key, gt_match in gt_lookup.items():
            gt_frag1, gt_local1, gt_frag2, gt_local2 = key
            
            # Check if this GT match involves our fragment pair
            if ((gt_frag1 == frag1 and gt_frag2 == frag2) or 
                (gt_frag1 == frag2 and gt_frag2 == frag1)):
                
                if key not in existing_keys:
                    # Create a match for this missing GT
                    logger.warning(f"Found GT match not in similarity results: {key}")
                    
                    # Create minimal match with GT labeling
                    missing_match = MultiScaleClusterMatch(
                        scale=scale,
                        fragment_1=gt_frag1,
                        fragment_2=gt_frag2,
                        cluster_id_1=gt_match.get('cluster_id_1', ''),
                        cluster_id_2=gt_match.get('cluster_id_2', ''),
                        local_id_1=gt_local1,
                        local_id_2=gt_local2,
                        normal_similarity=0.0,  # Unknown
                        size_similarity=0.0,    # Unknown
                        shape_similarity=0.0,   # Unknown
                        spatial_proximity=0.0,  # Unknown
                        match_confidence=0.0,   # Low since not found by similarity
                        is_ground_truth=True,
                        gt_confidence=gt_match.get('confidence', 0.0),
                        gt_overlap_ratio_1=gt_match.get('overlap_ratio_1', 0.0),
                        gt_overlap_ratio_2=gt_match.get('overlap_ratio_2', 0.0),
                        gt_contact_type=gt_match.get('contact_type', 'unknown')
                    )
                    
                    missing_matches.append(missing_match)
        
        if missing_matches:
            logger.info(f"Added {len(missing_matches)} missing GT matches for {frag1} ↔ {frag2}")
        
        return missing_matches
    
    def _count_gt_matches_for_pair(self, frag1: str, frag2: str, scale: str) -> int:
        """Count how many GT matches exist for a specific fragment pair at a scale."""
        gt_lookup = self.gt_matches_by_scale.get(scale, {})
        count = 0
        
        for key in gt_lookup.keys():
            gt_frag1, gt_local1, gt_frag2, gt_local2 = key
            
            # Check if this GT match involves our fragment pair (either direction)
            if ((gt_frag1 == frag1 and gt_frag2 == frag2) or 
                (gt_frag1 == frag2 and gt_frag2 == frag1)):
                count += 1
        
        # Since we store bidirectional keys, divide by 2 to get actual matches
        return count // 2
    
    def _print_precision_analysis(self, all_matches_by_scale: Dict):
        """Print precision analysis: True Positives vs False Positives per pair."""
        print("\n" + "="*80)
        print("PRECISION ANALYSIS: TP vs FP vs GT (with Recall)")
        print("="*80)
        
        for scale in self.scales:
            matches = all_matches_by_scale.get(scale, [])
            if not matches:
                continue
                
            print(f"\n{scale.upper()} SCALE:")
            print("-" * 50)
            
            # Group matches by fragment pair (normalize pair order)
            pair_stats = {}
            for match in matches:
                # Always put smaller fragment name first to avoid duplicates
                frag_a, frag_b = sorted([match.fragment_1, match.fragment_2])
                pair_key = (frag_a, frag_b)
                if pair_key not in pair_stats:
                    pair_stats[pair_key] = {'total': 0, 'gt': 0}
                pair_stats[pair_key]['total'] += 1
                if match.is_ground_truth:
                    pair_stats[pair_key]['gt'] += 1
            
            # Calculate overall metrics
            total_predicted = len(matches)
            total_true_positives = sum(1 for m in matches if m.is_ground_truth)
            total_false_positives = total_predicted - total_true_positives
            
            # Count total GT matches available for this scale
            total_gt_available = len(self.gt_matches_by_scale.get(scale, {})) // 2  # Divide by 2 for bidirectional
            
            print(f"OVERALL: {total_predicted} predicted, {total_true_positives} TP, {total_false_positives} FP")
            print(f"GT AVAILABLE: {total_gt_available} total GT matches in this scale")
            if total_predicted > 0:
                precision = total_true_positives / total_predicted
                print(f"PRECISION: {precision:.3f} ({precision*100:.1f}%)")
            if total_gt_available > 0:
                recall = total_true_positives / total_gt_available
                print(f"RECALL: {recall:.3f} ({recall*100:.1f}%)")
            
            print(f"\nPER FRAGMENT PAIR:")
            print("Format: Pair → Predicted | TP | FP | GT_Available | Precision | Recall")
            print("-" * 80)
            
            for (frag1, frag2), stats in sorted(pair_stats.items()):
                total = stats['total']
                tp = stats['gt']
                fp = total - tp
                precision = tp / total if total > 0 else 0
                
                # Count GT matches available for this specific pair
                gt_available = self._count_gt_matches_for_pair(frag1, frag2, scale)
                recall = tp / gt_available if gt_available > 0 else 0
                
                print(f"  {frag1} ↔ {frag2}: {total:4d} | {tp:3d} | {fp:4d} | {gt_available:3d} | {precision:.3f} | {recall:.3f}")
        
        print("\n" + "="*80)
    
    def _build_multi_scale_graph(self, all_matches_by_scale: Dict) -> nx.Graph:
        """Build assembly graph with multi-scale edges."""
        
        G = nx.Graph()
        
        # Add nodes for all clusters at all scales
        for frag_name, scales_dict in self.hierarchical_clusters.items():
            for scale_name, clusters in scales_dict.items():
                for i, cluster in enumerate(clusters):
                    cluster_id = self._get_cluster_attribute(cluster, 'cluster_id', f"{frag_name}_c{scale_name}_{i+1:03d}")
                    
                    node_id = f"{frag_name}_{scale_name}_{i}"
                    G.add_node(node_id,
                              fragment=frag_name,
                              scale=scale_name,
                              local_id=i,
                              cluster_id=cluster_id,
                              size_signature=self._get_cluster_attribute(cluster, 'size_signature', 0),
                              anisotropy_signature=self._get_cluster_attribute(cluster, 'anisotropy_signature', 0))
        
        # Add edges from matches
        edge_count = 0
        for scale, matches in all_matches_by_scale.items():
            for match in matches:
                node1 = f"{match.fragment_1}_{scale}_{match.local_id_1}"
                node2 = f"{match.fragment_2}_{scale}_{match.local_id_2}"
                
                if G.has_node(node1) and G.has_node(node2):
                    G.add_edge(node1, node2,
                              scale=scale,
                              match_confidence=match.match_confidence,
                              normal_similarity=match.normal_similarity,
                              is_ground_truth=match.is_ground_truth,
                              gt_confidence=match.gt_confidence,
                              weight=match.match_confidence)
                    edge_count += 1
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {edge_count} edges")
        return G
    
    def _extract_topology_features(self, all_matches_by_scale: Dict) -> Dict:
        """Extract topology features from multi-scale matches."""
        
        features = {
            'scale_statistics': {},
            'cross_scale_relationships': {},
            'fragment_connectivity': {}
        }
        
        # Scale-wise statistics
        for scale, matches in all_matches_by_scale.items():
            gt_matches = [m for m in matches if m.is_ground_truth]
            
            features['scale_statistics'][scale] = {
                'total_matches': len(matches),
                'gt_matches': len(gt_matches),
                'avg_confidence': np.mean([m.match_confidence for m in matches]) if matches else 0,
                'avg_normal_similarity': np.mean([m.normal_similarity for m in matches]) if matches else 0,
                'gt_coverage': len(gt_matches) / len(matches) if matches else 0
            }
        
        # Fragment connectivity
        for frag_name in self.hierarchical_clusters.keys():
            fragment_connections = {}
            
            for scale, matches in all_matches_by_scale.items():
                connections = [m for m in matches if frag_name in [m.fragment_1, m.fragment_2]]
                fragment_connections[scale] = len(connections)
            
            features['fragment_connectivity'][frag_name] = fragment_connections
        
        return features
    
    def _save_multi_scale_results(self, all_matches_by_scale: Dict, 
                                assembly_graph: nx.Graph, topology_features: Dict):
        """Save all results in HDF5 format."""
        
        output_file = self.output_dir / "unified_multi_scale_assembly.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Save matches by scale
            for scale, matches in all_matches_by_scale.items():
                if not matches:
                    continue
                
                scale_group = f.create_group(f'matches_{scale}')
                
                # Convert to arrays
                data_arrays = {
                    'fragment_1': [m.fragment_1.encode('utf8') for m in matches],
                    'fragment_2': [m.fragment_2.encode('utf8') for m in matches],
                    'cluster_id_1': [m.cluster_id_1.encode('utf8') for m in matches],
                    'cluster_id_2': [m.cluster_id_2.encode('utf8') for m in matches],
                    'local_id_1': [m.local_id_1 for m in matches],
                    'local_id_2': [m.local_id_2 for m in matches],
                    'normal_similarity': [m.normal_similarity for m in matches],
                    'size_similarity': [m.size_similarity for m in matches],
                    'shape_similarity': [m.shape_similarity for m in matches],
                    'spatial_proximity': [m.spatial_proximity for m in matches],
                    'match_confidence': [m.match_confidence for m in matches],
                    'is_ground_truth': [m.is_ground_truth for m in matches],
                    'gt_confidence': [m.gt_confidence for m in matches],
                    'gt_overlap_ratio_1': [m.gt_overlap_ratio_1 for m in matches],
                    'gt_overlap_ratio_2': [m.gt_overlap_ratio_2 for m in matches],
                    'gt_contact_type': [m.gt_contact_type.encode('utf8') for m in matches]
                }
                
                for key, values in data_arrays.items():
                    scale_group.create_dataset(key, data=np.array(values))
            
            # Save metadata
            metadata = f.create_group('metadata')
            metadata.attrs['scales_processed'] = [s.encode('utf8') for s in self.scales]
            metadata.attrs['total_fragments'] = len(self.hierarchical_clusters)
            
            for scale in self.scales:
                matches = all_matches_by_scale.get(scale, [])
                metadata.attrs[f'total_matches_{scale}'] = len(matches)
                metadata.attrs[f'gt_matches_{scale}'] = sum(1 for m in matches if m.is_ground_truth)
        
        logger.info(f"Saved unified assembly results to {output_file}")
    
    def _generate_comprehensive_report(self, all_matches_by_scale: Dict):
        """Generate comprehensive human-readable report."""
        
        report_file = self.output_dir / "unified_assembly_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("UNIFIED MULTI-SCALE ASSEMBLY EXTRACTION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            total_matches = sum(len(matches) for matches in all_matches_by_scale.values())
            total_gt = sum(sum(1 for m in matches if m.is_ground_truth) 
                          for matches in all_matches_by_scale.values())
            
            f.write("OVERALL SUMMARY:\n")
            f.write(f"Total matches across all scales: {total_matches:,}\n")
            f.write(f"Ground truth matches: {total_gt:,}\n")
            f.write(f"GT percentage: {total_gt/total_matches*100:.1f}%\n\n")
            
            # Scale-wise breakdown
            f.write("SCALE-WISE BREAKDOWN:\n")
            f.write("-"*50 + "\n")
            
            for scale in self.scales:
                matches = all_matches_by_scale.get(scale, [])
                gt_matches = [m for m in matches if m.is_ground_truth]
                
                f.write(f"\n{scale.upper()} Scale:\n")
                f.write(f"  Total matches: {len(matches):,}\n")
                f.write(f"  GT matches: {len(gt_matches):,}\n")
                
                if matches:
                    f.write(f"  GT percentage: {len(gt_matches)/len(matches)*100:.1f}%\n")
                    f.write(f"  Avg confidence: {np.mean([m.match_confidence for m in matches]):.3f}\n")
                    f.write(f"  Avg normal similarity: {np.mean([m.normal_similarity for m in matches]):.3f}\n")
                
                # Top matches by confidence
                if matches:
                    top_matches = sorted(matches, key=lambda m: m.match_confidence, reverse=True)[:5]
                    f.write(f"  Top 5 matches by confidence:\n")
                    for i, m in enumerate(top_matches[:5]):
                        gt_marker = " (GT)" if m.is_ground_truth else ""
                        f.write(f"    {i+1}. {m.cluster_id_1} ↔ {m.cluster_id_2}: "
                               f"conf={m.match_confidence:.3f}, norm={m.normal_similarity:.3f}{gt_marker}\n")
            
            # GT analysis
            f.write(f"\nGROUND TRUTH ANALYSIS:\n")
            f.write("-"*50 + "\n")
            
            all_gt_matches = []
            for matches in all_matches_by_scale.values():
                all_gt_matches.extend([m for m in matches if m.is_ground_truth])
            
            if all_gt_matches:
                f.write(f"Total GT matches found: {len(all_gt_matches)}\n")
                
                # GT confidence distribution
                gt_confidences = [m.gt_confidence for m in all_gt_matches if m.gt_confidence > 0]
                if gt_confidences:
                    f.write(f"GT confidence - avg: {np.mean(gt_confidences):.3f}, "
                           f"min: {np.min(gt_confidences):.3f}, max: {np.max(gt_confidences):.3f}\n")
                
                # Contact type distribution
                contact_types = {}
                for m in all_gt_matches:
                    ct = m.gt_contact_type or 'unknown'
                    contact_types[ct] = contact_types.get(ct, 0) + 1
                
                f.write(f"Contact type distribution:\n")
                for ct, count in sorted(contact_types.items()):
                    f.write(f"  {ct}: {count}\n")
        
        logger.info(f"Saved comprehensive report to {report_file}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Multi-Scale Assembly Extraction")
    parser.add_argument("--clusters", default="output/feature_clusters.pkl",
                       help="Path to hierarchical cluster file")
    parser.add_argument("--segments", default="output/segmented_fragments.pkl",
                       help="Path to segmentation file")
    parser.add_argument("--ground_truth", default="Ground_Truth/multi_scale_cluster_ground_truth.json",
                       help="Path to multi-scale ground truth file")
    parser.add_argument("--ply_dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with PLY files")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory")
    parser.add_argument("--scales", nargs='+', default=["1k", "5k", "10k"],
                       help="Scales to process")
    parser.add_argument("--contact_threshold", type=float, default=2.0,
                       help="Point-to-point contact threshold in mm")
    
    args = parser.parse_args()
    
    try:
        # Initialize extractor
        extractor = UnifiedAssemblyExtractor(
            clusters_file=args.clusters,
            segments_file=args.segments,
            ground_truth_file=args.ground_truth,
            ply_dir=args.ply_dir,
            output_dir=args.output_dir,
            scales=args.scales,
            contact_threshold=args.contact_threshold
        )
        
        # Extract assembly knowledge
        all_matches_by_scale, assembly_graph = extractor.extract_multi_scale_assembly_knowledge()
        
        # Print summary
        print("\n" + "="*70)
        print("UNIFIED ASSEMBLY EXTRACTION COMPLETE")
        print("="*70)
        
        total_matches = sum(len(matches) for matches in all_matches_by_scale.values())
        total_gt = sum(sum(1 for m in matches if m.is_ground_truth) 
                      for matches in all_matches_by_scale.values())
        
        print(f"\nOVERALL SUMMARY:")
        print(f"Contact pairs: {len(extractor.contact_pairs)}")
        print(f"Total matches: {total_matches:,}")
        print(f"True positives: {total_gt:,}")
        print(f"False positives: {total_matches - total_gt:,}")
        if total_matches > 0:
            precision = total_gt / total_matches
            print(f"Overall precision: {precision:.3f} ({precision*100:.1f}%)")
        
        print(f"\nResults saved to: {args.output_dir}")
        
        logger.info("Unified assembly extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Assembly extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()