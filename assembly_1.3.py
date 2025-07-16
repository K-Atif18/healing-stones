#!/usr/bin/env python3
"""
Phase 1.3: Assembly Knowledge with Cluster Relationships
Uses ground truth from Blender to identify matching cluster pairs.
"""

import numpy as np
import pickle
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import logging
from tqdm import tqdm
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusterMatch:
    """Represents a potential match between clusters from different fragments."""
    cluster_id_1: int
    cluster_id_2: int
    fragment_1: str
    fragment_2: str
    distance: float
    normal_similarity: float
    is_ground_truth: bool
    confidence: float

class AssemblyKnowledgeExtractor:
    def __init__(self, 
                 clusters_file: str = "output/feature_clusters.pkl",
                 segments_file: str = "output/segmented_fragments.pkl",
                 ground_truth_file: str = "Ground_Truth/blender/ground_truth_assembly.json",
                 output_dir: str = "output"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load cluster data
        logger.info("Loading cluster data...")
        with open(clusters_file, 'rb') as f:
            self.cluster_data = pickle.load(f)
        
        # Load segmentation data
        logger.info("Loading segmentation data...")
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Load ground truth
        logger.info("Loading ground truth assembly...")
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        # Organize clusters by fragment
        self._organize_clusters()
        
        # Parameters
        self.contact_threshold = 5.0  # mm
        self.normal_angle_threshold = 30.0  # degrees
        
    def _organize_clusters(self):
        """Organize clusters by their source fragment."""
        self.clusters_by_fragment = {}
        self.cluster_lookup = {}
        
        # First pass: build lookup table
        for cluster in self.cluster_data['clusters']:
            self.cluster_lookup[cluster['cluster_id']] = cluster
        
        # Second pass: organize by fragment
        # This requires tracking which fragment each cluster belongs to
        # We'll need to enhance the cluster data to include fragment info
        logger.info("Organizing clusters by fragment...")
        
        # For now, we'll use a simple heuristic based on cluster IDs
        # In practice, you should store fragment info with each cluster
        clusters_per_fragment = len(self.cluster_data['clusters']) // len(self.segment_data)
        
        fragment_names = sorted(self.segment_data.keys())
        for i, fragment_name in enumerate(fragment_names):
            start_idx = i * clusters_per_fragment
            end_idx = (i + 1) * clusters_per_fragment if i < len(fragment_names) - 1 else len(self.cluster_data['clusters'])
            
            fragment_clusters = []
            for j in range(start_idx, end_idx):
                if j < len(self.cluster_data['clusters']):
                    cluster = self.cluster_data['clusters'][j]
                    cluster['fragment'] = fragment_name
                    fragment_clusters.append(cluster)
            
            self.clusters_by_fragment[fragment_name] = fragment_clusters
    
    def extract_assembly_knowledge(self):
        """Main method to extract assembly knowledge."""
        logger.info("Extracting assembly knowledge...")
        
        # 1. Apply ground truth transforms to clusters
        transformed_clusters = self._transform_clusters_to_world()
        
        # 2. Find cluster-level contacts
        cluster_matches = self._find_cluster_contacts(transformed_clusters)
        
        # 3. Build assembly graph
        assembly_graph = self._build_assembly_graph(cluster_matches)
        
        # 4. Mine overlap topology
        topology_features = self._mine_topology_features()
        
        # 5. Save results
        self._save_assembly_knowledge(cluster_matches, assembly_graph, topology_features)
        
        return cluster_matches, assembly_graph
    
    def _transform_clusters_to_world(self):
        """Transform cluster positions using ground truth poses."""
        transformed_clusters = {}
        
        for fragment_name, fragment_data in self.ground_truth['fragments'].items():
            if fragment_name not in self.clusters_by_fragment:
                logger.warning(f"No clusters found for fragment {fragment_name}")
                continue
            
            # Get transformation matrix
            transform_matrix = np.array(fragment_data['transform_matrix'])
            
            # Transform each cluster's barycenter
            fragment_clusters = []
            for cluster in self.clusters_by_fragment[fragment_name]:
                # Create a copy of the cluster
                transformed_cluster = cluster.copy()
                
                # Transform barycenter (add homogeneous coordinate)
                barycenter_homo = np.append(cluster['barycenter'], 1.0)
                transformed_bary = transform_matrix @ barycenter_homo
                transformed_cluster['barycenter_world'] = transformed_bary[:3]
                
                # Transform principal axes (rotation only)
                rotation_matrix = transform_matrix[:3, :3]
                transformed_axes = rotation_matrix @ cluster['principal_axes']
                transformed_cluster['principal_axes_world'] = transformed_axes
                
                fragment_clusters.append(transformed_cluster)
            
            transformed_clusters[fragment_name] = fragment_clusters
        
        return transformed_clusters
    
    def _find_cluster_contacts(self, transformed_clusters):
        """Find which clusters are in contact across fragments."""
        cluster_matches = []
        
        # Get contact pairs from ground truth
        contact_pairs = self.ground_truth['contact_pairs']
        
        for frag1_name, frag2_name in tqdm(contact_pairs, desc="Finding cluster contacts"):
            if frag1_name not in transformed_clusters or frag2_name not in transformed_clusters:
                continue
            
            clusters1 = transformed_clusters[frag1_name]
            clusters2 = transformed_clusters[frag2_name]
            
            # Build KD-tree for efficient nearest neighbor search
            positions2 = np.array([c['barycenter_world'] for c in clusters2])
            tree2 = cKDTree(positions2)
            
            # Find nearby cluster pairs
            for cluster1 in clusters1:
                pos1 = cluster1['barycenter_world']
                
                # Find clusters within contact threshold
                indices = tree2.query_ball_point(pos1, self.contact_threshold * 2)  # 2x for cluster scale
                
                for idx in indices:
                    cluster2 = clusters2[idx]
                    
                    # Calculate match metrics
                    distance = np.linalg.norm(pos1 - cluster2['barycenter_world'])
                    
                    # Compare principal directions (normals)
                    # Use the first principal axis as the main normal
                    normal1 = cluster1['principal_axes_world'][:, 0]
                    normal2 = cluster2['principal_axes_world'][:, 0]
                    
                    # Check if normals are opposing (for matching surfaces)
                    normal_dot = np.dot(normal1, -normal2)  # Negative because matching surfaces face each other
                    normal_similarity = (normal_dot + 1) / 2  # Normalize to [0, 1]
                    
                    # Check if this is a ground truth match
                    is_gt = self._is_ground_truth_match(cluster1, cluster2, frag1_name, frag2_name)
                    
                    # Calculate confidence based on distance and normal alignment
                    confidence = self._calculate_match_confidence(distance, normal_similarity)
                    
                    match = ClusterMatch(
                        cluster_id_1=cluster1['cluster_id'],
                        cluster_id_2=cluster2['cluster_id'],
                        fragment_1=frag1_name,
                        fragment_2=frag2_name,
                        distance=distance,
                        normal_similarity=normal_similarity,
                        is_ground_truth=is_gt,
                        confidence=confidence
                    )
                    
                    cluster_matches.append(match)
        
        logger.info(f"Found {len(cluster_matches)} potential cluster matches")
        gt_matches = sum(1 for m in cluster_matches if m.is_ground_truth)
        logger.info(f"Ground truth matches: {gt_matches}")
        
        return cluster_matches
    
    def _is_ground_truth_match(self, cluster1, cluster2, frag1_name, frag2_name):
        for contact in self.ground_truth.get('contact_details', []):
            if ((contact['fragment_1'] == frag1_name and contact['fragment_2'] == frag2_name) or
                (contact['fragment_1'] == frag2_name and contact['fragment_2'] == frag1_name)):

                # Determine which cluster belongs to which fragment in this contact
                if contact['fragment_1'] == cluster1['fragment']:
                    contact_local = np.array(contact['contact_center_1'])
                    transform = np.array(self.ground_truth['fragments'][contact['fragment_1']]['transform_matrix'])
                else:
                    contact_local = np.array(contact['contact_center_2'])
                    transform = np.array(self.ground_truth['fragments'][contact['fragment_2']]['transform_matrix'])

                # Transform to world space
                contact_world = (transform @ np.append(contact_local, 1.0))[:3]

                dist1 = np.linalg.norm(cluster1['barycenter_world'] - contact_world)
                dist2 = np.linalg.norm(cluster2['barycenter_world'] - contact_world)

                if dist1 < cluster1['scale'] and dist2 < cluster2['scale']:
                    return True
        return False
    
    
    def _calculate_match_confidence(self, distance, normal_similarity):
        """Calculate confidence score for a cluster match."""
        # Distance component (closer is better)
        dist_score = np.exp(-distance / self.contact_threshold)
        
        # Normal component (more aligned is better)
        normal_score = normal_similarity
        
        # Combined confidence
        confidence = 0.7 * dist_score + 0.3 * normal_score
        
        return float(confidence)
    
    def _build_assembly_graph(self, cluster_matches):
        """Build a graph representing assembly relationships."""
        G = nx.Graph()
        
        # Add nodes (clusters)
        for fragment_name, clusters in self.clusters_by_fragment.items():
            for cluster in clusters:
                node_id = f"{fragment_name}_{cluster['cluster_id']}"
                G.add_node(node_id, 
                          cluster_id=cluster['cluster_id'],
                          fragment=fragment_name,
                          scale=cluster['scale'],
                          size_signature=cluster['size_signature'],
                          anisotropy=cluster['anisotropy_signature'])
        
        # Add intra-fragment edges (from overlap graph)
        for edge in self.cluster_data['overlap_graph_edges']:
            # Find which fragments these clusters belong to
            cluster1 = self.cluster_lookup[edge[0]]
            cluster2 = self.cluster_lookup[edge[1]]
            
            if cluster1.get('fragment') and cluster2.get('fragment'):
                if cluster1['fragment'] == cluster2['fragment']:
                    node1 = f"{cluster1['fragment']}_{edge[0]}"
                    node2 = f"{cluster2['fragment']}_{edge[1]}"
                    G.add_edge(node1, node2, 
                              type='intra_fragment',
                              weight=1.0)
        
        # Add inter-fragment edges (potential matches)
        for match in cluster_matches:
            node1 = f"{match.fragment_1}_{match.cluster_id_1}"
            node2 = f"{match.fragment_2}_{match.cluster_id_2}"
            
            G.add_edge(node1, node2,
                      type='inter_fragment',
                      distance=match.distance,
                      normal_similarity=match.normal_similarity,
                      is_ground_truth=match.is_ground_truth,
                      confidence=match.confidence,
                      weight=match.confidence)
        
        logger.info(f"Assembly graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def _mine_topology_features(self):
        """Extract topological features from the cluster relationships."""
        features = {
            'multi_scale_hierarchy': {},
            'neighborhood_patterns': {},
            'cross_fragment_patterns': {}
        }
        
        # Multi-scale hierarchy
        for fragment_name, clusters in self.clusters_by_fragment.items():
            # Group clusters by scale
            by_scale = {}
            for cluster in clusters:
                scale = cluster['scale']
                if scale not in by_scale:
                    by_scale[scale] = []
                by_scale[scale].append(cluster)
            
            # Find parent-child relationships
            scales = sorted(by_scale.keys())
            hierarchy = []
            
            for i in range(len(scales) - 1):
                small_scale = scales[i]
                large_scale = scales[i + 1]
                
                for small_cluster in by_scale[small_scale]:
                    for large_cluster in by_scale[large_scale]:
                        # Check if small cluster is contained in large cluster
                        dist = np.linalg.norm(
                            np.array(small_cluster['barycenter']) - 
                            np.array(large_cluster['barycenter'])
                        )
                        
                        if dist < large_cluster['scale'] / 2:
                            hierarchy.append({
                                'parent': large_cluster['cluster_id'],
                                'child': small_cluster['cluster_id'],
                                'scale_ratio': large_scale / small_scale
                            })
            
            features['multi_scale_hierarchy'][fragment_name] = hierarchy
        
        return features
    
    def _save_assembly_knowledge(self, cluster_matches, assembly_graph, topology_features):
        """Save assembly knowledge to HDF5 file."""
        output_file = self.output_dir / "cluster_assembly_priors.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Save cluster matches
            matches_group = f.create_group('cluster_matches')
            
            # Convert matches to arrays
            n_matches = len(cluster_matches)
            cluster_ids_1 = np.array([m.cluster_id_1 for m in cluster_matches])
            cluster_ids_2 = np.array([m.cluster_id_2 for m in cluster_matches])
            distances = np.array([m.distance for m in cluster_matches])
            normal_sims = np.array([m.normal_similarity for m in cluster_matches])
            is_gt = np.array([m.is_ground_truth for m in cluster_matches])
            confidences = np.array([m.confidence for m in cluster_matches])
            
            matches_group.create_dataset('cluster_id_1', data=cluster_ids_1)
            matches_group.create_dataset('cluster_id_2', data=cluster_ids_2)
            matches_group.create_dataset('distances', data=distances)
            matches_group.create_dataset('normal_similarities', data=normal_sims)
            matches_group.create_dataset('is_ground_truth', data=is_gt)
            matches_group.create_dataset('confidences', data=confidences)
            
            # Save fragment names as attributes
            frag1_names = [m.fragment_1 for m in cluster_matches]
            frag2_names = [m.fragment_2 for m in cluster_matches]
            matches_group.create_dataset('fragment_1', data=np.array(frag1_names, dtype='S'))
            matches_group.create_dataset('fragment_2', data=np.array(frag2_names, dtype='S'))
            
            # Save assembly graph
            graph_group = f.create_group('assembly_graph')
            
            # Convert graph to adjacency matrix
            node_list = list(assembly_graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(node_list)}
            n_nodes = len(node_list)
            
            # Save node information
            graph_group.create_dataset('node_names', data=np.array(node_list, dtype='S'))
            graph_group.attrs['n_nodes'] = n_nodes
            graph_group.attrs['n_edges'] = assembly_graph.number_of_edges()
            
            # Save edges with attributes
            edges_data = []
            for u, v, data in assembly_graph.edges(data=True):
                edge_info = {
                    'source': node_to_idx[u],
                    'target': node_to_idx[v],
                    'type': data.get('type', 'unknown'),
                    'weight': data.get('weight', 1.0),
                    'is_ground_truth': data.get('is_ground_truth', False)
                }
                edges_data.append(edge_info)
            
            # Store edges as structured array
            if edges_data:
                edge_dtype = np.dtype([
                    ('source', 'i4'),
                    ('target', 'i4'),
                    ('weight', 'f4'),
                    ('is_ground_truth', 'bool')
                ])
                edges_array = np.zeros(len(edges_data), dtype=edge_dtype)
                for i, edge in enumerate(edges_data):
                    edges_array[i] = (
                        edge['source'],
                        edge['target'],
                        edge['weight'],
                        edge['is_ground_truth']
                    )
                graph_group.create_dataset('edges', data=edges_array)
            
            # Save topology features
            topo_group = f.create_group('topology_features')
            
            # Save multi-scale hierarchy
            hierarchy_group = topo_group.create_group('multi_scale_hierarchy')
            for fragment_name, hierarchy in topology_features['multi_scale_hierarchy'].items():
                if hierarchy:
                    frag_group = hierarchy_group.create_group(fragment_name)
                    parent_ids = [h['parent'] for h in hierarchy]
                    child_ids = [h['child'] for h in hierarchy]
                    scale_ratios = [h['scale_ratio'] for h in hierarchy]
                    
                    frag_group.create_dataset('parent_ids', data=parent_ids)
                    frag_group.create_dataset('child_ids', data=child_ids)
                    frag_group.create_dataset('scale_ratios', data=scale_ratios)
            
            # Save metadata
            metadata = f.create_group('metadata')
            metadata.attrs['contact_threshold'] = self.contact_threshold
            metadata.attrs['normal_angle_threshold'] = self.normal_angle_threshold
            metadata.attrs['n_fragments'] = len(self.clusters_by_fragment)
            metadata.attrs['n_clusters_total'] = len(self.cluster_data['clusters'])
            metadata.attrs['n_matches'] = n_matches
            metadata.attrs['n_gt_matches'] = int(np.sum(is_gt))
        
        logger.info(f"Saved assembly knowledge to {output_file}")
        
        # Also save a summary report
        self._save_summary_report(cluster_matches, assembly_graph)
    
    def _save_summary_report(self, cluster_matches, assembly_graph):
        """Save a human-readable summary report."""
        report_file = self.output_dir / "assembly_knowledge_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("ASSEMBLY KNOWLEDGE EXTRACTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Fragment summary
            f.write("FRAGMENTS:\n")
            f.write("-"*30 + "\n")
            for fragment_name in sorted(self.clusters_by_fragment.keys()):
                n_clusters = len(self.clusters_by_fragment[fragment_name])
                f.write(f"{fragment_name}: {n_clusters} clusters\n")
            f.write("\n")
            
            # Contact pairs from ground truth
            f.write("GROUND TRUTH CONTACT PAIRS:\n")
            f.write("-"*30 + "\n")
            for pair in self.ground_truth['contact_pairs']:
                f.write(f"{pair[0]} <-> {pair[1]}\n")
            f.write("\n")
            
            # Cluster matches summary
            f.write("CLUSTER MATCHES:\n")
            f.write("-"*30 + "\n")
            f.write(f"Total potential matches: {len(cluster_matches)}\n")
            
            gt_matches = [m for m in cluster_matches if m.is_ground_truth]
            f.write(f"Ground truth matches: {len(gt_matches)}\n")
            
            high_conf_matches = [m for m in cluster_matches if m.confidence > 0.8]
            f.write(f"High confidence matches (>0.8): {len(high_conf_matches)}\n")
            
            # Statistics by fragment pair
            f.write("\nMatches by fragment pair:\n")
            pair_stats = {}
            for match in cluster_matches:
                pair = tuple(sorted([match.fragment_1, match.fragment_2]))
                if pair not in pair_stats:
                    pair_stats[pair] = {'total': 0, 'gt': 0}
                pair_stats[pair]['total'] += 1
                if match.is_ground_truth:
                    pair_stats[pair]['gt'] += 1
            
            for pair, stats in sorted(pair_stats.items()):
                f.write(f"  {pair[0]} <-> {pair[1]}: {stats['total']} matches ({stats['gt']} GT)\n")
            
            # Assembly graph summary
            f.write(f"\nASSEMBLY GRAPH:\n")
            f.write("-"*30 + "\n")
            f.write(f"Nodes: {assembly_graph.number_of_nodes()}\n")
            f.write(f"Edges: {assembly_graph.number_of_edges()}\n")
            
            # Edge type breakdown
            edge_types = {}
            for u, v, data in assembly_graph.edges(data=True):
                edge_type = data.get('type', 'unknown')
                if edge_type not in edge_types:
                    edge_types[edge_type] = 0
                edge_types[edge_type] += 1
            
            f.write("\nEdge types:\n")
            for edge_type, count in edge_types.items():
                f.write(f"  {edge_type}: {count}\n")
            
            # Sample high-confidence matches
            f.write(f"\nSAMPLE HIGH-CONFIDENCE MATCHES:\n")
            f.write("-"*30 + "\n")
            
            sorted_matches = sorted(cluster_matches, key=lambda m: m.confidence, reverse=True)
            for match in sorted_matches[:10]:
                f.write(f"Cluster {match.cluster_id_1} ({match.fragment_1}) <-> "
                       f"Cluster {match.cluster_id_2} ({match.fragment_2})\n")
                f.write(f"  Distance: {match.distance:.2f}mm\n")
                f.write(f"  Normal similarity: {match.normal_similarity:.3f}\n")
                f.write(f"  Confidence: {match.confidence:.3f}\n")
                f.write(f"  Ground truth: {'YES' if match.is_ground_truth else 'NO'}\n")
                f.write("\n")
        
        logger.info(f"Saved summary report to {report_file}")


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1.3: Assembly Knowledge Extraction")
    parser.add_argument("--clusters", default="output/feature_clusters.pkl",
                       help="Path to feature clusters file")
    parser.add_argument("--segments", default="output/segmented_fragments.pkl",
                       help="Path to segmented fragments file")
    parser.add_argument("--ground_truth", default="Ground_Truth/blender/ground_truth_assembly.json",
                       help="Path to ground truth assembly file")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory")
    parser.add_argument("--contact_threshold", type=float, default=5.0,
                       help="Contact distance threshold in mm")
    
    args = parser.parse_args()
    
    # Check if ground truth exists
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        logger.error(f"Ground truth file not found: {gt_path}")
        logger.error("Please run the Blender extraction script first!")
        return
    
    # Extract assembly knowledge
    extractor = AssemblyKnowledgeExtractor(
        clusters_file=args.clusters,
        segments_file=args.segments,
        ground_truth_file=args.ground_truth,
        output_dir=args.output_dir
    )
    
    extractor.contact_threshold = args.contact_threshold
    
    # Run extraction
    cluster_matches, assembly_graph = extractor.extract_assembly_knowledge()
    
    logger.info("Phase 1.3 complete!")
    logger.info(f"Output saved to {args.output_dir}/cluster_assembly_priors.h5")


if __name__ == "__main__":
    main()