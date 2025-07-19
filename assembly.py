#!/usr/bin/env python3
"""
Updated Assembly Knowledge Extractor with Cluster-Level Ground Truth Support
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
    contact_point_ratio: float = 0.0
    gt_confidence: float = 0.0  # Ground truth confidence if available

class EnhancedAssemblyExtractor:
    def __init__(self, 
                 clusters_file: str = "output/feature_clusters_fixed.pkl",
                 segments_file: str = "output/segmented_fragments.pkl",
                 ground_truth_file: str = "Ground_Truth/cluster_level_ground_truth.json",
                 ply_dir: str = "Ground_Truth/reconstructed/artifact_1",
                 output_dir: str = "output"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ply_dir = Path(ply_dir)
        
        # Parameters
        self.contact_threshold = 5.0  # mm - threshold for cluster matching
        self.break_surface_threshold = 10.0  # mm - threshold to check if cluster is on break surface
        
        # Load all data
        self._load_data(clusters_file, segments_file, ground_truth_file)
        
        # Build ground truth lookup for fast access
        self._build_gt_lookup()
        
    def _load_data(self, clusters_file, segments_file, ground_truth_file):
        """Load all required data including cluster-level ground truth."""
        logger.info("Loading data...")
        
        # Load clusters
        with open(clusters_file, 'rb') as f:
            self.cluster_data = pickle.load(f)
        
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Load ground truth - try cluster-level first, fall back to original
        if Path(ground_truth_file).exists():
            with open(ground_truth_file, 'r') as f:
                self.ground_truth = json.load(f)
            logger.info("Loaded cluster-level ground truth")
        else:
            # Fall back to original ground truth
            fallback_file = "Ground_Truth/ground_truth_from_positioned.json"
            with open(fallback_file, 'r') as f:
                self.ground_truth = json.load(f)
            logger.warning(f"Cluster-level GT not found, using fragment-level GT")
        
        # Check if using pre-positioned fragments
        self.using_positioned = self.ground_truth.get('extraction_info', {}).get('fragments_are_pre_positioned', False)
        if self.using_positioned:
            logger.info("Using pre-positioned fragments - no transformation needed")
        
        # Organize clusters by fragment
        self._organize_clusters_improved()
        
        # Load fragment point clouds and identify break surface clusters
        self._identify_break_surface_clusters()
    
    def _build_gt_lookup(self):
        """Build lookup table for ground truth cluster matches."""
        self.gt_matches_lookup = {}
        self.gt_matches_by_pair = {}
        
        # Check if we have cluster-level ground truth
        if 'cluster_ground_truth_matches' in self.ground_truth:
            logger.info("Building cluster-level ground truth lookup...")
            
            for match in self.ground_truth['cluster_ground_truth_matches']:
                frag1 = match['fragment_1']
                frag2 = match['fragment_2']
                cluster1 = match['cluster_id_1']
                cluster2 = match['cluster_id_2']
                
                # Create bidirectional lookup
                key1 = (frag1, cluster1, frag2, cluster2)
                key2 = (frag2, cluster2, frag1, cluster1)
                
                self.gt_matches_lookup[key1] = match
                self.gt_matches_lookup[key2] = match
                
                # Store by fragment pair
                pair = tuple(sorted([frag1, frag2]))
                if pair not in self.gt_matches_by_pair:
                    self.gt_matches_by_pair[pair] = []
                self.gt_matches_by_pair[pair].append(match)
            
            logger.info(f"Loaded {len(self.ground_truth['cluster_ground_truth_matches'])} cluster GT matches")
        else:
            logger.warning("No cluster-level ground truth found in file")
    
    def _organize_clusters_improved(self):
        """Organize clusters by fragment using segment data counts."""
        self.clusters_by_fragment = {}
        self.cluster_lookup = {}
        
        fragment_names = sorted(self.segment_data.keys())
        cluster_idx = 0
        
        for frag_name in fragment_names:
            n_clusters = self.segment_data[frag_name].get('n_clusters', 0)
            fragment_clusters = []
            
            for i in range(n_clusters):
                if cluster_idx < len(self.cluster_data['clusters']):
                    cluster = self.cluster_data['clusters'][cluster_idx].copy()
                    cluster['fragment'] = frag_name
                    cluster['global_id'] = cluster_idx
                    cluster['local_id'] = i  # Add local ID within fragment
                    
                    self.cluster_lookup[cluster['cluster_id']] = cluster
                    fragment_clusters.append(cluster)
                    cluster_idx += 1
            
            self.clusters_by_fragment[frag_name] = fragment_clusters
            logger.info(f"{frag_name}: assigned {len(fragment_clusters)} clusters")
    
    def _identify_break_surface_clusters(self):
        """Use all available clusters for each fragment."""
        logger.info("Preparing clusters for matching...")
        
        self.break_surface_clusters = {}
        self.break_surface_points = {}
        
        for frag_name in self.segment_data.keys():
            ply_file = self.ply_dir / f"{frag_name}.ply"
            
            if not ply_file.exists():
                logger.warning(f"PLY file not found: {ply_file}")
                continue
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_file))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Get break surface points (green) for reference
            green_mask = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.4) & (colors[:, 2] < 0.4)
            break_points = points[green_mask]
            self.break_surface_points[frag_name] = break_points
            
            # Prepare all clusters
            all_clusters = []
            for cluster in self.clusters_by_fragment.get(frag_name, []):
                # Transform cluster position if needed
                if self.using_positioned:
                    cluster_pos = cluster['barycenter']
                else:
                    # Apply transform if available
                    transform = np.array(self.ground_truth['fragments'][frag_name]['transform_matrix'])
                    bary_homo = np.append(cluster['barycenter'], 1.0)
                    cluster_pos = (transform @ bary_homo)[:3]
                
                cluster['world_position'] = cluster_pos
                cluster['is_break_surface'] = True  # Consider all clusters
                all_clusters.append(cluster)
            
            self.break_surface_clusters[frag_name] = all_clusters
            logger.info(f"{frag_name}: prepared {len(all_clusters)} clusters")
    
    def extract_assembly_knowledge(self):
        """Extract assembly knowledge with cluster-level ground truth."""
        logger.info("Extracting assembly knowledge with cluster GT...")
        
        # Find cluster matches for each contact pair
        all_matches = []
        
        # Get contact pairs from ground truth or detect them
        if 'contact_pairs' in self.ground_truth:
            contact_pairs = self.ground_truth['contact_pairs']
        else:
            # Detect from cluster GT matches
            contact_pairs = list(self.gt_matches_by_pair.keys())
        
        for pair in contact_pairs:
            if isinstance(pair, list):
                frag1, frag2 = pair[0], pair[1]
            else:
                frag1, frag2 = pair[0], pair[1]
            
            if frag1 not in self.break_surface_clusters or frag2 not in self.break_surface_clusters:
                continue
            
            logger.info(f"Processing contact pair: {frag1} <-> {frag2}")
            
            # Get ground truth matches for this pair if available
            pair_key = tuple(sorted([frag1, frag2]))
            gt_matches_for_pair = self.gt_matches_by_pair.get(pair_key, [])
            
            # Find matches between clusters
            pair_matches = self._find_cluster_matches_for_pair(
                frag1, frag2, gt_matches_for_pair
            )
            
            all_matches.extend(pair_matches)
        
        # Build assembly graph
        assembly_graph = self._build_assembly_graph(all_matches)
        
        # Mine topology features
        topology_features = self._mine_topology_features()
        
        # Save results
        self._save_assembly_knowledge(all_matches, assembly_graph, topology_features)
        
        # Log statistics
        total_matches = len(all_matches)
        gt_matches = sum(1 for m in all_matches if m.is_ground_truth)
        logger.info(f"\nTotal matches found: {total_matches}")
        logger.info(f"Ground truth matches identified: {gt_matches}")
        if total_matches > 0:
            logger.info(f"GT percentage: {gt_matches/total_matches*100:.1f}%")
        
        return all_matches, assembly_graph
    
    def _find_cluster_matches_for_pair(self, frag1: str, frag2: str, gt_matches_for_pair: List):
        """Find all potential matches between clusters of two fragments."""
        clusters1 = self.break_surface_clusters[frag1]
        clusters2 = self.break_surface_clusters[frag2]
        
        if not clusters1 or not clusters2:
            return []
        
        matches = []
        
        # Build KD-tree for efficient search
        positions2 = np.array([c['world_position'] for c in clusters2])
        tree2 = cKDTree(positions2)
        
        # For each cluster in frag1, find potential matches in frag2
        for cluster1 in clusters1:
            pos1 = cluster1['world_position']
            
            # Find nearby clusters
            search_radius = max(cluster1['scale'] * 2, self.contact_threshold * 2)
            indices = tree2.query_ball_point(pos1, search_radius)
            
            for idx in indices:
                cluster2 = clusters2[idx]
                
                # Check if this is a ground truth match
                is_gt, gt_info = self._check_cluster_gt_match(
                    cluster1, cluster2, frag1, frag2, gt_matches_for_pair
                )
                
                # Calculate match metrics
                match = self._evaluate_cluster_match(
                    cluster1, cluster2, frag1, frag2, is_gt, gt_info
                )
                
                if match.confidence > 0.1:  # Keep all reasonable matches
                    matches.append(match)
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Log statistics
        if matches:
            logger.info(f"  Found {len(matches)} matches, top confidence: {matches[0].confidence:.3f}")
            gt_count = sum(1 for m in matches if m.is_ground_truth)
            logger.info(f"  Ground truth matches: {gt_count}")
            
            # Log some GT matches for debugging
            gt_matches = [m for m in matches if m.is_ground_truth]
            for i, m in enumerate(gt_matches[:3]):
                logger.info(f"    GT match {i+1}: C{m.cluster_id_1} <-> C{m.cluster_id_2}, "
                           f"conf: {m.confidence:.3f}, gt_conf: {m.gt_confidence:.3f}")
        
        return matches
    
    def _check_cluster_gt_match(self, cluster1, cluster2, frag1, frag2, gt_matches_for_pair):
        """Check if two clusters form a ground truth match."""
        # Use local cluster IDs for matching
        cluster_id_1 = cluster1.get('local_id', cluster1['cluster_id'])
        cluster_id_2 = cluster2.get('local_id', cluster2['cluster_id'])
        
        # Check in GT lookup
        key = (frag1, cluster_id_1, frag2, cluster_id_2)
        if key in self.gt_matches_lookup:
            gt_match = self.gt_matches_lookup[key]
            return True, gt_match
        
        # Also check reverse key
        key_rev = (frag2, cluster_id_2, frag1, cluster_id_1)
        if key_rev in self.gt_matches_lookup:
            gt_match = self.gt_matches_lookup[key_rev]
            return True, gt_match
        
        # Check in the pair-specific list
        for gt_match in gt_matches_for_pair:
            if ((gt_match['fragment_1'] == frag1 and gt_match['cluster_id_1'] == cluster_id_1 and
                 gt_match['fragment_2'] == frag2 and gt_match['cluster_id_2'] == cluster_id_2) or
                (gt_match['fragment_1'] == frag2 and gt_match['cluster_id_1'] == cluster_id_2 and
                 gt_match['fragment_2'] == frag1 and gt_match['cluster_id_2'] == cluster_id_1)):
                return True, gt_match
        
        return False, None
    
    def _evaluate_cluster_match(self, cluster1, cluster2, frag1, frag2, is_gt, gt_info):
        """Evaluate if two clusters form a good match."""
        pos1 = cluster1['world_position']
        pos2 = cluster2['world_position']
        
        # Distance between clusters
        distance = np.linalg.norm(pos1 - pos2)
        
        # Normal similarity
        normal_similarity = self._compute_normal_similarity(cluster1, cluster2, frag1, frag2)
        
        # Calculate confidence
        dist_score = np.exp(-distance / self.contact_threshold)
        
        # Base confidence
        confidence = 0.5 * dist_score + 0.5 * normal_similarity
        
        # Get GT confidence if available
        gt_confidence = 0.0
        if is_gt and gt_info:
            gt_confidence = gt_info.get('confidence', 0.8)
            # Boost confidence for GT matches
            confidence = max(confidence, 0.7)
        
        # Use local IDs for the match
        cluster_id_1 = cluster1.get('local_id', cluster1['cluster_id'])
        cluster_id_2 = cluster2.get('local_id', cluster2['cluster_id'])
        
        return ClusterMatch(
            cluster_id_1=cluster_id_1,
            cluster_id_2=cluster_id_2,
            fragment_1=frag1,
            fragment_2=frag2,
            distance=distance,
            normal_similarity=normal_similarity,
            is_ground_truth=is_gt,
            confidence=float(confidence),
            contact_point_ratio=0.0,  # Not computed here
            gt_confidence=float(gt_confidence)
        )
    
    def _compute_normal_similarity(self, cluster1, cluster2, frag1, frag2):
        """Compute normal similarity between clusters."""
        if 'principal_axes' not in cluster1 or 'principal_axes' not in cluster2:
            return 0.5
        
        # Transform axes if needed
        if self.using_positioned:
            axes1 = cluster1['principal_axes']
            axes2 = cluster2['principal_axes']
        else:
            # Apply rotation part of transform
            transform1 = np.array(self.ground_truth['fragments'][frag1]['transform_matrix'])
            transform2 = np.array(self.ground_truth['fragments'][frag2]['transform_matrix'])
            axes1 = transform1[:3, :3] @ cluster1['principal_axes']
            axes2 = transform2[:3, :3] @ cluster2['principal_axes']
        
        normal1 = axes1[:, 0]
        normal2 = axes2[:, 0]
        
        # Matching surfaces should have opposing normals
        normal_dot = np.dot(normal1, -normal2)
        normal_similarity = (normal_dot + 1) / 2  # Normalize to [0, 1]
        
        return normal_similarity
    
    def _build_assembly_graph(self, cluster_matches):
        """Build assembly graph with intra and inter-fragment edges."""
        G = nx.Graph()
        
        # Add nodes for all clusters
        for fragment_name, clusters in self.clusters_by_fragment.items():
            for cluster in clusters:
                node_id = f"{fragment_name}_{cluster['cluster_id']}"
                G.add_node(node_id, 
                          cluster_id=cluster['cluster_id'],
                          fragment=fragment_name,
                          scale=cluster['scale'],
                          is_break_surface=cluster.get('is_break_surface', False))
        
        # Add intra-fragment edges from overlap graph
        for edge in self.cluster_data.get('overlap_graph_edges', []):
            cluster1 = self.cluster_lookup.get(edge[0])
            cluster2 = self.cluster_lookup.get(edge[1])
            
            if cluster1 and cluster2 and cluster1.get('fragment') == cluster2.get('fragment'):
                node1 = f"{cluster1['fragment']}_{edge[0]}"
                node2 = f"{cluster2['fragment']}_{edge[1]}"
                if G.has_node(node1) and G.has_node(node2):
                    G.add_edge(node1, node2, type='intra_fragment', weight=1.0)
        
        # Add inter-fragment edges from matches
        for match in cluster_matches:
            # Use global cluster IDs for graph nodes
            cluster1_global = None
            cluster2_global = None
            
            # Find global IDs from local IDs
            for c in self.clusters_by_fragment.get(match.fragment_1, []):
                if c.get('local_id', c['cluster_id']) == match.cluster_id_1:
                    cluster1_global = c['cluster_id']
                    break
            
            for c in self.clusters_by_fragment.get(match.fragment_2, []):
                if c.get('local_id', c['cluster_id']) == match.cluster_id_2:
                    cluster2_global = c['cluster_id']
                    break
            
            if cluster1_global is not None and cluster2_global is not None:
                node1 = f"{match.fragment_1}_{cluster1_global}"
                node2 = f"{match.fragment_2}_{cluster2_global}"
                
                if G.has_node(node1) and G.has_node(node2):
                    G.add_edge(node1, node2,
                              type='inter_fragment',
                              distance=match.distance,
                              is_ground_truth=match.is_ground_truth,
                              confidence=match.confidence,
                              gt_confidence=match.gt_confidence,
                              weight=match.confidence)
        
        return G
    
    def _mine_topology_features(self):
        """Extract topology features from cluster relationships."""
        features = {'multi_scale_hierarchy': {}}
        
        for fragment_name, clusters in self.clusters_by_fragment.items():
            # Group by scale
            by_scale = {}
            for cluster in clusters:
                scale = cluster['scale']
                if scale not in by_scale:
                    by_scale[scale] = []
                by_scale[scale].append(cluster)
            
            # Find parent-child relationships
            hierarchy = []
            scales = sorted(by_scale.keys())
            
            for i in range(len(scales) - 1):
                for small in by_scale[scales[i]]:
                    for large in by_scale[scales[i + 1]]:
                        dist = np.linalg.norm(
                            np.array(small['barycenter']) - np.array(large['barycenter'])
                        )
                        if dist < large['scale'] / 2:
                            hierarchy.append({
                                'parent': large['cluster_id'],
                                'child': small['cluster_id'],
                                'scale_ratio': scales[i + 1] / scales[i]
                            })
            
            features['multi_scale_hierarchy'][fragment_name] = hierarchy
        
        return features
    
    def _save_assembly_knowledge(self, cluster_matches, assembly_graph, topology_features):
        """Save all results."""
        output_file = self.output_dir / "cluster_assembly_with_gt.h5"
        
        with h5py.File(output_file, 'w') as f:
            # Save matches
            matches_group = f.create_group('cluster_matches')
            
            if cluster_matches:
                # Convert to arrays
                data = {
                    'cluster_id_1': [m.cluster_id_1 for m in cluster_matches],
                    'cluster_id_2': [m.cluster_id_2 for m in cluster_matches],
                    'fragment_1': [m.fragment_1.encode('utf8') for m in cluster_matches],
                    'fragment_2': [m.fragment_2.encode('utf8') for m in cluster_matches],
                    'distances': [m.distance for m in cluster_matches],
                    'normal_similarities': [m.normal_similarity for m in cluster_matches],
                    'is_ground_truth': [m.is_ground_truth for m in cluster_matches],
                    'confidences': [m.confidence for m in cluster_matches],
                    'gt_confidences': [m.gt_confidence for m in cluster_matches]
                }
                
                for key, values in data.items():
                    matches_group.create_dataset(key, data=np.array(values))
            
            # Save metadata
            metadata = f.create_group('metadata')
            metadata.attrs['n_matches'] = len(cluster_matches)
            metadata.attrs['n_gt_matches'] = sum(1 for m in cluster_matches if m.is_ground_truth)
            metadata.attrs['contact_threshold'] = self.contact_threshold
            metadata.attrs['using_positioned_fragments'] = self.using_positioned
            metadata.attrs['has_cluster_level_gt'] = 'cluster_ground_truth_matches' in self.ground_truth
        
        logger.info(f"Saved assembly knowledge to {output_file}")
        
        # Save summary report
        self._save_summary_report(cluster_matches, assembly_graph)
    
    def _save_summary_report(self, cluster_matches, assembly_graph):
        """Save human-readable summary."""
        report_file = self.output_dir / "assembly_with_cluster_gt_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("ASSEMBLY EXTRACTION WITH CLUSTER-LEVEL GT REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall statistics
            f.write("SUMMARY:\n")
            f.write(f"Total matches: {len(cluster_matches)}\n")
            
            gt_matches = [m for m in cluster_matches if m.is_ground_truth]
            f.write(f"Ground truth matches: {len(gt_matches)}\n")
            if len(cluster_matches) > 0:
                f.write(f"Ground truth percentage: {len(gt_matches)/len(cluster_matches)*100:.1f}%\n\n")
            else:
                f.write("Ground truth percentage: N/A (no matches found)\n\n")
            
            # GT source info
            if 'cluster_ground_truth_matches' in self.ground_truth:
                f.write("Using cluster-level ground truth\n")
                f.write(f"Total GT matches in file: {len(self.ground_truth['cluster_ground_truth_matches'])}\n\n")
            else:
                f.write("Using fragment-level ground truth (no cluster GT available)\n\n")
            
            # Matches by fragment pair
            f.write("MATCHES BY FRAGMENT PAIR:\n")
            f.write("-"*80 + "\n")
            pair_matches = {}
            for match in cluster_matches:
                pair = tuple(sorted([match.fragment_1, match.fragment_2]))
                if pair not in pair_matches:
                    pair_matches[pair] = {'total': 0, 'gt': 0, 'matches': []}
                pair_matches[pair]['total'] += 1
                if match.is_ground_truth:
                    pair_matches[pair]['gt'] += 1
                pair_matches[pair]['matches'].append(match)
            
            for pair, stats in sorted(pair_matches.items()):
                f.write(f"\n{pair[0]} <-> {pair[1]}: {stats['total']} matches ({stats['gt']} GT)\n")
                
                # Show top GT matches for this pair
                gt_matches_pair = sorted([m for m in stats['matches'] if m.is_ground_truth], 
                                       key=lambda m: m.gt_confidence, reverse=True)
                if gt_matches_pair:
                    f.write("  Top GT matches:\n")
                    for m in gt_matches_pair[:3]:
                        f.write(f"    C{m.cluster_id_1} <-> C{m.cluster_id_2}: "
                               f"conf={m.confidence:.3f}, gt_conf={m.gt_confidence:.3f}, "
                               f"dist={m.distance:.1f}mm\n")
            
            # Top ground truth matches overall
            if gt_matches:
                f.write("\n\nTOP GROUND TRUTH MATCHES OVERALL:\n")
                f.write("-"*80 + "\n")
                for i, match in enumerate(sorted(gt_matches, key=lambda m: m.gt_confidence, reverse=True)[:15]):
                    f.write(f"\n{i+1}. {match.fragment_1}:C{match.cluster_id_1} <-> "
                           f"{match.fragment_2}:C{match.cluster_id_2}\n")
                    f.write(f"   Distance: {match.distance:.2f}mm\n")
                    f.write(f"   Normal similarity: {match.normal_similarity:.3f}\n")
                    f.write(f"   Match confidence: {match.confidence:.3f}\n")
                    f.write(f"   GT confidence: {match.gt_confidence:.3f}\n")
            
            # Graph statistics
            f.write("\n\nGRAPH STATISTICS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Nodes: {assembly_graph.number_of_nodes()}\n")
            f.write(f"Edges: {assembly_graph.number_of_edges()}\n")
            
            intra_edges = [e for e in assembly_graph.edges(data=True) if e[2].get('type') == 'intra_fragment']
            inter_edges = [e for e in assembly_graph.edges(data=True) if e[2].get('type') == 'inter_fragment']
            gt_edges = [e for e in inter_edges if e[2].get('is_ground_truth', False)]
            
            f.write(f"Intra-fragment edges: {len(intra_edges)}\n")
            f.write(f"Inter-fragment edges: {len(inter_edges)}\n")
            f.write(f"Ground truth edges: {len(gt_edges)}\n")
        
        logger.info(f"Saved report to {report_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Assembly Extraction with Cluster GT")
    parser.add_argument("--clusters", default="output/feature_clusters_fixed.pkl",
                       help="Path to cluster file")
    parser.add_argument("--segments", default="output/segmented_fragments.pkl",
                       help="Path to segmentation file")
    parser.add_argument("--ground_truth", default="Ground_Truth/cluster_level_ground_truth.json",
                       help="Path to ground truth file")
    parser.add_argument("--ply_dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with PLY files")
    parser.add_argument("--output_dir", default="output",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Run extraction
    extractor = EnhancedAssemblyExtractor(
        clusters_file=args.clusters,
        segments_file=args.segments,
        ground_truth_file=args.ground_truth,
        ply_dir=args.ply_dir,
        output_dir=args.output_dir
    )
    
    cluster_matches, assembly_graph = extractor.extract_assembly_knowledge()
    
    logger.info("\nEnhanced assembly extraction complete!")


if __name__ == "__main__":
    main()