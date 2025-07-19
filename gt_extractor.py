#!/usr/bin/env python3
"""
Advanced Cluster-Level Ground Truth Extractor
Extracts ground truth at the cluster level by analyzing which clusters 
from different fragments are actually in contact in the assembled state.
"""

import numpy as np
import open3d as o3d
import json
import pickle
import h5py
from pathlib import Path
from scipy.spatial import cKDTree
import logging
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Set, Optional
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ClusterGroundTruth:
    """Ground truth match between two clusters."""
    fragment_1: str
    fragment_2: str
    cluster_id_1: int
    cluster_id_2: int
    contact_points: int
    mean_distance: float
    contact_area: float
    confidence: float
    cluster_center_1: np.ndarray
    cluster_center_2: np.ndarray
    is_primary_contact: bool

class AdvancedClusterGTExtractor:
    def __init__(self,
                 positioned_dir: str = "Ground_Truth/reconstructed/artifact_1",
                 clusters_file: str = "output/feature_clusters_fixed.pkl",
                 segments_file: str = "output/segmented_fragments.pkl",
                 output_dir: str = "Ground_Truth",
                 contact_threshold: float = 3.0,
                 cluster_contact_threshold: float = 10.0):
        
        self.positioned_dir = Path(positioned_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.contact_threshold = contact_threshold  # For point-level contacts
        self.cluster_contact_threshold = cluster_contact_threshold  # For cluster-level contacts
        
        # Load cluster and segment data
        self._load_cluster_data(clusters_file, segments_file)
        
        # Storage
        self.fragments = {}
        self.fragment_points = {}
        self.cluster_points = {}
        self.cluster_gt_matches = []
        
    def _load_cluster_data(self, clusters_file: str, segments_file: str):
        """Load cluster and segmentation data."""
        logger.info("Loading cluster data...")
        
        with open(clusters_file, 'rb') as f:
            self.cluster_data = pickle.load(f)
        
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Organize clusters by fragment
        self._organize_clusters()
        
    def _organize_clusters(self):
        """Organize clusters by fragment with proper indexing."""
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
                    cluster['local_id'] = i
                    
                    self.cluster_lookup[cluster_idx] = cluster
                    fragment_clusters.append(cluster)
                    cluster_idx += 1
            
            self.clusters_by_fragment[frag_name] = fragment_clusters
            logger.info(f"{frag_name}: {len(fragment_clusters)} clusters")
    
    def extract_cluster_ground_truth(self):
        """Main method to extract cluster-level ground truth."""
        logger.info("Extracting cluster-level ground truth...")
        
        # Step 1: Load positioned fragments with cluster assignments
        self._load_positioned_fragments_with_clusters()
        
        # Step 2: Find fragment-level contacts (as before)
        contact_pairs = self._find_fragment_contacts()
        
        # Step 3: For each contact pair, find cluster-level matches
        for frag1, frag2 in tqdm(contact_pairs, desc="Finding cluster matches"):
            cluster_matches = self._find_cluster_matches(frag1, frag2)
            self.cluster_gt_matches.extend(cluster_matches)
        
        # Step 4: Compile and save enhanced ground truth
        ground_truth = self._compile_enhanced_ground_truth(contact_pairs)
        
        # Save results
        self._save_ground_truth(ground_truth)
        
        # Print detailed summary
        self._print_detailed_summary(ground_truth)
        
        return ground_truth
    
    def _load_positioned_fragments_with_clusters(self):
        """Load fragments and map points to clusters."""
        ply_files = sorted(self.positioned_dir.glob("*.ply"))
        
        logger.info(f"Loading {len(ply_files)} positioned fragments with cluster assignments...")
        
        for ply_file in tqdm(ply_files, desc="Loading fragments"):
            fragment_name = ply_file.stem
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_file))
            if not pcd.has_points():
                continue
            
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Store fragment data
            self.fragments[fragment_name] = {
                'point_cloud': pcd,
                'n_points': len(points),
                'file_path': str(ply_file)
            }
            
            # Get cluster assignments for this fragment
            if fragment_name in self.segment_data:
                segment_indices = self.segment_data[fragment_name].get('segment_indices', [])
                
                # Map each point to its cluster
                point_to_cluster = np.full(len(points), -1, dtype=int)
                
                # Get clusters for this fragment
                fragment_clusters = self.clusters_by_fragment.get(fragment_name, [])
                
                # For each cluster, find its points
                cluster_points_dict = {}
                
                for cluster in fragment_clusters:
                    cluster_id = cluster['local_id']
                    # Find points belonging to this cluster
                    cluster_mask = (segment_indices == cluster_id)
                    cluster_point_indices = np.where(cluster_mask)[0]
                    
                    if len(cluster_point_indices) > 0:
                        point_to_cluster[cluster_point_indices] = cluster_id
                        cluster_points_dict[cluster_id] = {
                            'indices': cluster_point_indices,
                            'points': points[cluster_point_indices],
                            'center': np.mean(points[cluster_point_indices], axis=0)
                        }
                
                self.fragment_points[fragment_name] = {
                    'points': points,
                    'colors': colors,
                    'point_to_cluster': point_to_cluster,
                    'cluster_points': cluster_points_dict
                }
                
                logger.info(f"  {fragment_name}: {len(cluster_points_dict)} clusters with points")
    
    def _find_fragment_contacts(self):
        """Find which fragments are in contact."""
        logger.info("Finding fragment contacts...")
        
        fragment_names = sorted(self.fragments.keys())
        contact_pairs = []
        
        # Check all pairs
        for i in range(len(fragment_names)):
            for j in range(i + 1, len(fragment_names)):
                frag1 = fragment_names[i]
                frag2 = fragment_names[j]
                
                if self._check_fragment_contact(frag1, frag2):
                    contact_pairs.append((frag1, frag2))
        
        logger.info(f"Found {len(contact_pairs)} fragment contact pairs")
        return contact_pairs
    
    def _check_fragment_contact(self, frag1: str, frag2: str) -> bool:
        """Quick check if two fragments are in contact."""
        points1 = self.fragment_points[frag1]['points']
        points2 = self.fragment_points[frag2]['points']
        
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
    
    def _find_cluster_matches(self, frag1: str, frag2: str) -> List[ClusterGroundTruth]:
        """Find cluster-level matches between two fragments."""
        matches = []
        
        cluster_points1 = self.fragment_points[frag1]['cluster_points']
        cluster_points2 = self.fragment_points[frag2]['cluster_points']
        
        # For each cluster in fragment 1
        for cluster_id1, cluster_data1 in cluster_points1.items():
            points1 = cluster_data1['points']
            center1 = cluster_data1['center']
            
            # Build KD-tree for efficiency
            tree1 = cKDTree(points1)
            
            # Check against each cluster in fragment 2
            for cluster_id2, cluster_data2 in cluster_points2.items():
                points2 = cluster_data2['points']
                center2 = cluster_data2['center']
                
                # Quick distance check between cluster centers
                center_dist = np.linalg.norm(center1 - center2)
                if center_dist > self.cluster_contact_threshold * 2:
                    continue
                
                # Detailed contact analysis
                contact_info = self._analyze_cluster_contact(
                    cluster_data1, cluster_data2, 
                    cluster_id1, cluster_id2,
                    frag1, frag2
                )
                
                if contact_info is not None:
                    matches.append(contact_info)
        
        # Sort by confidence and mark primary contacts
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Mark top matches as primary contacts
        if matches:
            # Group by cluster to avoid marking multiple matches for same cluster
            cluster1_matched = set()
            cluster2_matched = set()
            
            for match in matches:
                if (match.cluster_id_1 not in cluster1_matched and 
                    match.cluster_id_2 not in cluster2_matched):
                    match.is_primary_contact = True
                    cluster1_matched.add(match.cluster_id_1)
                    cluster2_matched.add(match.cluster_id_2)
        
        return matches
    
    def _analyze_cluster_contact(self, cluster_data1: Dict, cluster_data2: Dict,
                                cluster_id1: int, cluster_id2: int,
                                frag1: str, frag2: str) -> Optional[ClusterGroundTruth]:
        """Analyze contact between two clusters."""
        points1 = cluster_data1['points']
        points2 = cluster_data2['points']
        
        # Build KD-trees
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Find contact points
        contact_mask1 = np.zeros(len(points1), dtype=bool)
        contact_mask2 = np.zeros(len(points2), dtype=bool)
        distances = []
        
        # Check points in cluster 1
        for i, point in enumerate(points1):
            dist, _ = tree2.query(point)
            if dist < self.contact_threshold:
                contact_mask1[i] = True
                distances.append(dist)
        
        # Check points in cluster 2
        for i, point in enumerate(points2):
            dist, _ = tree1.query(point)
            if dist < self.contact_threshold:
                contact_mask2[i] = True
                distances.append(dist)
        
        # Need minimum contact points
        total_contact_points = np.sum(contact_mask1) + np.sum(contact_mask2)
        if total_contact_points < 10:  # Minimum threshold
            return None
        
        # Calculate contact metrics
        contact_points1 = points1[contact_mask1]
        contact_points2 = points2[contact_mask2]
        
        # Contact centers
        if len(contact_points1) > 0:
            contact_center1 = np.mean(contact_points1, axis=0)
        else:
            contact_center1 = cluster_data1['center']
        
        if len(contact_points2) > 0:
            contact_center2 = np.mean(contact_points2, axis=0)
        else:
            contact_center2 = cluster_data2['center']
        
        # Estimate contact area
        all_contact_points = np.vstack([contact_points1, contact_points2]) if len(contact_points1) > 0 and len(contact_points2) > 0 else contact_points1 if len(contact_points1) > 0 else contact_points2
        
        if len(all_contact_points) > 3:
            # Use PCA for contact area estimation
            centered = all_contact_points - np.mean(all_contact_points, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            # Approximate area as ellipse
            contact_area = np.pi * np.sqrt(max(eigenvalues[1], 0)) * np.sqrt(max(eigenvalues[2], 0))
        else:
            contact_area = 0.0
        
        # Calculate confidence based on multiple factors
        contact_ratio = total_contact_points / min(len(points1), len(points2))
        mean_distance = np.mean(distances) if distances else 0.0
        
        # Confidence score
        distance_score = np.exp(-mean_distance / self.contact_threshold)
        size_score = min(total_contact_points / 100.0, 1.0)  # Normalize by expected contact size
        confidence = 0.5 * distance_score + 0.3 * size_score + 0.2 * contact_ratio
        
        return ClusterGroundTruth(
            fragment_1=frag1,
            fragment_2=frag2,
            cluster_id_1=cluster_id1,
            cluster_id_2=cluster_id2,
            contact_points=int(total_contact_points),
            mean_distance=float(mean_distance),
            contact_area=float(contact_area),
            confidence=float(confidence),
            cluster_center_1=contact_center1,
            cluster_center_2=contact_center2,
            is_primary_contact=False  # Will be set later
        )
    
    def _compile_enhanced_ground_truth(self, contact_pairs):
        """Compile enhanced ground truth with cluster-level information."""
        # Fragment information (as before)
        fragments = {}
        for frag_name, frag_data in self.fragments.items():
            pcd = frag_data['point_cloud']
            bbox = pcd.get_axis_aligned_bounding_box()
            
            fragments[frag_name] = {
                'transform_matrix': np.eye(4).tolist(),
                'vertex_count': frag_data['n_points'],
                'bbox': {
                    'min': bbox.min_bound.tolist(),
                    'max': bbox.max_bound.tolist()
                },
                'file_path': frag_data['file_path']
            }
        
        # Cluster-level ground truth matches
        cluster_matches = []
        for match in self.cluster_gt_matches:
            match_dict = {
                'fragment_1': match.fragment_1,
                'fragment_2': match.fragment_2,
                'cluster_id_1': match.cluster_id_1,
                'cluster_id_2': match.cluster_id_2,
                'contact_points': match.contact_points,
                'mean_distance': match.mean_distance,
                'contact_area': match.contact_area,
                'confidence': match.confidence,
                'cluster_center_1': match.cluster_center_1.tolist(),
                'cluster_center_2': match.cluster_center_2.tolist(),
                'is_primary_contact': match.is_primary_contact
            }
            cluster_matches.append(match_dict)
        
        # Cluster statistics
        cluster_stats = {}
        for frag_name, clusters in self.clusters_by_fragment.items():
            cluster_stats[frag_name] = {
                'n_clusters': len(clusters),
                'clusters_with_matches': len(set(
                    m.cluster_id_1 if m.fragment_1 == frag_name else m.cluster_id_2 
                    for m in self.cluster_gt_matches 
                    if m.fragment_1 == frag_name or m.fragment_2 == frag_name
                ))
            }
        
        return {
            'fragments': fragments,
            'contact_pairs': [(f1, f2) for f1, f2 in contact_pairs],
            'cluster_ground_truth_matches': cluster_matches,
            'cluster_statistics': cluster_stats,
            'extraction_info': {
                'source_directory': str(self.positioned_dir),
                'contact_threshold': self.contact_threshold,
                'cluster_contact_threshold': self.cluster_contact_threshold,
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_cluster_matches': len(self.cluster_gt_matches),
                'primary_matches': sum(1 for m in self.cluster_gt_matches if m.is_primary_contact)
            }
        }
    
    def _save_ground_truth(self, ground_truth):
        """Save ground truth in multiple formats."""
        # Save as JSON
        json_file = self.output_dir / "cluster_level_ground_truth.json"
        with open(json_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        logger.info(f"Saved JSON ground truth to {json_file}")
        
        # Save as HDF5 for efficient loading
        h5_file = self.output_dir / "cluster_level_ground_truth.h5"
        with h5py.File(h5_file, 'w') as f:
            # Save cluster matches
            if self.cluster_gt_matches:
                matches_group = f.create_group('cluster_matches')
                
                # Convert to arrays
                data_arrays = {
                    'fragment_1': [m.fragment_1.encode('utf8') for m in self.cluster_gt_matches],
                    'fragment_2': [m.fragment_2.encode('utf8') for m in self.cluster_gt_matches],
                    'cluster_id_1': [m.cluster_id_1 for m in self.cluster_gt_matches],
                    'cluster_id_2': [m.cluster_id_2 for m in self.cluster_gt_matches],
                    'contact_points': [m.contact_points for m in self.cluster_gt_matches],
                    'mean_distances': [m.mean_distance for m in self.cluster_gt_matches],
                    'contact_areas': [m.contact_area for m in self.cluster_gt_matches],
                    'confidences': [m.confidence for m in self.cluster_gt_matches],
                    'is_primary': [m.is_primary_contact for m in self.cluster_gt_matches]
                }
                
                for key, values in data_arrays.items():
                    matches_group.create_dataset(key, data=np.array(values))
                
                # Save cluster centers
                centers_1 = np.array([m.cluster_center_1 for m in self.cluster_gt_matches])
                centers_2 = np.array([m.cluster_center_2 for m in self.cluster_gt_matches])
                matches_group.create_dataset('cluster_centers_1', data=centers_1)
                matches_group.create_dataset('cluster_centers_2', data=centers_2)
            
            # Save metadata
            metadata = f.create_group('metadata')
            for key, value in ground_truth['extraction_info'].items():
                metadata.attrs[key] = value
        
        logger.info(f"Saved HDF5 ground truth to {h5_file}")
    
    def _print_detailed_summary(self, ground_truth):
        """Print detailed summary of cluster-level ground truth."""
        print("\n" + "="*80)
        print("CLUSTER-LEVEL GROUND TRUTH EXTRACTION SUMMARY")
        print("="*80)
        
        print(f"\nFragments: {len(ground_truth['fragments'])}")
        print(f"Contact pairs: {len(ground_truth['contact_pairs'])}")
        print(f"Total cluster matches: {len(self.cluster_gt_matches)}")
        print(f"Primary matches: {sum(1 for m in self.cluster_gt_matches if m.is_primary_contact)}")
        
        print("\nCluster Statistics by Fragment:")
        for frag, stats in ground_truth['cluster_statistics'].items():
            print(f"  {frag}: {stats['n_clusters']} clusters, "
                  f"{stats['clusters_with_matches']} with matches "
                  f"({stats['clusters_with_matches']/stats['n_clusters']*100:.1f}%)")
        
        print("\nTop Cluster Matches by Confidence:")
        print("-"*80)
        
        # Sort by confidence
        sorted_matches = sorted(self.cluster_gt_matches, key=lambda m: m.confidence, reverse=True)
        
        for i, match in enumerate(sorted_matches[:10]):
            print(f"\n{i+1}. {match.fragment_1}:C{match.cluster_id_1} <-> "
                  f"{match.fragment_2}:C{match.cluster_id_2}")
            print(f"   Contact points: {match.contact_points}")
            print(f"   Mean distance: {match.mean_distance:.3f}mm")
            print(f"   Contact area: {match.contact_area:.1f}mm²")
            print(f"   Confidence: {match.confidence:.3f}")
            print(f"   Primary contact: {match.is_primary_contact}")
        
        # Contact pair summary
        print("\n\nCluster Matches by Fragment Pair:")
        print("-"*80)
        
        pair_stats = {}
        for match in self.cluster_gt_matches:
            pair = (match.fragment_1, match.fragment_2)
            if pair not in pair_stats:
                pair_stats[pair] = {'total': 0, 'primary': 0}
            pair_stats[pair]['total'] += 1
            if match.is_primary_contact:
                pair_stats[pair]['primary'] += 1
        
        for pair, stats in sorted(pair_stats.items()):
            print(f"{pair[0]} <-> {pair[1]}: {stats['total']} matches "
                  f"({stats['primary']} primary)")
    
    def visualize_cluster_match(self, match: ClusterGroundTruth):
        """Visualize a specific cluster match."""
        # Load point clouds
        pcd1 = self.fragments[match.fragment_1]['point_cloud']
        pcd2 = self.fragments[match.fragment_2]['point_cloud']
        
        # Get cluster points
        cluster_points1 = self.fragment_points[match.fragment_1]['cluster_points'][match.cluster_id_1]
        cluster_points2 = self.fragment_points[match.fragment_2]['cluster_points'][match.cluster_id_2]
        
        # Create colored point clouds
        pcd1_viz = o3d.geometry.PointCloud(pcd1)
        pcd2_viz = o3d.geometry.PointCloud(pcd2)
        
        # Color fragments differently
        colors1 = np.array([[0.7, 0.7, 0.7]] * len(pcd1.points))
        colors2 = np.array([[0.5, 0.5, 0.5]] * len(pcd2.points))
        
        # Highlight matched clusters
        colors1[cluster_points1['indices']] = [1.0, 0.2, 0.2]  # Red
        colors2[cluster_points2['indices']] = [0.2, 1.0, 0.2]  # Green
        
        pcd1_viz.colors = o3d.utility.Vector3dVector(colors1)
        pcd2_viz.colors = o3d.utility.Vector3dVector(colors2)
        
        # Add cluster centers as spheres
        sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
        sphere1.translate(match.cluster_center_1)
        sphere1.paint_uniform_color([1.0, 0.0, 0.0])
        
        sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
        sphere2.translate(match.cluster_center_2)
        sphere2.paint_uniform_color([0.0, 1.0, 0.0])
        
        # Visualize
        o3d.visualization.draw_geometries(
            [pcd1_viz, pcd2_viz, sphere1, sphere2],
            window_name=f"Cluster Match: {match.fragment_1}:C{match.cluster_id_1} <-> "
                       f"{match.fragment_2}:C{match.cluster_id_2} (Conf: {match.confidence:.3f})"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract cluster-level ground truth")
    parser.add_argument("--positioned-dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with positioned fragments")
    parser.add_argument("--clusters", default="output/feature_clusters_fixed.pkl",
                       help="Path to cluster file")
    parser.add_argument("--segments", default="output/segmented_fragments.pkl",
                       help="Path to segmentation file")
    parser.add_argument("--output-dir", default="Ground_Truth",
                       help="Output directory")
    parser.add_argument("--contact-threshold", type=float, default=3.0,
                       help="Point contact threshold in mm")
    parser.add_argument("--cluster-threshold", type=float, default=10.0,
                       help="Cluster contact threshold in mm")
    parser.add_argument("--visualize", type=int, default=0,
                       help="Number of top matches to visualize")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = AdvancedClusterGTExtractor(
        positioned_dir=args.positioned_dir,
        clusters_file=args.clusters,
        segments_file=args.segments,
        output_dir=args.output_dir,
        contact_threshold=args.contact_threshold,
        cluster_contact_threshold=args.cluster_threshold
    )
    
    # Extract ground truth
    ground_truth = extractor.extract_cluster_ground_truth()
    
    # Visualize top matches if requested
    if args.visualize > 0 and extractor.cluster_gt_matches:
        sorted_matches = sorted(extractor.cluster_gt_matches, 
                              key=lambda m: m.confidence, reverse=True)
        for i, match in enumerate(sorted_matches[:args.visualize]):
            print(f"\nVisualizing match {i+1}/{args.visualize}")
            extractor.visualize_cluster_match(match)


if __name__ == "__main__":
    main()