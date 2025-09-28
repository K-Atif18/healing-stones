#!/usr/bin/env python3
"""
Multi-Scale Cluster-Level Ground Truth Extractor
Works with unified clustering pipeline and handles partial overlapping contacts
across different scales (1k, 5k, 10k)

MODIFICATION: Improved fragment contact detection + radius-based cluster matching
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

# Copy of SurfaceCluster class for pickle compatibility
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
class ScaleClusterGroundTruth:
    """Ground truth match between two clusters at a specific scale."""
    scale: str  # "1k", "5k", "10k"
    fragment_1: str
    fragment_2: str
    cluster_id_1: str  # Full cluster ID like "frag_1_c1k_001"
    cluster_id_2: str  # Full cluster ID like "frag_2_c1k_003"
    local_id_1: int   # Local cluster ID within fragment and scale
    local_id_2: int   # Local cluster ID within fragment and scale
    contact_points: int
    overlap_ratio_1: float  # How much of cluster 1 is in contact (0.0 to 1.0)
    overlap_ratio_2: float  # How much of cluster 2 is in contact (0.0 to 1.0)
    mean_distance: float
    contact_area: float
    confidence: float
    cluster_center_1: np.ndarray
    cluster_center_2: np.ndarray
    contact_type: str  # "full", "partial", "minimal"
    is_primary_contact: bool

class MultiScaleClusterGTExtractor:
    def __init__(self,
                 positioned_dir: str = "Ground_Truth/reconstructed/artifact_1",
                 clusters_file: str = "output/feature_clusters.pkl",
                 segments_file: str = "output/segmented_fragments.pkl",
                 output_dir: str = "Ground_Truth",
                 contact_threshold: float = 2.0,  # Point-to-point contact distance
                 max_cluster_distance: float = 30.0,  # Max distance between cluster barycenters
                 scales: List[str] = ["1k", "5k", "10k"]):
        
        self.positioned_dir = Path(positioned_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.contact_threshold = contact_threshold  # For actual point-to-point contacts (2mm)
        self.max_cluster_distance = max_cluster_distance  # Max barycenter distance for consideration
        self.scales = scales
        
        # Load cluster and segment data
        self._load_unified_cluster_data(clusters_file, segments_file)
        
        # Storage
        self.fragments = {}
        self.fragment_points = {}
        self.cluster_gt_matches = {scale: [] for scale in self.scales}
        self.contact_pairs = []  # Store contact pairs for JSON output
        
    def _load_unified_cluster_data(self, clusters_file: str, segments_file: str):
        """Load hierarchical cluster data from unified pipeline with fallback."""
        logger.info("Loading unified cluster data...")
        
        # Try to load hierarchical clusters
        try:
            with open(clusters_file, 'rb') as f:
                self.hierarchical_clusters = pickle.load(f)
            logger.info("Successfully loaded hierarchical cluster data")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Failed to load hierarchical clusters: {e}")
            
            # Try flat format as fallback
            flat_file = Path(clusters_file).parent / "feature_clusters_flat.pkl"
            if flat_file.exists():
                logger.info("Trying flat cluster format...")
                self._load_flat_and_convert(flat_file)
            else:
                raise FileNotFoundError(f"Neither hierarchical nor flat cluster file found!")
        
        # Load segmentation data
        try:
            with open(segments_file, 'rb') as f:
                self.segment_data = pickle.load(f)
            logger.info("Successfully loaded segmentation data")
        except Exception as e:
            logger.warning(f"Failed to load segmentation data: {e}")
            self.segment_data = {}
        
        logger.info(f"Loaded hierarchical clusters for {len(self.hierarchical_clusters)} fragments")
        
        # Print structure for verification
        for frag_name, scales in self.hierarchical_clusters.items():
            for scale_name, clusters in scales.items():
                logger.info(f"  {frag_name} - {scale_name}: {len(clusters)} clusters")
    
    def _load_flat_and_convert(self, flat_file: Path):
        """Load flat cluster data and convert to hierarchical."""
        with open(flat_file, 'rb') as f:
            flat_data = pickle.load(f)
        
        if isinstance(flat_data, dict) and 'clusters' in flat_data:
            clusters = flat_data['clusters']
        else:
            clusters = flat_data
        
        # Convert to hierarchical structure
        self.hierarchical_clusters = {}
        
        for cluster_data in clusters:
            # Handle both object and dict formats
            if hasattr(cluster_data, 'fragment_name'):
                # Object format
                fragment = cluster_data.fragment_name
                scale = cluster_data.scale_name
                cluster_dict = {
                    'cluster_id': cluster_data.cluster_id,
                    'barycenter': cluster_data.barycenter,
                    'point_count': cluster_data.point_count,
                    'point_indices': cluster_data.point_indices,
                    'original_point_indices': getattr(cluster_data, 'original_point_indices', []),
                    'eigenvalues': cluster_data.eigenvalues,
                    'size_signature': cluster_data.size_signature,
                    'anisotropy_signature': cluster_data.anisotropy_signature,
                    'local_id': cluster_data.local_id
                }
            else:
                # Dict format
                fragment = cluster_data.get('fragment', 'unknown')
                scale = cluster_data.get('scale', '1k')
                cluster_dict = cluster_data
            
            if fragment not in self.hierarchical_clusters:
                self.hierarchical_clusters[fragment] = {}
            if scale not in self.hierarchical_clusters[fragment]:
                self.hierarchical_clusters[fragment][scale] = []
            
            self.hierarchical_clusters[fragment][scale].append(cluster_dict)
        
        logger.info("Successfully converted flat clusters to hierarchical structure")
    
    def extract_multi_scale_ground_truth(self):
        """Main method to extract multi-scale cluster-level ground truth."""
        logger.info("Extracting multi-scale cluster-level ground truth...")
        
        # Step 1: Load positioned fragments with cluster assignments
        self._load_positioned_fragments_with_clusters()
        
        # Step 2: Find fragment-level contacts and store them (IMPROVED VERSION)
        self.contact_pairs = self._find_fragment_contacts_improved()
        
        # Step 3: For each scale, find cluster-level matches using radius-based approach
        for scale in self.scales:
            logger.info(f"\nProcessing scale: {scale}")
            
            for frag1, frag2 in tqdm(self.contact_pairs, desc=f"Finding {scale} cluster matches"):
                cluster_matches = self._find_scale_cluster_matches_proximity_based(frag1, frag2, scale)
                self.cluster_gt_matches[scale].extend(cluster_matches)
            
            # Sort by confidence and mark primary contacts for this scale
            self._mark_primary_contacts(scale)
            
            logger.info(f"Found {len(self.cluster_gt_matches[scale])} matches for scale {scale}")
        
        # Step 4: Compile and save enhanced ground truth
        ground_truth = self._compile_multi_scale_ground_truth()
        
        # Save results
        self._save_ground_truth(ground_truth)
        
        # Print detailed summary
        self._print_detailed_summary(ground_truth)
        
        return ground_truth
    
    def _load_positioned_fragments_with_clusters(self):
        """Load fragments and map points to clusters for all scales."""
        ply_files = sorted(self.positioned_dir.glob("*.ply"))
        
        logger.info(f"Loading {len(ply_files)} positioned fragments with multi-scale cluster assignments...")
        
        for ply_file in tqdm(ply_files, desc="Loading fragments"):
            fragment_name = ply_file.stem
            
            # Skip if not in our cluster data
            if fragment_name not in self.hierarchical_clusters:
                logger.warning(f"Fragment {fragment_name} not found in cluster data")
                continue
            
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
            
            # Get segment indices
            segment_indices = None
            if fragment_name in self.segment_data:
                segment_indices = self.segment_data[fragment_name].get('segment_indices', None)
            
            # Create cluster point mappings for each scale
            fragment_cluster_data = {'points': points, 'colors': colors}
            
            for scale in self.scales:
                scale_clusters = self.hierarchical_clusters[fragment_name].get(scale, [])
                cluster_points_dict = {}
                
                for i, cluster in enumerate(scale_clusters):
                    # Handle both dict and object formats
                    if isinstance(cluster, dict):
                        cluster_id = cluster.get('cluster_id', f"{fragment_name}_c{scale}_{i+1:03d}")
                        point_indices = cluster.get('point_indices', [])
                        original_point_indices = cluster.get('original_point_indices', [])
                        barycenter = np.array(cluster.get('barycenter', [0, 0, 0]))
                    else:
                        cluster_id = getattr(cluster, 'cluster_id', f"{fragment_name}_c{scale}_{i+1:03d}")
                        point_indices = getattr(cluster, 'point_indices', [])
                        original_point_indices = getattr(cluster, 'original_point_indices', [])
                        barycenter = np.array(getattr(cluster, 'barycenter', [0, 0, 0]))
                    
                    # Use original_point_indices if available (maps to full point cloud)
                    if original_point_indices is not None and len(original_point_indices) > 0:
                        valid_indices = [idx for idx in original_point_indices if 0 <= idx < len(points)]
                    else:
                        # Fallback: try to map from break surface indices
                        valid_indices = []
                        if segment_indices is not None:
                            # Find break surface points first
                            if colors is not None:
                                green_mask = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.4) & (colors[:, 2] < 0.4)
                                break_indices = np.where(green_mask)[0]
                            else:
                                break_indices = np.arange(len(points) // 3)
                            
                            # Map cluster point indices to original indices
                            for pt_idx in point_indices:
                                if 0 <= pt_idx < len(break_indices):
                                    valid_indices.append(break_indices[pt_idx])
                    
                    if len(valid_indices) > 0:
                        cluster_points_dict[i] = {
                            'cluster_id': cluster_id,
                            'local_id': i,
                            'indices': np.array(valid_indices),
                            'points': points[valid_indices],
                            'center': barycenter,
                            'n_points': len(valid_indices)
                        }
                
                fragment_cluster_data[f'cluster_points_{scale}'] = cluster_points_dict
                logger.debug(f"  {fragment_name} - {scale}: {len(cluster_points_dict)} clusters with points")
            
            self.fragment_points[fragment_name] = fragment_cluster_data
    
    def _find_fragment_contacts_improved(self):
        """IMPROVED: Find which fragments are in contact using better detection from second script."""
        logger.info("Finding fragment contacts (improved method)...")
        
        fragment_names = sorted(self.fragments.keys())
        contact_pairs = []
        
        # Check all pairs
        for i in range(len(fragment_names)):
            for j in range(i + 1, len(fragment_names)):
                frag1 = fragment_names[i]
                frag2 = fragment_names[j]
                
                # Use improved contact detection
                if self._check_fragment_contact_improved(frag1, frag2):
                    contact_pairs.append((frag1, frag2))
        
        logger.info(f"Found {len(contact_pairs)} fragment contact pairs (improved method)")
        
        # PRINT THE CONTACT PAIRS
        print(f"\nGT SCRIPT - CONTACT PAIRS FOUND ({len(contact_pairs)}):")
        for i, (frag1, frag2) in enumerate(contact_pairs):
            print(f"  {i+1}. {frag1} ↔ {frag2}")
        print()
        
        return contact_pairs
    
    def _check_fragment_contact_improved(self, frag1: str, frag2: str) -> bool:
        """IMPROVED: More thorough check if two fragments are in contact - using multiple approaches from second script."""
        if frag1 not in self.fragment_points or frag2 not in self.fragment_points:
            return False
            
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
        
        # More thorough contact check - try multiple samples to avoid missing contacts
        tree2 = cKDTree(points2)
        
        # Method 1: Try 3 different random samples for robustness (from second script)
        for trial in range(3):
            sample_size = min(1000, len(points1))
            sample_indices = np.random.choice(len(points1), sample_size, replace=False)
            sample_points = points1[sample_indices]
            
            distances, _ = tree2.query(sample_points)
            
            if np.min(distances) < self.contact_threshold:
                return True
        
        # Method 2: If random sampling failed, try a systematic approach for edge cases
        # Sample every Nth point for a more deterministic check
        step_size = max(1, len(points1) // 500)  # Sample ~500 points systematically
        systematic_indices = np.arange(0, len(points1), step_size)
        systematic_points = points1[systematic_indices]
        
        distances, _ = tree2.query(systematic_points)
        
        if np.min(distances) < self.contact_threshold:
            return True
        
        # Method 3: Final check - try the reverse direction (points2 against points1)
        # This catches cases where one fragment is much smaller than the other
        tree1 = cKDTree(points1)
        sample_size = min(1000, len(points2))
        sample_indices = np.random.choice(len(points2), sample_size, replace=False)
        sample_points = points2[sample_indices]
        
        distances, _ = tree1.query(sample_points)
        
        return np.min(distances) < self.contact_threshold
    

    def _find_scale_cluster_matches_proximity_based(self, frag1: str, frag2: str, scale: str) -> List[ScaleClusterGroundTruth]:
        """PROXIMITY-BASED: Find cluster matches using strict 2mm proximity."""
        matches = []
        
        cluster_points1 = self.fragment_points[frag1].get(f'cluster_points_{scale}', {})
        cluster_points2 = self.fragment_points[frag2].get(f'cluster_points_{scale}', {})
        
        if not cluster_points1 or not cluster_points2:
            return matches
        
        logger.debug(f"Proximity-based matching: {len(cluster_points1)} vs {len(cluster_points2)} clusters for {frag1}-{frag2} at {scale}")
        
        # For each cluster in fragment 1
        for local_id1, cluster_data1 in cluster_points1.items():
            points1 = cluster_data1['points']
            center1 = cluster_data1['center']
            cluster_id1 = cluster_data1['cluster_id']
            
            # Build KD-tree for cluster 1 points
            tree1 = cKDTree(points1)
            
            # Check against each cluster in fragment 2
            for local_id2, cluster_data2 in cluster_points2.items():
                points2 = cluster_data2['points']
                center2 = cluster_data2['center']
                cluster_id2 = cluster_data2['cluster_id']
                
                # Quick center distance check - but more generous than before
                center_dist = np.linalg.norm(center1 - center2)
                max_reasonable_distance = 50.0  # mm - only to avoid obviously impossible pairs
                if center_dist > max_reasonable_distance:
                    continue
                
                # PRIMARY TEST: Check if ANY points from cluster2 are within 2mm of cluster1
                # This is the key test - direct point-to-point proximity
                if self._check_clusters_have_close_points(points1, points2):
                    
                    # If they have close points, do detailed analysis
                    contact_info = self._analyze_close_clusters(
                        cluster_data1, cluster_data2, 
                        frag1, frag2, scale
                    )
                    
                    if contact_info is not None:
                        matches.append(contact_info)
                        logger.debug(f"Proximity match: {cluster_id1} <-> {cluster_id2} "
                                   f"(conf={contact_info.confidence:.3f})")
        
        logger.info(f"Found {len(matches)} proximity matches for {frag1}-{frag2} at {scale}")
        return matches

    def _check_clusters_have_close_points(self, points1: np.ndarray, points2: np.ndarray) -> bool:
        """Check if any points from the two clusters are within contact_threshold of each other."""
        # Build KD-tree for points1
        tree1 = cKDTree(points1)
        
        # For efficiency, check a sample of points2 against tree1
        max_check_points = min(500, len(points2))  # Don't check too many points for performance
        
        if len(points2) > max_check_points:
            # Sample points2 randomly
            sample_indices = np.random.choice(len(points2), max_check_points, replace=False)
            sample_points2 = points2[sample_indices]
        else:
            sample_points2 = points2
        
        # Check each sample point against all points in cluster1
        for point in sample_points2:
            dist, _ = tree1.query(point)
            if dist < self.contact_threshold:  # 2mm threshold
                return True
        
        # Also check the reverse direction for better coverage
        tree2 = cKDTree(points2)
        max_check_points1 = min(500, len(points1))
        
        if len(points1) > max_check_points1:
            sample_indices1 = np.random.choice(len(points1), max_check_points1, replace=False)
            sample_points1 = points1[sample_indices1]
        else:
            sample_points1 = points1
        
        for point in sample_points1:
            dist, _ = tree2.query(point)
            if dist < self.contact_threshold:  # 2mm threshold
                return True
        
        return False

    def _analyze_close_clusters(self, cluster_data1: Dict, cluster_data2: Dict,
                               frag1: str, frag2: str, scale: str) -> Optional[ScaleClusterGroundTruth]:
        """Analyze contact between two clusters that have been confirmed to have close points."""
        points1 = cluster_data1['points']
        points2 = cluster_data2['points']
        center1 = cluster_data1['center']
        center2 = cluster_data2['center']
        cluster_id1 = cluster_data1['cluster_id']
        cluster_id2 = cluster_data2['cluster_id']
        local_id1 = cluster_data1['local_id']
        local_id2 = cluster_data2['local_id']
        
        # Build KD-trees
        tree1 = cKDTree(points1) 
        tree2 = cKDTree(points2)
        
        # Find all contact points with exact 2mm threshold
        contact_mask1 = np.zeros(len(points1), dtype=bool)
        contact_mask2 = np.zeros(len(points2), dtype=bool)
        distances = []
        
        # Check ALL points in cluster 1 against cluster 2
        for i, point in enumerate(points1):
            dist, _ = tree2.query(point)
            if dist < self.contact_threshold:  # Exact 2mm
                contact_mask1[i] = True
                distances.append(dist)
        
        # Check ALL points in cluster 2 against cluster 1
        for i, point in enumerate(points2):
            dist, _ = tree1.query(point)
            if dist < self.contact_threshold:  # Exact 2mm
                contact_mask2[i] = True
                distances.append(dist)
        
        # Calculate contact metrics
        contact_count1 = np.sum(contact_mask1)
        contact_count2 = np.sum(contact_mask2)
        total_contact_points = contact_count1 + contact_count2
        
        # Minimum contact requirement - at least 3 points must be close
        if total_contact_points < 3:
            return None
        
        # Calculate overlap ratios
        overlap_ratio1 = contact_count1 / len(points1) if len(points1) > 0 else 0.0
        overlap_ratio2 = contact_count2 / len(points2) if len(points2) > 0 else 0.0
        
        # Contact type classification - more conservative
        max_overlap = max(overlap_ratio1, overlap_ratio2)
        if max_overlap > 0.7:
            contact_type = "full"
        elif max_overlap > 0.3:
            contact_type = "partial"
        else:
            contact_type = "minimal"
        
        # Calculate contact centers
        if contact_count1 > 0:
            contact_points1 = points1[contact_mask1]
            contact_center1 = np.mean(contact_points1, axis=0)
        else:
            contact_center1 = center1
        
        if contact_count2 > 0:
            contact_points2 = points2[contact_mask2]
            contact_center2 = np.mean(contact_points2, axis=0)
        else:
            contact_center2 = center2
        
        # Calculate mean distance and contact area
        mean_distance = np.mean(distances) if distances else self.contact_threshold
        
        # Simple contact area estimation
        if distances:
            # Area based on number of contact points and average proximity
            avg_proximity = self.contact_threshold - mean_distance
            contact_area = total_contact_points * (avg_proximity ** 2)
        else:
            contact_area = 0.0
        
        # Confidence calculation - straightforward approach
        # Based on: how many points are in contact, how close they are, overlap ratios
        distance_score = 1.0 - (mean_distance / self.contact_threshold)  # closer = better
        size_score = min(total_contact_points / 20.0, 1.0)  # normalize by reasonable contact size
        overlap_score = (overlap_ratio1 + overlap_ratio2) / 2.0
        
        # Weight the scores
        confidence = 0.4 * distance_score + 0.4 * size_score + 0.2 * overlap_score
        confidence = max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
        
        return ScaleClusterGroundTruth(
            scale=scale,
            fragment_1=frag1,
            fragment_2=frag2,
            cluster_id_1=cluster_id1,
            cluster_id_2=cluster_id2,
            local_id_1=local_id1,
            local_id_2=local_id2,
            contact_points=int(total_contact_points),
            overlap_ratio_1=float(overlap_ratio1),
            overlap_ratio_2=float(overlap_ratio2),
            mean_distance=float(mean_distance),
            contact_area=float(contact_area),
            confidence=float(confidence),
            cluster_center_1=contact_center1,
            cluster_center_2=contact_center2,
            contact_type=contact_type,
            is_primary_contact=False  # Will be set later
        )
    
    def _mark_primary_contacts(self, scale: str):
        """Mark primary contacts for a specific scale."""
        matches = self.cluster_gt_matches[scale]
        
        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        # Track which clusters have been matched to avoid duplicates
        matched_clusters = set()
        
        for match in matches:
            cluster_key1 = f"{match.fragment_1}_{match.local_id_1}"
            cluster_key2 = f"{match.fragment_2}_{match.local_id_2}"
            
            # Mark as primary if neither cluster is already matched and confidence is high
            if (cluster_key1 not in matched_clusters and 
                cluster_key2 not in matched_clusters and
                match.confidence > 0.5):
                match.is_primary_contact = True
                matched_clusters.add(cluster_key1)
                matched_clusters.add(cluster_key2)
    
    def _compile_multi_scale_ground_truth(self):
        """Compile multi-scale ground truth with cluster-level information."""
        # Fragment information
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
        
        # Multi-scale cluster matches
        cluster_matches_by_scale = {}
        for scale in self.scales:
            scale_matches = []
            for match in self.cluster_gt_matches[scale]:
                match_dict = {
                    'fragment_1': match.fragment_1,
                    'fragment_2': match.fragment_2,
                    'cluster_id_1': match.cluster_id_1,
                    'cluster_id_2': match.cluster_id_2,
                    'local_id_1': match.local_id_1,
                    'local_id_2': match.local_id_2,
                    'contact_points': match.contact_points,
                    'overlap_ratio_1': match.overlap_ratio_1,
                    'overlap_ratio_2': match.overlap_ratio_2,
                    'mean_distance': match.mean_distance,
                    'contact_area': match.contact_area,
                    'confidence': match.confidence,
                    'cluster_center_1': match.cluster_center_1.tolist(),
                    'cluster_center_2': match.cluster_center_2.tolist(),
                    'contact_type': match.contact_type,
                    'is_primary_contact': match.is_primary_contact
                }
                scale_matches.append(match_dict)
            cluster_matches_by_scale[scale] = scale_matches
        
        # Multi-scale cluster statistics
        cluster_stats = {}
        for frag_name in self.hierarchical_clusters.keys():
            frag_stats = {}
            for scale in self.scales:
                scale_clusters = self.hierarchical_clusters[frag_name].get(scale, [])
                matches_for_scale = self.cluster_gt_matches[scale]
                
                clusters_with_matches = len(set(
                    m.local_id_1 if m.fragment_1 == frag_name else m.local_id_2 
                    for m in matches_for_scale 
                    if m.fragment_1 == frag_name or m.fragment_2 == frag_name
                ))
                
                frag_stats[scale] = {
                    'n_clusters': len(scale_clusters),
                    'clusters_with_matches': clusters_with_matches,
                    'match_ratio': clusters_with_matches / max(1, len(scale_clusters))
                }
            cluster_stats[frag_name] = frag_stats
        
        # **MODIFICATION: Add fragment contact pairs to JSON output**
        contact_pairs_for_json = []
        for frag1, frag2 in self.contact_pairs:
            # Store as ordered pairs with additional metadata if needed
            contact_pairs_for_json.append({
                'fragment_1': frag1,
                'fragment_2': frag2,
                'contact_detected': True,
                'contact_threshold_used': self.contact_threshold
            })
        
        return {
            'fragments': fragments,
            'contact_pairs': contact_pairs_for_json,  # **MODIFIED: Now includes detailed contact info**
            'fragment_contact_pairs_simple': [(f1, f2) for f1, f2 in self.contact_pairs],  # **ADDED: Simple list format for backward compatibility**
            'cluster_ground_truth_matches_by_scale': cluster_matches_by_scale,
            'cluster_statistics_by_scale': cluster_stats,
            'extraction_info': {
                'source_directory': str(self.positioned_dir),
                'contact_threshold': self.contact_threshold,
                'max_cluster_distance': self.max_cluster_distance,
                'scales_processed': self.scales,
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_contact_pairs': len(self.contact_pairs),
                'total_matches_by_scale': {
                    scale: len(self.cluster_gt_matches[scale]) 
                    for scale in self.scales
                },
                'primary_matches_by_scale': {
                    scale: sum(1 for m in self.cluster_gt_matches[scale] if m.is_primary_contact)
                    for scale in self.scales
                }
            }
        }
    
    def _save_ground_truth(self, ground_truth):
        """Save multi-scale ground truth in multiple formats."""
        # Save as JSON
        json_file = self.output_dir / "multi_scale_cluster_ground_truth.json"
        with open(json_file, 'w') as f:
            json.dump(ground_truth, f, indent=2, default=self._json_serializer)
        logger.info(f"Saved JSON ground truth to {json_file}")
        
        # Save as HDF5 for efficient loading
        h5_file = self.output_dir / "multi_scale_cluster_ground_truth.h5"
        with h5py.File(h5_file, 'w') as f:
            # Save matches for each scale
            for scale in self.scales:
                matches = self.cluster_gt_matches[scale]
                if matches:
                    scale_group = f.create_group(f'cluster_matches_{scale}')
                    
                    # Convert to arrays
                    data_arrays = {
                        'fragment_1': [m.fragment_1.encode('utf8') for m in matches],
                        'fragment_2': [m.fragment_2.encode('utf8') for m in matches],
                        'cluster_id_1': [m.cluster_id_1.encode('utf8') for m in matches],
                        'cluster_id_2': [m.cluster_id_2.encode('utf8') for m in matches],
                        'local_id_1': [m.local_id_1 for m in matches],
                        'local_id_2': [m.local_id_2 for m in matches],
                        'contact_points': [m.contact_points for m in matches],
                        'overlap_ratio_1': [m.overlap_ratio_1 for m in matches],
                        'overlap_ratio_2': [m.overlap_ratio_2 for m in matches],
                        'mean_distances': [m.mean_distance for m in matches],
                        'contact_areas': [m.contact_area for m in matches],
                        'confidences': [m.confidence for m in matches],
                        'contact_types': [m.contact_type.encode('utf8') for m in matches],
                        'is_primary': [m.is_primary_contact for m in matches]
                    }
                    
                    for key, values in data_arrays.items():
                        scale_group.create_dataset(key, data=np.array(values))
                    
                    # Save cluster centers
                    centers_1 = np.array([m.cluster_center_1 for m in matches])
                    centers_2 = np.array([m.cluster_center_2 for m in matches])
                    scale_group.create_dataset('cluster_centers_1', data=centers_1)
                    scale_group.create_dataset('cluster_centers_2', data=centers_2)
            
            # Save contact pairs
            if self.contact_pairs:
                contact_group = f.create_group('contact_pairs')
                fragment_1_list = [pair[0].encode('utf8') for pair in self.contact_pairs]
                fragment_2_list = [pair[1].encode('utf8') for pair in self.contact_pairs]
                contact_group.create_dataset('fragment_1', data=np.array(fragment_1_list))
                contact_group.create_dataset('fragment_2', data=np.array(fragment_2_list))
            
            # Save metadata
            metadata = f.create_group('metadata')
            for key, value in ground_truth['extraction_info'].items():
                if isinstance(value, dict):
                    sub_group = metadata.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.attrs[sub_key] = sub_value
                else:
                    metadata.attrs[key] = value
        
        logger.info(f"Saved HDF5 ground truth to {h5_file}")
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _print_detailed_summary(self, ground_truth):
        """Print detailed summary of multi-scale cluster-level ground truth."""
        print("\n" + "="*80)
        print("MULTI-SCALE CLUSTER-LEVEL GROUND TRUTH EXTRACTION SUMMARY")
        print("="*80)
        
        print(f"\nFragments: {len(ground_truth['fragments'])}")
        print(f"Contact pairs: {len(ground_truth['contact_pairs'])}")
        
        # Print contact pairs for reference
        print(f"\nFragment Contact Pairs:")
        for i, contact_info in enumerate(ground_truth['contact_pairs']):
            frag1 = contact_info['fragment_1']
            frag2 = contact_info['fragment_2']
            print(f"  {i+1}. {frag1} ↔ {frag2}")
        
        # Scale-wise summary
        print(f"\nMatches by Scale:")
        for scale in self.scales:
            total = len(self.cluster_gt_matches[scale])
            primary = sum(1 for m in self.cluster_gt_matches[scale] if m.is_primary_contact)
            print(f"  {scale.upper()}: {total} total matches ({primary} primary)")
        
        # Contact type distribution
        print(f"\nContact Type Distribution:")
        for scale in self.scales:
            matches = self.cluster_gt_matches[scale]
            if matches:
                type_counts = {}
                for match in matches:
                    type_counts[match.contact_type] = type_counts.get(match.contact_type, 0) + 1
                
                print(f"  {scale.upper()}: ", end="")
                for contact_type, count in sorted(type_counts.items()):
                    print(f"{contact_type}: {count}, ", end="")
                print()
        
        # Fragment statistics
        print(f"\nCluster Statistics by Fragment and Scale:")
        for frag, scale_stats in ground_truth['cluster_statistics_by_scale'].items():
            print(f"\n  {frag}:")
            for scale, stats in scale_stats.items():
                match_ratio = stats['match_ratio']
                print(f"    {scale.upper()}: {stats['n_clusters']} clusters, "
                      f"{stats['clusters_with_matches']} with matches "
                      f"({match_ratio:.1%})")
        
        # Top matches for each scale
        print(f"\nTop Matches by Scale (showing overlap ratios):")
        for scale in self.scales:
            matches = self.cluster_gt_matches[scale]
            if matches:
                print(f"\n{scale.upper()} Scale - Top 3 Matches:")
                print("-" * 60)
                
                sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
                
                for i, match in enumerate(sorted_matches[:3]):
                    print(f"{i+1}. {match.cluster_id_1} <-> {match.cluster_id_2}")
                    print(f"   Contact: {match.contact_points} points, {match.contact_type}")
                    print(f"   Overlap ratios: {match.overlap_ratio_1:.2f} | {match.overlap_ratio_2:.2f}")
                    print(f"   Distance: {match.mean_distance:.2f}mm, Area: {match.contact_area:.1f}mm²")
                    print(f"   Confidence: {match.confidence:.3f}, Primary: {match.is_primary_contact}")
                    print()
    
    def visualize_scale_cluster_match(self, match: ScaleClusterGroundTruth):
        """Visualize a specific cluster match with overlap highlighting."""
        # Load point clouds
        pcd1 = self.fragments[match.fragment_1]['point_cloud']
        pcd2 = self.fragments[match.fragment_2]['point_cloud']
        
        # Get cluster points
        cluster_points1 = self.fragment_points[match.fragment_1][f'cluster_points_{match.scale}'][match.local_id_1]
        cluster_points2 = self.fragment_points[match.fragment_2][f'cluster_points_{match.scale}'][match.local_id_2]
        
        # Create colored point clouds
        pcd1_viz = o3d.geometry.PointCloud(pcd1)
        pcd2_viz = o3d.geometry.PointCloud(pcd2)
        
        # Color fragments differently
        colors1 = np.array([[0.7, 0.7, 0.7]] * len(pcd1.points))
        colors2 = np.array([[0.5, 0.5, 0.5]] * len(pcd2.points))
        
        # Highlight matched clusters with intensity based on overlap
        intensity1 = min(1.0, match.overlap_ratio_1 + 0.3)  # Ensure visibility
        intensity2 = min(1.0, match.overlap_ratio_2 + 0.3)
        
        colors1[cluster_points1['indices']] = [1.0 * intensity1, 0.2, 0.2]  # Red with intensity
        colors2[cluster_points2['indices']] = [0.2, 1.0 * intensity2, 0.2]  # Green with intensity
        
        pcd1_viz.colors = o3d.utility.Vector3dVector(colors1)
        pcd2_viz.colors = o3d.utility.Vector3dVector(colors2)
        
        # Add cluster centers as spheres
        sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        sphere1.translate(match.cluster_center_1)
        sphere1.paint_uniform_color([1.0, 0.0, 0.0])
        
        sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
        sphere2.translate(match.cluster_center_2)
        sphere2.paint_uniform_color([0.0, 1.0, 0.0])
        
        # Add line connecting cluster centers
        line_points = [match.cluster_center_1, match.cluster_center_2]
        line_lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_lines)
        line_set.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow line
        
        # Visualize
        window_title = (f"{match.scale.upper()} Scale Match: {match.cluster_id_1} <-> {match.cluster_id_2}\n"
                       f"Contact: {match.contact_type}, Overlaps: {match.overlap_ratio_1:.2f}|{match.overlap_ratio_2:.2f}, "
                       f"Conf: {match.confidence:.3f}")
        
        o3d.visualization.draw_geometries(
            [pcd1_viz, pcd2_viz, sphere1, sphere2, line_set],
            window_name=window_title[:100]  # Truncate for display
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract multi-scale cluster-level ground truth")
    parser.add_argument("--positioned-dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with positioned fragments")
    parser.add_argument("--clusters", default="output/feature_clusters.pkl",
                       help="Path to hierarchical cluster file")
    parser.add_argument("--segments", default="output/segmented_fragments.pkl",
                       help="Path to segmentation file")
    parser.add_argument("--output-dir", default="Ground_Truth",
                       help="Output directory")
    parser.add_argument("--contact-threshold", type=float, default=2.0,
                       help="Point-to-point contact threshold in mm")
    parser.add_argument("--max-cluster-distance", type=float, default=10.0,
                       help="Maximum distance between cluster centers to consider in mm")
    parser.add_argument("--scales", nargs='+', default=["1k", "5k", "10k"],
                       help="Scales to process")
    parser.add_argument("--visualize", type=int, default=0,
                       help="Number of top matches per scale to visualize")
    parser.add_argument("--visualize-scale", default=None,
                       help="Specific scale to visualize (1k, 5k, 10k)")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = MultiScaleClusterGTExtractor(
        positioned_dir=args.positioned_dir,
        clusters_file=args.clusters,
        segments_file=args.segments,
        output_dir=args.output_dir,
        contact_threshold=args.contact_threshold,
        max_cluster_distance=args.max_cluster_distance,
        scales=args.scales
    )
    
    # Extract ground truth
    ground_truth = extractor.extract_multi_scale_ground_truth()
    
    # Visualize top matches if requested
    if args.visualize > 0:
        scales_to_viz = [args.visualize_scale] if args.visualize_scale else args.scales
        
        for scale in scales_to_viz:
            if scale in extractor.cluster_gt_matches and extractor.cluster_gt_matches[scale]:
                print(f"\nVisualizing top {args.visualize} matches for scale {scale}")
                sorted_matches = sorted(extractor.cluster_gt_matches[scale], 
                                      key=lambda m: m.confidence, reverse=True)
                
                for i, match in enumerate(sorted_matches[:args.visualize]):
                    print(f"\nVisualizing {scale} match {i+1}/{args.visualize}")
                    print(f"Match: {match.cluster_id_1} <-> {match.cluster_id_2}")
                    print(f"Overlap ratios: {match.overlap_ratio_1:.2f} | {match.overlap_ratio_2:.2f}")
                    print(f"Contact type: {match.contact_type}")
                    extractor.visualize_scale_cluster_match(match)


if __name__ == "__main__":
    main()