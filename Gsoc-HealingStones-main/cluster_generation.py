#!/usr/bin/env python3
"""
Unified Point-Count Based Fragment Processing Pipeline
Combines clustering, mapping, and ground truth preparation in one script
"""

import numpy as np
import open3d as o3d
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import logging
from tqdm import tqdm
import time
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SurfaceCluster:
    """Cluster representation with point-count based naming."""
    cluster_id: str  # Format: frag_1_c1k_001, frag_1_c5k_001, etc.
    fragment_name: str
    point_count_scale: int  # 1000, 5000, 10000
    scale_name: str  # "1k", "5k", "10k"
    local_id: int  # Local cluster ID within fragment and scale
    barycenter: np.ndarray
    principal_axes: np.ndarray  # 3x3 matrix
    eigenvalues: np.ndarray  # 3 values
    size_signature: float
    anisotropy_signature: float
    point_count: int
    point_indices: List[int]  # Indices of points in this cluster (relative to break surface)
    original_point_indices: List[int]  # Indices in original point cloud
    neighbors: List[str] = None  # List of neighboring cluster IDs

class ClusterGeneration:
    """Unified pipeline that does everything in one pass."""
    
    def __init__(self, 
                 data_dir: str = "Ground_Truth/artifact_1",
                 output_dir: str = "output",
                 green_threshold: float = 0.6,
                 original_downsample: int = 50000,
                 point_count_scales: List[int] = [1000, 5000, 10000]):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.green_threshold = green_threshold
        self.original_downsample = original_downsample
        self.point_count_scales = sorted(point_count_scales)  # [1000, 5000, 10000]
        
        # Results storage - organized for immediate access
        self.segmented_fragments = {}
        self.hierarchical_clusters = {}  # fragment -> scale -> clusters
        self.cluster_registry = {}  # cluster_id -> cluster_object
        self.spatial_lookup = {}  # For fast spatial queries
        self.fragment_mappings = {}  # Complete mapping structures
        
        logger.info(f"Initialized unified pipeline with point count scales: {self.point_count_scales}")
        
    def process_all_fragments(self):
        """Process all fragments with real-time mapping."""
        ply_files = sorted(self.data_dir.glob("*.ply"))
        
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.data_dir}")
        
        logger.info(f"Found {len(ply_files)} fragments to process")
        
        # Initialize structures
        self._initialize_data_structures()
        
        for i, ply_file in enumerate(ply_files):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing fragment {i+1}/{len(ply_files)}: {ply_file.name}")
            logger.info(f"{'='*60}")
            
            try:
                # Process single fragment with immediate mapping
                self._process_fragment_with_mapping(ply_file)
                
                # Update global mappings in real-time
                self._update_global_mappings(ply_file.stem)
                
                # Save intermediate results
                self._save_intermediate_results()
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process {ply_file.name}: {e}")
                continue
        
        # Finalize and save everything
        self._finalize_pipeline()
        
    def _initialize_data_structures(self):
        """Initialize all data structures."""
        logger.info("Initializing unified data structures...")
        
        self.fragment_mappings = {
            'fragment_to_clusters': {},
            'cluster_to_fragment': {},
            'scale_distributions': {scale: {} for scale in ['1k', '5k', '10k']},
            'point_assignments': {},
            'spatial_indices': {},
            'processing_order': []
        }
        
    def _process_fragment_with_mapping(self, ply_file: Path):
        """Process fragment and create mappings immediately."""
        start_time = time.time()
        fragment_name = ply_file.stem
        
        # Add to processing order
        self.fragment_mappings['processing_order'].append(fragment_name)
        
        # Load point cloud
        logger.info("Loading point cloud...")
        pcd = o3d.io.read_point_cloud(str(ply_file))
        
        if not pcd.has_points():
            raise ValueError(f"No points in {ply_file.name}")
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        n_points = len(points)
        logger.info(f"Loaded {n_points:,} points")
        
        if colors is None:
            raise ValueError(f"No colors in {ply_file.name} - cannot detect break surfaces")
        
        # Phase 1: Color-based segmentation
        logger.info("Phase 1: Color-based segmentation")
        break_indices, original_indices = self._segment_by_color(colors)
        
        logger.info(f"Break surface: {len(break_indices):,} points ({len(break_indices)/n_points*100:.1f}%)")
        logger.info(f"Original surface: {len(original_indices):,} points")
        
        # Downsample original surface
        if len(original_indices) > self.original_downsample:
            logger.info(f"Downsampling original surface: {len(original_indices):,} → {self.original_downsample:,}")
            np.random.seed(42)
            downsampled_original = np.random.choice(
                original_indices, 
                size=self.original_downsample, 
                replace=False
            )
        else:
            downsampled_original = original_indices
        
        # Initialize fragment structures
        self.hierarchical_clusters[fragment_name] = {}
        self.fragment_mappings['fragment_to_clusters'][fragment_name] = {}
        
        # Phase 2: Point-count clustering with immediate assignment
        logger.info("Phase 2: Point-count clustering with real-time mapping")
        
        segment_indices = np.full(n_points, -1, dtype=int)  # -1 = unassigned
        total_clusters = 0
        
        if len(break_indices) > 0:
            break_points = points[break_indices]
            
            # Process each scale separately with immediate mapping
            for scale_points in self.point_count_scales:
                scale_name = f"{scale_points//1000}k"
                logger.info(f"\nProcessing scale: {scale_name} ({scale_points} points per cluster)")
                
                # Create clusters for this scale
                scale_clusters = self._create_scale_clusters_with_mapping(
                    break_points, break_indices, scale_points, scale_name, 
                    fragment_name, points
                )
                
                # Store in hierarchical structure
                self.hierarchical_clusters[fragment_name][scale_name] = scale_clusters
                
                # Update fragment mappings
                cluster_ids = [c.cluster_id for c in scale_clusters]
                self.fragment_mappings['fragment_to_clusters'][fragment_name][scale_name] = cluster_ids
                
                # Update segment indices
                self._assign_points_to_segments(segment_indices, scale_clusters, break_indices)
                
                total_clusters += len(scale_clusters)
                logger.info(f"Created {len(scale_clusters)} clusters for scale {scale_name}")
        
        # Store segmentation results with complete mapping
        surface_patches = {
            'break_0': break_indices.tolist(),
            'original': downsampled_original.tolist()
        }
        
        patch_classifications = {
            'break_0': 'break',
            'original': 'original'
        }
        
        self.segmented_fragments[fragment_name] = {
            'surface_patches': surface_patches,
            'patch_classifications': patch_classifications,
            'segment_indices': segment_indices,
            'n_points': n_points,
            'n_break': len(break_indices),
            'n_original_full': len(original_indices),
            'n_original_downsampled': len(downsampled_original),
            'n_clusters_total': total_clusters,
            'n_assigned': int(np.sum(segment_indices >= 0)),
            'cluster_counts_by_scale': {
                f"{s//1000}k": len(self.hierarchical_clusters[fragment_name].get(f"{s//1000}k", []))
                for s in self.point_count_scales
            }
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Fragment {fragment_name} completed in {elapsed:.1f} seconds")
        logger.info(f"Total clusters created: {total_clusters}")
        
    def _segment_by_color(self, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fast color-based segmentation."""
        green_mask = (
            (colors[:, 1] > self.green_threshold) &  # High green
            (colors[:, 0] < 0.4) &  # Low red
            (colors[:, 2] < 0.4)    # Low blue
        )
        
        break_indices = np.where(green_mask)[0]
        original_indices = np.where(~green_mask)[0]
        
        return break_indices, original_indices
    
    def _create_scale_clusters_with_mapping(self, break_points: np.ndarray, 
                                          break_indices: np.ndarray,
                                          target_points: int, scale_name: str,
                                          fragment_name: str, all_points: np.ndarray) -> List[SurfaceCluster]:
        """Create clusters for a specific scale with immediate mapping."""
        
        if len(break_points) < target_points:
            logger.warning(f"Not enough break points ({len(break_points)}) for {target_points}-point clusters")
            return []
        
        # Calculate number of clusters
        n_clusters = len(break_points) // target_points
        logger.info(f"Creating {n_clusters} clusters of {target_points} points each")
        
        if n_clusters == 0:
            return []
        
        clusters = []
        
        # Smart initialization of cluster centers
        cluster_centers = self._initialize_smart_centers(break_points, n_clusters)
        
        # Build spatial index
        tree = cKDTree(break_points)
        
        # Balanced assignment algorithm
        cluster_assignments = self._balanced_cluster_assignment(
            break_points, cluster_centers, target_points, n_clusters
        )
        
        # Create cluster objects with immediate registration
        for i in range(n_clusters):
            cluster_point_indices = cluster_assignments[i]
            
            if len(cluster_point_indices) >= target_points * 0.8:  # At least 80% full
                cluster_points = break_points[cluster_point_indices]
                cluster_break_indices = break_indices[cluster_point_indices]
                
                # Generate cluster ID
                cluster_id = f"{fragment_name}_c{scale_name}_{i+1:03d}"
                
                # Create cluster object
                cluster = self._create_cluster_object(
                    cluster_points, cluster_point_indices, cluster_break_indices,
                    cluster_id, fragment_name, target_points, scale_name, i+1
                )
                
                if cluster is not None:
                    clusters.append(cluster)
                    
                    # Register immediately
                    self.cluster_registry[cluster_id] = cluster
                    
                    # Update cluster-to-fragment mapping
                    self.fragment_mappings['cluster_to_fragment'][cluster_id] = {
                        'fragment': fragment_name,
                        'scale': scale_name,
                        'local_id': i+1,
                        'point_count': cluster.point_count,
                        'barycenter': cluster.barycenter.tolist()
                    }
                    
                    # Update point assignments
                    self.fragment_mappings['point_assignments'][cluster_id] = cluster_break_indices.tolist()
        
        return clusters
    
    def _initialize_smart_centers(self, points: np.ndarray, n_clusters: int) -> np.ndarray:
        """Smart initialization using density-aware sampling."""
        if n_clusters >= len(points):
            return points.copy()
        
        # Start with point closest to centroid
        overall_center = np.mean(points, axis=0)
        distances_to_center = np.linalg.norm(points - overall_center, axis=1)
        first_idx = np.argmin(distances_to_center)
        
        centers = [points[first_idx]]
        
        # Use modified farthest point sampling
        for _ in range(n_clusters - 1):
            min_distances = np.full(len(points), np.inf)
            
            for center in centers:
                distances = np.linalg.norm(points - center, axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Add randomness to avoid edge cases
            n_candidates = max(1, len(points) // 20)
            candidate_indices = np.argpartition(min_distances, -n_candidates)[-n_candidates:]
            next_idx = np.random.choice(candidate_indices)
            
            centers.append(points[next_idx])
        
        return np.array(centers)
    
    def _balanced_cluster_assignment(self, points: np.ndarray, centers: np.ndarray,
                                   target_points: int, n_clusters: int) -> List[List[int]]:
        """Balanced assignment ensuring equal cluster sizes."""
        
        cluster_assignments = [[] for _ in range(n_clusters)]
        cluster_capacities = [target_points] * n_clusters
        
        # Iterative assignment with balancing
        for iteration in range(3):
            # Reset for this iteration
            if iteration > 0:
                cluster_assignments = [[] for _ in range(n_clusters)]
                cluster_capacities = [target_points] * n_clusters
            
            # Calculate distances to all centers
            distances = np.array([
                np.linalg.norm(points - center, axis=1) 
                for center in centers
            ])  # Shape: (n_clusters, n_points)
            
            # Sort points by minimum distance to any center
            min_distances = np.min(distances, axis=0)
            sorted_indices = np.argsort(min_distances)
            
            # Greedy assignment respecting capacity
            for point_idx in sorted_indices:
                point_distances = distances[:, point_idx]
                sorted_clusters = np.argsort(point_distances)
                
                # Assign to first available cluster
                for cluster_idx in sorted_clusters:
                    if cluster_capacities[cluster_idx] > 0:
                        cluster_assignments[cluster_idx].append(point_idx)
                        cluster_capacities[cluster_idx] -= 1
                        break
            
            # Update centers based on assignments
            for i in range(n_clusters):
                if len(cluster_assignments[i]) > 0:
                    assigned_points = points[cluster_assignments[i]]
                    centers[i] = np.mean(assigned_points, axis=0)
        
        return cluster_assignments
    
    def _create_cluster_object(self, points: np.ndarray, point_indices: List[int],
                             break_indices: List[int], cluster_id: str, 
                             fragment_name: str, target_points: int, 
                             scale_name: str, local_id: int) -> Optional[SurfaceCluster]:
        """Create a complete cluster object."""
        try:
            # Barycenter
            barycenter = np.mean(points, axis=0)
            
            # PCA
            centered = points - barycenter
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (descending)
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Ensure positive eigenvalues
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Compute signatures
            size_signature = np.sqrt(np.sum(eigenvalues))
            anisotropy_signature = np.sqrt(eigenvalues[1] / eigenvalues[2]) if eigenvalues[2] > 1e-10 else 1.0
            
            cluster = SurfaceCluster(
                cluster_id=cluster_id,
                fragment_name=fragment_name,
                point_count_scale=target_points,
                scale_name=scale_name,
                local_id=local_id,
                barycenter=barycenter,
                principal_axes=eigenvectors,
                eigenvalues=eigenvalues,
                size_signature=size_signature,
                anisotropy_signature=anisotropy_signature,
                point_count=len(points),
                point_indices=point_indices,
                original_point_indices=break_indices,
                neighbors=[]
            )
            
            return cluster
            
        except Exception as e:
            logger.warning(f"Failed to create cluster {cluster_id}: {e}")
            return None
    
    def _assign_points_to_segments(self, segment_indices: np.ndarray, 
                                 clusters: List[SurfaceCluster],
                                 break_indices: np.ndarray):
        """Assign points to segment indices."""
        for cluster in clusters:
            # Generate unique segment ID for this cluster
            segment_id = hash(cluster.cluster_id) % 100000
            
            # Assign all points in this cluster
            for break_relative_idx in cluster.point_indices:
                if break_relative_idx < len(break_indices):
                    original_idx = break_indices[break_relative_idx]
                    segment_indices[original_idx] = segment_id
    
    def _update_global_mappings(self, fragment_name: str):
        """Update global mapping structures after processing a fragment."""
        
        # Update scale distributions
        for scale in ['1k', '5k', '10k']:
            count = len(self.hierarchical_clusters[fragment_name].get(scale, []))
            self.fragment_mappings['scale_distributions'][scale][fragment_name] = count
        
        # Update spatial indices for this fragment
        self._build_spatial_indices(fragment_name)
        
    def _build_spatial_indices(self, fragment_name: str):
        """Build spatial indices for fast lookup."""
        self.fragment_mappings['spatial_indices'][fragment_name] = {}
        
        for scale_name, clusters in self.hierarchical_clusters[fragment_name].items():
            if clusters:
                centers = np.array([c.barycenter for c in clusters])
                cluster_ids = [c.cluster_id for c in clusters]
                
                self.fragment_mappings['spatial_indices'][fragment_name][scale_name] = {
                    'tree': cKDTree(centers),
                    'centers': centers,
                    'cluster_ids': cluster_ids
                }
    
    def _save_intermediate_results(self):
        """Save intermediate results."""
        # Save current state
        with open(self.output_dir / "segmented_fragments_temp.pkl", "wb") as f:
            pickle.dump(self.segmented_fragments, f)
        
        with open(self.output_dir / "hierarchical_clusters_temp.pkl", "wb") as f:
            pickle.dump(self.hierarchical_clusters, f)
        
        with open(self.output_dir / "fragment_mappings_temp.pkl", "wb") as f:
            pickle.dump(self.fragment_mappings, f)
    
    def _finalize_pipeline(self):
        """Finalize the complete pipeline."""
        logger.info("\n" + "="*70)
        logger.info("FINALIZING UNIFIED PIPELINE")
        logger.info("="*70)
        
        # Build global spatial lookup
        self._build_global_spatial_lookup()
        
        # Extract ground truth mappings
        self._extract_ground_truth_mappings()
        
        # Save all final results
        self._save_final_results()
        
        # Generate comprehensive report
        self._generate_final_report()
        
        # Clean up temp files
        self._cleanup_temp_files()
        
        logger.info("✅ Unified pipeline completed successfully!")
    
    def _build_global_spatial_lookup(self):
        """Build global spatial lookup across all fragments and scales."""
        logger.info("Building global spatial lookup...")
        
        all_centers = []
        all_cluster_ids = []
        all_fragments = []
        all_scales = []
        
        for fragment_name, scales in self.hierarchical_clusters.items():
            for scale_name, clusters in scales.items():
                for cluster in clusters:
                    all_centers.append(cluster.barycenter)
                    all_cluster_ids.append(cluster.cluster_id)
                    all_fragments.append(fragment_name)
                    all_scales.append(scale_name)
        
        if all_centers:
            self.fragment_mappings['global_spatial_lookup'] = {
                'tree': cKDTree(np.array(all_centers)),
                'centers': np.array(all_centers),
                'cluster_ids': all_cluster_ids,
                'fragments': all_fragments,
                'scales': all_scales
            }
            
            logger.info(f"Global spatial index built with {len(all_centers)} clusters")
    
    def _extract_ground_truth_mappings(self):
        """Extract cluster-level ground truth mappings."""
        logger.info("Extracting ground truth mappings...")
        
        # Look for ground truth files
        gt_files = [
            "Ground_Truth/ground_truth_from_positioned.json",
            "Ground_Truth/blender/ground_truth_assembly.json",
            "Ground_Truth/cluster_level_ground_truth.json"
        ]
        
        ground_truth = None
        for gt_file in gt_files:
            if Path(gt_file).exists():
                with open(gt_file, 'r') as f:
                    ground_truth = json.load(f)
                logger.info(f"Loaded ground truth: {gt_file}")
                break
        
        if ground_truth is None:
            logger.warning("No ground truth file found - creating placeholder")
            self.cluster_ground_truth = {'cluster_matches': []}
            return
        
        # Extract cluster-level matches from fragment-level GT
        cluster_matches = []
        contact_details = ground_truth.get('contact_details', [])
        
        for contact in contact_details:
            frag1 = contact.get('fragment_1')
            frag2 = contact.get('fragment_2')
            
            if not frag1 or not frag2:
                continue
            
            center1 = np.array(contact.get('contact_center_1', [0, 0, 0]))
            center2 = np.array(contact.get('contact_center_2', [0, 0, 0]))
            
            # Find cluster matches for each scale
            for scale in ['1k', '5k', '10k']:
                matches = self._find_cluster_matches_for_contact(
                    frag1, frag2, center1, center2, scale, contact
                )
                cluster_matches.extend(matches)
        
        self.cluster_ground_truth = {
            'cluster_matches': cluster_matches,
            'extraction_method': 'proximity_based_unified',
            'total_matches': len(cluster_matches),
            'matches_by_scale': {
                scale: len([m for m in cluster_matches if m.get('scale') == scale])
                for scale in ['1k', '5k', '10k']
            }
        }
        
        logger.info(f"Extracted {len(cluster_matches)} cluster-level GT matches")
    
    def _find_cluster_matches_for_contact(self, frag1: str, frag2: str,
                                        center1: np.ndarray, center2: np.ndarray,
                                        scale: str, contact: dict) -> List[Dict]:
        """Find cluster matches near contact points."""
        matches = []
        threshold = 25.0  # 25mm threshold
        
        # Get clusters for both fragments at this scale
        clusters1 = self.hierarchical_clusters.get(frag1, {}).get(scale, [])
        clusters2 = self.hierarchical_clusters.get(frag2, {}).get(scale, [])
        
        if not clusters1 or not clusters2:
            return matches
        
        # Find nearby clusters
        for i, c1 in enumerate(clusters1):
            dist1 = np.linalg.norm(c1.barycenter - center1)
            if dist1 < threshold:
                for j, c2 in enumerate(clusters2):
                    dist2 = np.linalg.norm(c2.barycenter - center2)
                    if dist2 < threshold:
                        match = {
                            'fragment_1': frag1,
                            'fragment_2': frag2,
                            'scale': scale,
                            'cluster_id_1': c1.cluster_id,
                            'cluster_id_2': c2.cluster_id,
                            'local_id_1': c1.local_id,
                            'local_id_2': c2.local_id,
                            'distance_to_contact_1': float(dist1),
                            'distance_to_contact_2': float(dist2),
                            'total_distance': float(dist1 + dist2),
                            'confidence': max(0.1, 1.0 - (dist1 + dist2) / (2 * threshold)),
                            'contact_center_1': center1.tolist(),
                            'contact_center_2': center2.tolist(),
                            'original_contact': {
                                'contact_id': contact.get('contact_id', ''),
                                'contact_area': contact.get('contact_area', 0)
                            }
                        }
                        matches.append(match)
        
        return matches
    
    def _save_final_results(self):
        """Save all final results."""
        logger.info("Saving final results...")
        
        # 1. Segmented fragments (with segment indices)
        with open(self.output_dir / "segmented_fragments.pkl", "wb") as f:
            pickle.dump(self.segmented_fragments, f)
        
        # 2. Hierarchical clusters (main structure)
        with open(self.output_dir / "feature_clusters.pkl", "wb") as f:
            pickle.dump(self.hierarchical_clusters, f)
        
        # 3. Complete mappings (for fast lookup)
        with open(self.output_dir / "cluster_mappings.pkl", "wb") as f:
            pickle.dump(self.fragment_mappings, f)
        
        # 4. Flat cluster list (for compatibility)
        flat_clusters = self._create_flat_cluster_list()
        with open(self.output_dir / "feature_clusters_flat.pkl", "wb") as f:
            pickle.dump(flat_clusters, f)
        
        # 5. Cluster registry (for direct access)
        cluster_registry_serializable = {
            cluster_id: self._serialize_cluster(cluster)
            for cluster_id, cluster in self.cluster_registry.items()
        }
        with open(self.output_dir / "cluster_registry.pkl", "wb") as f:
            pickle.dump(cluster_registry_serializable, f)
        
        # 6. Ground truth mappings
        with open(self.output_dir / "cluster_level_ground_truth.json", "w") as f:
            json.dump(self.cluster_ground_truth, f, indent=2, default=self._json_serializer)
        
        logger.info("All results saved successfully ✅")
    
    def _create_flat_cluster_list(self):
        """Create flat list for compatibility."""
        flat_clusters = []
        
        for fragment_name, scales in self.hierarchical_clusters.items():
            for scale_name, clusters in scales.items():
                for cluster in clusters:
                    cluster_dict = self._serialize_cluster(cluster)
                    flat_clusters.append(cluster_dict)
        
        return {
            'clusters': flat_clusters,
            'total_clusters': len(flat_clusters),
            'processing_order': self.fragment_mappings['processing_order']
        }
    
    def _serialize_cluster(self, cluster: SurfaceCluster) -> Dict:
        """Convert cluster object to serializable dictionary."""
        return {
            'cluster_id': cluster.cluster_id,
            'fragment': cluster.fragment_name,
            'scale': cluster.scale_name,
            'point_count_scale': cluster.point_count_scale,
            'local_id': cluster.local_id,
            'barycenter': cluster.barycenter.tolist(),
            'principal_axes': cluster.principal_axes.tolist(),
            'eigenvalues': cluster.eigenvalues.tolist(),
            'size_signature': float(cluster.size_signature),
            'anisotropy_signature': float(cluster.anisotropy_signature),
            'point_count': cluster.point_count,
            'point_indices': cluster.point_indices,
            'original_point_indices': cluster.original_point_indices
        }
    
    def _generate_final_report(self):
        """Generate comprehensive final report."""
        logger.info("Generating final report...")
        
        # Calculate statistics
        total_fragments = len(self.segmented_fragments)
        total_clusters = sum(
            len(clusters) 
            for scales in self.hierarchical_clusters.values() 
            for clusters in scales.values()
        )
        
        report = {
            'pipeline_summary': {
                'type': 'unified_point_count_clustering',
                'total_fragments': total_fragments,
                'total_clusters': total_clusters,
                'scales_processed': ['1k', '5k', '10k'],
                'point_count_scales': self.point_count_scales,
                'processing_order': self.fragment_mappings['processing_order']
            },
            'fragment_details': {},
            'cluster_distribution': {},
            'ground_truth_coverage': {},
            'data_integrity': {}
        }
        
        # Fragment details
        for fragment_name, seg_data in self.segmented_fragments.items():
            scales_data = self.hierarchical_clusters.get(fragment_name, {})
            
            fragment_info = {
                'total_points': seg_data.get('n_points', 0),
                'break_points': seg_data.get('n_break', 0),
                'original_points': seg_data.get('n_original_downsampled', 0),
                'clusters_by_scale': {},
                'total_clusters': 0,
                'assignment_rate': seg_data.get('n_assigned', 0) / seg_data.get('n_break', 1)
            }
            
            for scale in ['1k', '5k', '10k']:
                n_clusters = len(scales_data.get(scale, []))
                fragment_info['clusters_by_scale'][scale] = n_clusters
                fragment_info['total_clusters'] += n_clusters
            
            report['fragment_details'][fragment_name] = fragment_info
        
        # Cluster distribution
        for scale in ['1k', '5k', '10k']:
            scale_total = sum(
                len(self.hierarchical_clusters.get(frag, {}).get(scale, []))
                for frag in self.hierarchical_clusters.keys()
            )
            report['cluster_distribution'][scale] = scale_total
        
        # Ground truth coverage
        if hasattr(self, 'cluster_ground_truth'):
            report['ground_truth_coverage'] = self.cluster_ground_truth.get('matches_by_scale', {})
            report['ground_truth_coverage']['total_matches'] = self.cluster_ground_truth.get('total_matches', 0)
        
        # Data integrity checks
        report['data_integrity'] = {
            'all_clusters_registered': len(self.cluster_registry) == total_clusters,
            'mapping_consistency': self._verify_mapping_consistency(),
            'spatial_index_coverage': self._verify_spatial_coverage()
        }
        
        # Save report
        with open(self.output_dir / "unified_pipeline_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)
        
        # Print summary
        self._print_final_summary(report)
    
    def _verify_mapping_consistency(self) -> bool:
        """Verify mapping consistency."""
        try:
            # Check fragment consistency
            seg_fragments = set(self.segmented_fragments.keys())
            cluster_fragments = set(self.hierarchical_clusters.keys())
            mapping_fragments = set(self.fragment_mappings['fragment_to_clusters'].keys())
            
            return seg_fragments == cluster_fragments == mapping_fragments
        except:
            return False
    
    def _verify_spatial_coverage(self) -> bool:
        """Verify spatial index coverage."""
        try:
            if 'global_spatial_lookup' not in self.fragment_mappings:
                return False
            
            global_count = len(self.fragment_mappings['global_spatial_lookup']['cluster_ids'])
            registry_count = len(self.cluster_registry)
            
            return global_count == registry_count
        except:
            return False
    
    def _print_final_summary(self, report):
        """Print comprehensive final summary."""
        print("\n" + "="*80)
        print("UNIFIED POINT-COUNT CLUSTERING PIPELINE - FINAL SUMMARY")
        print("="*80)
        
        summary = report['pipeline_summary']
        print(f"\n📊 OVERVIEW:")
        print(f"   Fragments processed: {summary['total_fragments']}")
        print(f"   Total clusters created: {summary['total_clusters']:,}")
        print(f"   Processing order: {' → '.join(summary['processing_order'])}")
        
        print(f"\n🎯 CLUSTER DISTRIBUTION BY SCALE:")
        for scale, count in report['cluster_distribution'].items():
            print(f"   {scale.upper()}: {count:,} clusters")
        
        print(f"\n📋 FRAGMENT BREAKDOWN:")
        for fragment_name, details in report['fragment_details'].items():
            print(f"\n   {fragment_name}:")
            print(f"     Total points: {details['total_points']:,}")
            print(f"     Break points: {details['break_points']:,}")
            print(f"     Assignment rate: {details['assignment_rate']:.1%}")
            print(f"     Clusters:")
            for scale, count in details['clusters_by_scale'].items():
                if count > 0:
                    avg_points = self.point_count_scales[['1k', '5k', '10k'].index(scale)]
                    print(f"       {scale.upper()}: {count} clusters (~{avg_points} points each)")
        
        if 'ground_truth_coverage' in report and report['ground_truth_coverage']:
            print(f"\n🎯 GROUND TRUTH COVERAGE:")
            print(f"   Total cluster matches: {report['ground_truth_coverage'].get('total_matches', 0)}")
            for scale, count in report['ground_truth_coverage'].items():
                if scale != 'total_matches':
                    print(f"   {scale.upper()}: {count} matches")
        
        integrity = report['data_integrity']
        print(f"\n✅ DATA INTEGRITY:")
        print(f"   All clusters registered: {'✓' if integrity['all_clusters_registered'] else '✗'}")
        print(f"   Mapping consistency: {'✓' if integrity['mapping_consistency'] else '✗'}")
        print(f"   Spatial index coverage: {'✓' if integrity['spatial_index_coverage'] else '✗'}")
        
        print(f"\n📁 OUTPUT FILES CREATED:")
        output_files = [
            ("segmented_fragments.pkl", "Fragment segmentation with point assignments"),
            ("feature_clusters.pkl", "Hierarchical cluster organization"),
            ("cluster_mappings.pkl", "Complete mapping structures & spatial indices"),
            ("feature_clusters_flat.pkl", "Flat cluster list (compatibility)"),
            ("cluster_registry.pkl", "Direct cluster object access"),
            ("cluster_level_ground_truth.json", "Cluster-level ground truth matches"),
            ("unified_pipeline_report.json", "Comprehensive analysis report")
        ]
        
        for filename, description in output_files:
            print(f"   ✓ {filename:<30} - {description}")
        
        print(f"\n🏷️  CLUSTER NAMING CONVENTION:")
        print(f"   Format: {{fragment}}_c{{scale}}_{{local_id:03d}}")
        print(f"   Examples:")
        for fragment in list(report['fragment_details'].keys())[:2]:
            for scale in ['1k', '5k']:
                if report['fragment_details'][fragment]['clusters_by_scale'][scale] > 0:
                    print(f"     {fragment}_c{scale}_001 - Fragment {fragment}, {scale} cluster, ID 001")
        
        print(f"\n🚀 READY FOR:")
        print(f"   • Assembly matching with cluster-level precision")
        print(f"   • Ground truth evaluation using cluster_level_ground_truth.json")
        print(f"   • Spatial queries using built-in spatial indices")
        print(f"   • Multi-scale analysis across 1k/5k/10k point clusters")
        
        print("="*80)
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        temp_files = [
            "segmented_fragments_temp.pkl",
            "hierarchical_clusters_temp.pkl", 
            "fragment_mappings_temp.pkl"
        ]
        
        for temp_file in temp_files:
            temp_path = self.output_dir / temp_file
            if temp_path.exists():
                temp_path.unlink()
        
        logger.info("Temporary files cleaned up ✅")
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy arrays."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def verify_pipeline_integrity(output_dir: str = "output"):
    """Verify complete pipeline integrity."""
    logger.info("Verifying unified pipeline integrity...")
    
    output_path = Path(output_dir)
    required_files = [
        "segmented_fragments.pkl",
        "feature_clusters.pkl",
        "cluster_mappings.pkl",
        "cluster_registry.pkl",
        "unified_pipeline_report.json"
    ]
    
    # Check file existence
    missing_files = [f for f in required_files if not (output_path / f).exists()]
    if missing_files:
        logger.error(f"❌ Missing files: {missing_files}")
        return False
    
    try:
        # Load and verify data consistency
        with open(output_path / "segmented_fragments.pkl", 'rb') as f:
            seg_data = pickle.load(f)
        
        with open(output_path / "feature_clusters.pkl", 'rb') as f:
            cluster_data = pickle.load(f)
        
        with open(output_path / "cluster_mappings.pkl", 'rb') as f:
            mappings = pickle.load(f)
        
        with open(output_path / "cluster_registry.pkl", 'rb') as f:
            registry = pickle.load(f)
        
        # Consistency checks
        seg_fragments = set(seg_data.keys())
        cluster_fragments = set(cluster_data.keys())
        mapping_fragments = set(mappings['fragment_to_clusters'].keys())
        
        if not (seg_fragments == cluster_fragments == mapping_fragments):
            logger.error("❌ Fragment set inconsistency detected")
            return False
        
        # Count verification
        total_clusters_hierarchical = sum(
            len(clusters) for scales in cluster_data.values() for clusters in scales.values()
        )
        total_clusters_registry = len(registry)
        
        if total_clusters_hierarchical != total_clusters_registry:
            logger.error(f"❌ Cluster count mismatch: {total_clusters_hierarchical} vs {total_clusters_registry}")
            return False
        
        logger.info("✅ All integrity checks passed!")
        logger.info(f"✅ {len(seg_fragments)} fragments, {total_clusters_hierarchical} clusters verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integrity verification failed: {e}")
        return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Point-Count Clustering Pipeline")
    parser.add_argument("--data_dir", default="Ground_Truth/artifact_1", 
                       help="Directory containing PLY files")
    parser.add_argument("--output_dir", default="output", 
                       help="Output directory for results")
    parser.add_argument("--green_threshold", type=float, default=0.6,
                       help="Threshold for green break surface detection")
    parser.add_argument("--original_downsample", type=int, default=50000,
                       help="Target points for original surface downsampling")
    parser.add_argument("--point_scales", nargs='+', type=int, 
                       default=[1000, 5000, 10000],
                       help="Point counts per cluster (e.g., 1000 5000 10000)")
    parser.add_argument("--verify_only", action="store_true",
                       help="Only verify existing results without processing")
    
    args = parser.parse_args()
    
    if args.verify_only:
        # Just verify existing results
        success = verify_pipeline_integrity(args.output_dir)
        if success:
            print("🎉 Pipeline verification successful!")
        else:
            print("❌ Pipeline verification failed!")
        return
    
    # Run complete pipeline
    pipeline = ClusterGeneration(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        green_threshold=args.green_threshold,
        original_downsample=args.original_downsample,
        point_count_scales=args.point_scales
    )
    
    start_time = time.time()
    
    try:
        # Run unified pipeline
        pipeline.process_all_fragments()
        
        # Verify results
        if verify_pipeline_integrity(args.output_dir):
            elapsed = time.time() - start_time
            print(f"\nUNIFIED PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"Total time: {elapsed:.1f} seconds")
            print(f"Results saved to: {args.output_dir}")
            print(f"\nReady for assembly matching and analysis!")
        else:
            print("❌ Pipeline completed but integrity verification failed!")
            
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()