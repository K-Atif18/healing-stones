#!/usr/bin/env python3
"""
Optimized Fragment Processing Pipeline for Large Point Clouds
Designed for million-point clouds with color-based break surface detection
"""

import numpy as np
import open3d as o3d
import pickle
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
    """Lightweight cluster representation."""
    cluster_id: int
    barycenter: np.ndarray
    principal_axes: np.ndarray  # 3x3 matrix
    eigenvalues: np.ndarray  # 3 values
    size_signature: float
    anisotropy_signature: float
    scale: float
    point_count: int
    neighbors: List[int] = None

class StreamlinedProcessor:
    """Streamlined processor for large point clouds."""
    
    def __init__(self, 
                 data_dir: str = "Ground_Truth/artifact_1",
                 output_dir: str = "output",
                 green_threshold: float = 0.6,
                 original_downsample: int = 50000,
                 cluster_scales: List[float] = [3.0, 10.0, 20.0],
                 target_clusters_per_scale: int = 200):
        
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.green_threshold = green_threshold
        self.original_downsample = original_downsample
        self.cluster_scales = cluster_scales
        self.target_clusters_per_scale = target_clusters_per_scale
        
        # Results storage
        self.segmented_fragments = {}
        self.all_feature_clusters = []
        self.cluster_id_counter = 0
        
        logger.info(f"Initialized with scales: {self.cluster_scales}")
        logger.info(f"Target clusters per scale: {self.target_clusters_per_scale}")
        
    def process_all_fragments(self):
        """Process all fragments sequentially."""
        ply_files = sorted(self.data_dir.glob("*.ply"))
        
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.data_dir}")
        
        logger.info(f"Found {len(ply_files)} fragments to process")
        
        for i, ply_file in enumerate(ply_files):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing fragment {i+1}/{len(ply_files)}: {ply_file.name}")
            logger.info(f"{'='*60}")
            
            try:
                # Process single fragment
                self._process_single_fragment(ply_file)
                
                # Save intermediate results after each fragment
                self._save_intermediate_results()
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to process {ply_file.name}: {e}")
                continue
        
        # Save final consolidated results
        self._save_final_results()
        
    def _process_single_fragment(self, ply_file: Path):
        """Process a single fragment efficiently."""
        start_time = time.time()
        fragment_name = ply_file.stem
        
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
        
        # Phase 1.1: Color-based segmentation
        logger.info("Phase 1.1: Color-based segmentation")
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
        
        # Create surface patches
        surface_patches = {
            'break_0': break_indices.tolist(),
            'original': downsampled_original.tolist()
        }
        
        patch_classifications = {
            'break_0': 'break',
            'original': 'original'
        }
        
        # Store Phase 1.1 results
        self.segmented_fragments[fragment_name] = {
            'surface_patches': surface_patches,
            'patch_classifications': patch_classifications,
            'n_points': n_points,
            'n_break': len(break_indices),
            'n_original_full': len(original_indices),
            'n_original_downsampled': len(downsampled_original)
        }
        
        # Phase 1.2: Feature cluster extraction (only on break surfaces)
        logger.info("Phase 1.2: Feature cluster extraction on break surfaces")
        
        if len(break_indices) > 0:
            break_points = points[break_indices]
            fragment_clusters = self._extract_break_surface_clusters(
                break_points, fragment_name
            )
            
            self.all_feature_clusters.extend(fragment_clusters)
            self.segmented_fragments[fragment_name]['n_clusters'] = len(fragment_clusters)
        else:
            logger.warning("No break surface points found")
            self.segmented_fragments[fragment_name]['n_clusters'] = 0
        
        elapsed = time.time() - start_time
        logger.info(f"Fragment {fragment_name} completed in {elapsed:.1f} seconds")
        
    def _segment_by_color(self, colors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fast color-based segmentation."""
        # Green detection for break surfaces
        green_mask = (
            (colors[:, 1] > self.green_threshold) &  # High green
            (colors[:, 0] < 0.4) &  # Low red
            (colors[:, 2] < 0.4)    # Low blue
        )
        
        break_indices = np.where(green_mask)[0]
        original_indices = np.where(~green_mask)[0]
        
        return break_indices, original_indices
    
    def _extract_break_surface_clusters(self, break_points: np.ndarray, 
                                      fragment_name: str) -> List[SurfaceCluster]:
        """Extract multi-scale clusters from break surface."""
        clusters = []
        
        # Adaptive sampling for very large break surfaces
        if len(break_points) > 500000:
            logger.info(f"Large break surface ({len(break_points):,} points), using adaptive sampling")
            sample_size = min(300000, len(break_points))
            sample_indices = np.random.choice(len(break_points), sample_size, replace=False)
            sampled_points = break_points[sample_indices]
        else:
            sampled_points = break_points
        
        logger.info(f"Working with {len(sampled_points):,} break surface points")
        
        for scale in self.cluster_scales:
            logger.info(f"Extracting clusters at scale {scale}mm")
            
            # Use hybrid approach: FPS + local clustering
            scale_clusters = self._extract_clusters_hybrid(
                sampled_points, scale, fragment_name
            )
            
            clusters.extend(scale_clusters)
            logger.info(f"Added {len(scale_clusters)} clusters at scale {scale}mm")
        
        # Build cluster overlap graph
        if len(clusters) > 0:
            self._build_cluster_topology(clusters)
        
        logger.info(f"Total clusters for fragment: {len(clusters)}")
        return clusters
    
    def _extract_clusters_hybrid(self, points: np.ndarray, scale: float,
                               fragment_name: str) -> List[SurfaceCluster]:
        """Hybrid cluster extraction using FPS + local clustering."""
        clusters = []
        
        # Determine number of seed points based on surface area and scale
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        approx_surface_area = bbox_size[0] * bbox_size[1]  # Simplified
        n_seeds = min(
            self.target_clusters_per_scale,
            int(approx_surface_area / (scale * scale * 10))  # Heuristic
        )
        n_seeds = max(20, n_seeds)  # At least 20 seeds
        
        logger.info(f"Using {n_seeds} seed points for scale {scale}mm")
        
        # Farthest Point Sampling for seed selection
        seed_indices = self._farthest_point_sampling(points, n_seeds)
        seed_points = points[seed_indices]
        
        # Build spatial index
        tree = cKDTree(points)
        
        # Extract clusters around each seed
        for i, seed_idx in enumerate(seed_indices):
            seed_point = points[seed_idx]
            
            # Find points within scale radius
            indices = tree.query_ball_point(seed_point, scale)
            
            if len(indices) >= 15:  # Minimum cluster size
                cluster_points = points[indices]
                
                # Compute cluster features
                cluster = self._compute_cluster_features(
                    cluster_points, scale, fragment_name
                )
                
                if cluster is not None:
                    clusters.append(cluster)
        
        # If too few clusters, add some random samples
        if len(clusters) < self.target_clusters_per_scale // 2:
            logger.info(f"Adding random samples to increase coverage")
            n_random = min(50, len(points) // 100)
            random_indices = np.random.choice(len(points), n_random, replace=False)
            
            for idx in random_indices:
                center = points[idx]
                indices = tree.query_ball_point(center, scale)
                
                if len(indices) >= 15:
                    cluster_points = points[indices]
                    cluster = self._compute_cluster_features(
                        cluster_points, scale, fragment_name
                    )
                    if cluster is not None:
                        clusters.append(cluster)
        
        return clusters
    
    def _farthest_point_sampling(self, points: np.ndarray, n_samples: int) -> np.ndarray:
        """Farthest Point Sampling for uniform coverage."""
        n_points = len(points)
        n_samples = min(n_samples, n_points)
        
        # Start with a random point
        selected_indices = [np.random.randint(n_points)]
        distances = np.full(n_points, np.inf)
        
        for _ in range(n_samples - 1):
            # Update distances to nearest selected point
            last_point = points[selected_indices[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            
            # Select farthest point
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
            
            # Early termination if all points are close
            if distances[next_idx] < 1.0:  # 1mm threshold
                break
        
        return np.array(selected_indices)
    
    def _compute_cluster_features(self, points: np.ndarray, scale: float,
                                fragment_name: str) -> Optional[SurfaceCluster]:
        """Compute PCA-based cluster features."""
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
            
            if eigenvalues[2] > 1e-10:
                anisotropy_signature = np.sqrt(eigenvalues[1] / eigenvalues[2])
            else:
                anisotropy_signature = 1.0
            
            cluster = SurfaceCluster(
                cluster_id=self.cluster_id_counter,
                barycenter=barycenter,
                principal_axes=eigenvectors,
                eigenvalues=eigenvalues,
                size_signature=size_signature,
                anisotropy_signature=anisotropy_signature,
                scale=scale,
                point_count=len(points),
                neighbors=[]
            )
            
            self.cluster_id_counter += 1
            return cluster
            
        except Exception as e:
            logger.warning(f"Failed to compute cluster features: {e}")
            return None
    
    def _build_cluster_topology(self, clusters: List[SurfaceCluster]):
        """Build overlap graph between clusters."""
        if len(clusters) < 2:
            return
        
        # Build spatial index of cluster centers
        centers = np.array([c.barycenter for c in clusters])
        tree = cKDTree(centers)
        
        # Find overlapping clusters
        for i, cluster in enumerate(clusters):
            # Find nearby clusters
            search_radius = cluster.scale * 2.0
            neighbor_indices = tree.query_ball_point(cluster.barycenter, search_radius)
            
            for j in neighbor_indices:
                if i != j:
                    other = clusters[j]
                    distance = np.linalg.norm(cluster.barycenter - other.barycenter)
                    
                    # Check for overlap
                    overlap_threshold = (cluster.scale + other.scale) * 0.5
                    if distance < overlap_threshold:
                        cluster.neighbors.append(other.cluster_id)
    
    def _save_intermediate_results(self):
        """Save results after each fragment."""
        # Save current segmented fragments
        with open(self.output_dir / "segmented_fragments_temp.pkl", "wb") as f:
            pickle.dump(self.segmented_fragments, f)
        
        # Save current clusters
        if self.all_feature_clusters:
            cluster_data = self._prepare_cluster_data()
            with open(self.output_dir / "feature_clusters_temp.pkl", "wb") as f:
                pickle.dump(cluster_data, f)
    
    def _save_final_results(self):
        """Save final deliverables."""
        logger.info("\nSaving final results...")
        
        # Phase 1.1 deliverable
        with open(self.output_dir / "segmented_fragments.pkl", "wb") as f:
            pickle.dump(self.segmented_fragments, f)
        
        # Phase 1.2 deliverable
        cluster_data = self._prepare_cluster_data()
        with open(self.output_dir / "feature_clusters.pkl", "wb") as f:
            pickle.dump(cluster_data, f)
        
        # Clean up temp files
        temp_files = [
            self.output_dir / "segmented_fragments_temp.pkl",
            self.output_dir / "feature_clusters_temp.pkl"
        ]
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        # Print summary
        self._print_summary()
    
    def _prepare_cluster_data(self) -> Dict:
        """Prepare cluster data for saving."""
        cluster_list = []
        overlap_edges = []
        
        for cluster in self.all_feature_clusters:
            cluster_dict = {
                'cluster_id': cluster.cluster_id,
                'barycenter': cluster.barycenter,
                'principal_axes': cluster.principal_axes,
                'eigenvalues': cluster.eigenvalues,
                'size_signature': cluster.size_signature,
                'anisotropy_signature': cluster.anisotropy_signature,
                'scale': cluster.scale,
                'point_count': cluster.point_count
            }
            cluster_list.append(cluster_dict)
            
            # Collect overlap edges
            for neighbor_id in cluster.neighbors:
                if cluster.cluster_id < neighbor_id:  # Avoid duplicates
                    overlap_edges.append((cluster.cluster_id, neighbor_id))
        
        return {
            'clusters': cluster_list,
            'overlap_graph_edges': overlap_edges,
            'total_clusters': len(cluster_list)
        }
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "="*70)
        print("FRAGMENT PROCESSING SUMMARY")
        print("="*70)
        print(f"Processed fragments: {len(self.segmented_fragments)}")
        print(f"Total feature clusters: {len(self.all_feature_clusters)}")
        
        total_break = 0
        total_original = 0
        
        for fragment_name, data in self.segmented_fragments.items():
            print(f"\n{fragment_name}:")
            print(f"  Total points: {data['n_points']:,}")
            print(f"  Break surface: {data['n_break']:,}")
            print(f"  Original surface: {data['n_original_downsampled']:,} (from {data['n_original_full']:,})")
            print(f"  Feature clusters: {data.get('n_clusters', 0)}")
            
            total_break += data['n_break']
            total_original += data['n_original_downsampled']
        
        print(f"\nTOTAL:")
        print(f"  Break surface points: {total_break:,}")
        print(f"  Original surface points (downsampled): {total_original:,}")
        print(f"  Feature clusters: {len(self.all_feature_clusters)}")
        
        # Scale distribution
        scale_counts = {}
        for cluster in self.all_feature_clusters:
            scale = cluster.scale
            scale_counts[scale] = scale_counts.get(scale, 0) + 1
        
        print(f"\nCLUSTER SCALE DISTRIBUTION:")
        for scale in sorted(scale_counts.keys()):
            print(f"  {scale}mm: {scale_counts[scale]} clusters")
        
        print(f"\nOutput saved to: {self.output_dir}")
        print("="*70)


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Fragment Processing for Large Point Clouds")
    parser.add_argument("--data_dir", default="Ground_Truth/artifact_1", 
                       help="Directory containing PLY files")
    parser.add_argument("--output_dir", default="output", 
                       help="Output directory for results")
    parser.add_argument("--green_threshold", type=float, default=0.6,
                       help="Threshold for green break surface detection")
    parser.add_argument("--original_downsample", type=int, default=50000,
                       help="Target points for original surface downsampling")
    parser.add_argument("--scales", nargs='+', type=float, 
                       default=[3.0, 10.0, 20.0],
                       help="Cluster extraction scales in mm")
    parser.add_argument("--target_clusters", type=int, default=100,
                       help="Target clusters per scale")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = StreamlinedProcessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        green_threshold=args.green_threshold,
        original_downsample=args.original_downsample,
        cluster_scales=args.scales,
        target_clusters_per_scale=args.target_clusters
    )
    
    # Process all fragments
    start_time = time.time()
    
    try:
        processor.process_all_fragments()
        elapsed = time.time() - start_time
        print(f"\nPipeline completed successfully in {elapsed:.1f} seconds!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()