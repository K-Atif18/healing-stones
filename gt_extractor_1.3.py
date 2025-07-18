#!/usr/bin/env python3
"""
Extract ground truth contact regions from pre-positioned fragments.
Uses fragments that are already in their correct assembled positions.
"""

import numpy as np
import open3d as o3d
import json
from pathlib import Path
from scipy.spatial import cKDTree
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PositionedFragmentGTExtractor:
    def __init__(self, 
                 positioned_dir: str = "Ground_Truth/reconstructed/artifact_1",
                 output_dir: str = "Ground_Truth",
                 contact_threshold: float = 3.0,
                 green_threshold: float = 0.6):
        
        self.positioned_dir = Path(positioned_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.contact_threshold = contact_threshold  # 3mm
        self.green_threshold = green_threshold
        
        # Fragment data storage
        self.fragments = {}
        self.break_surfaces = {}
        
    def extract_ground_truth(self):
        """Main method to extract ground truth from positioned fragments."""
        logger.info("Extracting ground truth from pre-positioned fragments...")
        logger.info(f"Fragment directory: {self.positioned_dir}")
        logger.info(f"Contact threshold: {self.contact_threshold}mm")
        
        # Step 1: Load all fragments and extract break surfaces
        self._load_fragments_and_break_surfaces()
        
        # Step 2: Find contact pairs based on break surface proximity
        contact_pairs = self._find_contact_pairs()
        
        # Step 3: Analyze detailed contact regions
        contact_details = self._analyze_contact_regions(contact_pairs)
        
        # Step 4: Compile and save ground truth
        ground_truth = self._compile_ground_truth(contact_pairs, contact_details)
        
        # Save results
        output_file = self.output_dir / "ground_truth_from_positioned.json"
        with open(output_file, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        
        logger.info(f"\nSaved ground truth to: {output_file}")
        
        # Print summary
        self._print_summary(ground_truth)
        
        return ground_truth
    
    def _load_fragments_and_break_surfaces(self):
        """Load all fragments and extract break surfaces."""
        ply_files = sorted(self.positioned_dir.glob("*.ply"))
        
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {self.positioned_dir}")
        
        logger.info(f"\nLoading {len(ply_files)} fragments...")
        
        for ply_file in tqdm(ply_files, desc="Loading fragments"):
            fragment_name = ply_file.stem
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(str(ply_file))
            
            if not pcd.has_points():
                logger.warning(f"Skipping {fragment_name}: no points")
                continue
            
            # Store fragment info
            self.fragments[fragment_name] = {
                'point_cloud': pcd,
                'n_points': len(pcd.points),
                'file_path': str(ply_file)
            }
            
            # Extract break surface
            if pcd.has_colors():
                break_indices = self._extract_break_surface_indices(pcd)
                break_points = np.asarray(pcd.points)[break_indices]
                
                self.break_surfaces[fragment_name] = {
                    'indices': break_indices,
                    'points': break_points,
                    'n_points': len(break_indices)
                }
                
                logger.info(f"  {fragment_name}: {len(pcd.points)} total, {len(break_indices)} break surface points ({len(break_indices)/len(pcd.points)*100:.1f}%)")
            else:
                logger.warning(f"  {fragment_name}: no colors, cannot extract break surface")
    
    def _extract_break_surface_indices(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """Extract indices of break surface points (green colored)."""
        colors = np.asarray(pcd.colors)
        
        # Detect green points
        green_mask = (
            (colors[:, 1] > self.green_threshold) &  # High green
            (colors[:, 0] < 0.4) &  # Low red
            (colors[:, 2] < 0.4)    # Low blue
        )
        
        return np.where(green_mask)[0]
    
    def _find_contact_pairs(self):
        """Find which fragments are in contact based on break surface proximity."""
        logger.info("\nFinding contact pairs...")
        
        fragment_names = sorted(self.break_surfaces.keys())
        contact_pairs = []
        
        # Check all pairs
        for i in range(len(fragment_names)):
            for j in range(i + 1, len(fragment_names)):
                frag1 = fragment_names[i]
                frag2 = fragment_names[j]
                
                if self._check_break_surface_contact(frag1, frag2):
                    contact_pairs.append((frag1, frag2))
        
        logger.info(f"Found {len(contact_pairs)} contact pairs")
        return contact_pairs
    
    def _check_break_surface_contact(self, frag1: str, frag2: str) -> bool:
        """Check if two fragments' break surfaces are in contact."""
        break1 = self.break_surfaces[frag1]['points']
        break2 = self.break_surfaces[frag2]['points']
        
        if len(break1) == 0 or len(break2) == 0:
            return False
        
        # Quick bounding box check
        min1, max1 = np.min(break1, axis=0), np.max(break1, axis=0)
        min2, max2 = np.min(break2, axis=0), np.max(break2, axis=0)
        
        # Check if bboxes are close
        for dim in range(3):
            if min1[dim] - self.contact_threshold > max2[dim]:
                return False
            if min2[dim] - self.contact_threshold > max1[dim]:
                return False
        
        # Detailed check: sample points for efficiency
        sample_size = min(1000, len(break1))
        if len(break1) > sample_size:
            sample_indices = np.random.choice(len(break1), sample_size, replace=False)
            sample_points = break1[sample_indices]
        else:
            sample_points = break1
        
        # Check if any sampled point is close to the other surface
        tree2 = cKDTree(break2)
        distances, _ = tree2.query(sample_points)
        
        return np.min(distances) < self.contact_threshold
    
    def _analyze_contact_regions(self, contact_pairs):
        """Analyze detailed contact regions for each pair."""
        logger.info("\nAnalyzing contact regions...")
        
        contact_details = []
        
        for frag1, frag2 in tqdm(contact_pairs, desc="Analyzing contacts"):
            details = self._compute_contact_details(frag1, frag2)
            if details:
                contact_details.append(details)
        
        return contact_details
    
    def _compute_contact_details(self, frag1: str, frag2: str):
        """Compute detailed contact information between two fragments."""
        break1 = self.break_surfaces[frag1]
        break2 = self.break_surfaces[frag2]
        
        points1 = break1['points']
        points2 = break2['points']
        
        # Build KD-trees
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        # Find contact points
        contact_mask1 = np.zeros(len(points1), dtype=bool)
        contact_mask2 = np.zeros(len(points2), dtype=bool)
        distances1 = []
        distances2 = []
        
        # Check each point in break1
        for i, point in enumerate(points1):
            dist, idx = tree2.query(point)
            if dist < self.contact_threshold:
                contact_mask1[i] = True
                distances1.append(dist)
        
        # Check each point in break2
        for i, point in enumerate(points2):
            dist, idx = tree1.query(point)
            if dist < self.contact_threshold:
                contact_mask2[i] = True
                distances2.append(dist)
        
        # Get contact points
        contact_points1 = points1[contact_mask1]
        contact_points2 = points2[contact_mask2]
        
        if len(contact_points1) == 0 and len(contact_points2) == 0:
            return None
        
        # Compute statistics
        all_distances = distances1 + distances2
        
        # Compute contact centers
        if len(contact_points1) > 0:
            contact_center1 = np.mean(contact_points1, axis=0)
        else:
            contact_center1 = np.mean(points1, axis=0)  # Fallback
        
        if len(contact_points2) > 0:
            contact_center2 = np.mean(contact_points2, axis=0)
        else:
            contact_center2 = np.mean(points2, axis=0)  # Fallback
        
        # Estimate contact area
        if len(contact_points1) > 3:
            # Use PCA to find contact plane
            centered = contact_points1 - contact_center1
            cov = np.cov(centered.T)
            eigenvalues, _ = np.linalg.eigh(cov)
            # Contact area approximation (ellipse)
            contact_area = np.pi * np.sqrt(max(eigenvalues[1], 0)) * np.sqrt(max(eigenvalues[2], 0))
        else:
            contact_area = 0.0
        
        # Get original indices for contact points
        original_indices1 = break1['indices'][contact_mask1]
        original_indices2 = break2['indices'][contact_mask2]
        
        return {
            'fragment_1': frag1,
            'fragment_2': frag2,
            'contact_point_count': len(contact_points1) + len(contact_points2),
            'contact_points_1': len(contact_points1),
            'contact_points_2': len(contact_points2),
            'contact_center_1': contact_center1.tolist(),
            'contact_center_2': contact_center2.tolist(),
            'mean_gap': float(np.mean(all_distances)) if all_distances else 0.0,
            'max_gap': float(np.max(all_distances)) if all_distances else 0.0,
            'min_gap': float(np.min(all_distances)) if all_distances else 0.0,
            'estimated_contact_area': float(contact_area),
            'contact_indices_1': original_indices1.tolist()[:1000],  # Limit for file size
            'contact_indices_2': original_indices2.tolist()[:1000]
        }
    
    def _compile_ground_truth(self, contact_pairs, contact_details):
        """Compile all ground truth data."""
        # Fragment information
        fragments = {}
        for frag_name, frag_data in self.fragments.items():
            pcd = frag_data['point_cloud']
            
            # Get bounding box
            bbox = pcd.get_axis_aligned_bounding_box()
            
            fragments[frag_name] = {
                # Since fragments are pre-positioned, transform is identity
                'transform_matrix': np.eye(4).tolist(),
                'vertex_count': frag_data['n_points'],
                'bbox': {
                    'min': bbox.min_bound.tolist(),
                    'max': bbox.max_bound.tolist()
                },
                'file_path': frag_data['file_path']
            }
        
        # Break surface statistics
        break_surface_stats = {}
        for frag_name, break_data in self.break_surfaces.items():
            total_points = self.fragments[frag_name]['n_points']
            break_points = break_data['n_points']
            
            break_surface_stats[frag_name] = {
                'n_break_vertices': break_points,
                'n_total_vertices': total_points,
                'percentage': (break_points / total_points * 100) if total_points > 0 else 0
            }
        
        return {
            'fragments': fragments,
            'contact_pairs': [(f1, f2) for f1, f2 in contact_pairs],
            'contact_details': contact_details,
            'break_surface_stats': break_surface_stats,
            'extraction_info': {
                'source_directory': str(self.positioned_dir),
                'contact_threshold': self.contact_threshold,
                'green_threshold': self.green_threshold,
                'extraction_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'fragments_are_pre_positioned': True
            }
        }
    
    def _print_summary(self, ground_truth):
        """Print summary of extracted ground truth."""
        print("\n" + "="*70)
        print("GROUND TRUTH EXTRACTION SUMMARY")
        print("="*70)
        
        print(f"\nFragments: {len(ground_truth['fragments'])}")
        print(f"Contact pairs: {len(ground_truth['contact_pairs'])}")
        
        print("\nBreak Surface Statistics:")
        total_break = 0
        for frag, stats in ground_truth['break_surface_stats'].items():
            print(f"  {frag}: {stats['n_break_vertices']:,} points ({stats['percentage']:.1f}%)")
            total_break += stats['n_break_vertices']
        print(f"  Total break surface points: {total_break:,}")
        
        print("\nContact Regions:")
        for contact in ground_truth['contact_details']:
            print(f"\n  {contact['fragment_1']} <-> {contact['fragment_2']}:")
            print(f"    Contact points: {contact['contact_point_count']} "
                  f"({contact['contact_points_1']} + {contact['contact_points_2']})")
            print(f"    Mean gap: {contact['mean_gap']:.2f}mm")
            print(f"    Gap range: [{contact['min_gap']:.2f}, {contact['max_gap']:.2f}]mm")
            print(f"    Contact area: {contact['estimated_contact_area']:.1f}mm²")
    
    def visualize_contact_pair(self, frag1_name: str, frag2_name: str, 
                             ground_truth_file: str = None):
        """Quick visualization of a contact pair."""
        if ground_truth_file:
            with open(ground_truth_file, 'r') as f:
                gt = json.load(f)
                # Use loaded ground truth
        
        if frag1_name not in self.fragments or frag2_name not in self.fragments:
            logger.error(f"Fragments not found")
            return
        
        # Get point clouds
        pcd1 = self.fragments[frag1_name]['point_cloud']
        pcd2 = self.fragments[frag2_name]['point_cloud']
        
        # Get break surfaces
        break_mask1 = self._extract_break_surface_indices(pcd1)
        break_mask2 = self._extract_break_surface_indices(pcd2)
        
        # Create colored point clouds
        pcd1_copy = o3d.geometry.PointCloud(pcd1)
        pcd2_copy = o3d.geometry.PointCloud(pcd2)
        
        # Color non-break surfaces gray
        colors1 = np.asarray(pcd1_copy.colors)
        colors2 = np.asarray(pcd2_copy.colors)
        
        colors1[~np.isin(np.arange(len(colors1)), break_mask1)] = [0.7, 0.7, 0.7]
        colors2[~np.isin(np.arange(len(colors2)), break_mask2)] = [0.5, 0.5, 0.5]
        
        # Highlight break surfaces
        colors1[break_mask1] = [1.0, 0.2, 0.2]  # Red
        colors2[break_mask2] = [0.2, 1.0, 0.2]  # Green
        
        pcd1_copy.colors = o3d.utility.Vector3dVector(colors1)
        pcd2_copy.colors = o3d.utility.Vector3dVector(colors2)
        
        # Visualize
        o3d.visualization.draw_geometries(
            [pcd1_copy, pcd2_copy],
            window_name=f"Contact: {frag1_name} <-> {frag2_name}"
        )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract ground truth from pre-positioned fragments")
    parser.add_argument("--input-dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with positioned fragments")
    parser.add_argument("--output-dir", default="Ground_Truth",
                       help="Output directory")
    parser.add_argument("--contact-threshold", type=float, default=3.0,
                       help="Contact threshold in mm")
    parser.add_argument("--green-threshold", type=float, default=0.6,
                       help="Green color threshold for break surface detection")
    parser.add_argument("--visualize", nargs=2, metavar=('FRAG1', 'FRAG2'),
                       help="Visualize a specific fragment pair after extraction")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = PositionedFragmentGTExtractor(
        positioned_dir=args.input_dir,
        output_dir=args.output_dir,
        contact_threshold=args.contact_threshold,
        green_threshold=args.green_threshold
    )
    
    # Extract ground truth
    ground_truth = extractor.extract_ground_truth()
    
    # Visualize if requested
    if args.visualize:
        extractor.visualize_contact_pair(args.visualize[0], args.visualize[1])


if __name__ == "__main__":
    main()