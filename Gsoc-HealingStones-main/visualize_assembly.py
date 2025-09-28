#!/usr/bin/env python3
"""
Visualize the matches found by the assembly pipeline
Shows both GT and non-GT matches with different colors
"""

import numpy as np
import open3d as o3d
import h5py
import pickle
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Match:
    fragment_1: str
    fragment_2: str
    cluster_id_1: int
    cluster_id_2: int
    distance: float
    confidence: float
    is_ground_truth: bool
    gt_confidence: float = 0.0

class AssemblyMatchVisualizer:
    def __init__(self,
                 assembly_h5: str = "output/cluster_assembly_with_gt.h5",
                 clusters_file: str = "output/feature_clusters_fixed.pkl",
                 segments_file: str = "output/segmented_fragments_with_indices.pkl",
                 ply_dir: str = "Ground_Truth/reconstructed/artifact_1"):
        
        self.ply_dir = Path(ply_dir)
        
        # Load assembly results
        logger.info("Loading assembly results...")
        self.matches = self._load_assembly_matches(assembly_h5)
        
        # Load cluster data
        with open(clusters_file, 'rb') as f:
            self.cluster_data = pickle.load(f)
        
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Organize data
        self._organize_clusters()
        self._organize_matches()
        
        # Print summary
        self._print_summary()
    
    def _load_assembly_matches(self, h5_file: str) -> List[Match]:
        """Load matches from assembly HDF5 file."""
        matches = []
        
        with h5py.File(h5_file, 'r') as f:
            if 'cluster_matches' not in f:
                logger.error("No cluster_matches found in HDF5 file")
                return matches
            
            matches_group = f['cluster_matches']
            
            # Load arrays
            n_matches = len(matches_group['cluster_id_1'])
            
            for i in range(n_matches):
                match = Match(
                    fragment_1=matches_group['fragment_1'][i].decode('utf-8'),
                    fragment_2=matches_group['fragment_2'][i].decode('utf-8'),
                    cluster_id_1=int(matches_group['cluster_id_1'][i]),
                    cluster_id_2=int(matches_group['cluster_id_2'][i]),
                    distance=float(matches_group['distances'][i]),
                    confidence=float(matches_group['confidences'][i]),
                    is_ground_truth=bool(matches_group['is_ground_truth'][i]),
                    gt_confidence=float(matches_group.get('gt_confidences', [0]*n_matches)[i])
                )
                matches.append(match)
        
        return matches
    
    def _organize_clusters(self):
        """Organize clusters by fragment."""
        self.clusters_by_fragment = {}
        cluster_idx = 0
        
        for frag_name in sorted(self.segment_data.keys()):
            n_clusters = self.segment_data[frag_name].get('n_clusters', 0)
            fragment_clusters = []
            
            for i in range(n_clusters):
                if cluster_idx < len(self.cluster_data['clusters']):
                    cluster = self.cluster_data['clusters'][cluster_idx].copy()
                    cluster['fragment'] = frag_name
                    cluster['local_id'] = i
                    cluster['global_id'] = cluster_idx
                    fragment_clusters.append(cluster)
                    cluster_idx += 1
            
            self.clusters_by_fragment[frag_name] = fragment_clusters
    
    def _organize_matches(self):
        """Organize matches by fragment pair."""
        self.matches_by_pair = {}
        self.gt_matches = []
        self.non_gt_matches = []
        
        for match in self.matches:
            # Organize by pair
            pair = tuple(sorted([match.fragment_1, match.fragment_2]))
            if pair not in self.matches_by_pair:
                self.matches_by_pair[pair] = {'gt': [], 'non_gt': []}
            
            if match.is_ground_truth:
                self.matches_by_pair[pair]['gt'].append(match)
                self.gt_matches.append(match)
            else:
                self.matches_by_pair[pair]['non_gt'].append(match)
                self.non_gt_matches.append(match)
        
        # Sort by confidence
        for pair in self.matches_by_pair:
            self.matches_by_pair[pair]['gt'].sort(key=lambda m: m.gt_confidence, reverse=True)
            self.matches_by_pair[pair]['non_gt'].sort(key=lambda m: m.confidence, reverse=True)
    
    def _print_summary(self):
        """Print summary of loaded matches."""
        print("\n" + "="*60)
        print("ASSEMBLY MATCHES SUMMARY")
        print("="*60)
        print(f"Total matches loaded: {len(self.matches)}")
        print(f"Ground truth matches: {len(self.gt_matches)}")
        print(f"Non-GT matches: {len(self.non_gt_matches)}")
        print(f"GT percentage: {len(self.gt_matches)/len(self.matches)*100:.1f}%")
        print(f"\nFragment pairs: {len(self.matches_by_pair)}")
        
        # Top GT matches
        if self.gt_matches:
            print("\nTop 5 GT matches by confidence:")
            top_gt = sorted(self.gt_matches, key=lambda m: m.gt_confidence, reverse=True)[:5]
            for i, m in enumerate(top_gt):
                print(f"  {i+1}. {m.fragment_1}:C{m.cluster_id_1} <-> "
                      f"{m.fragment_2}:C{m.cluster_id_2} "
                      f"(conf={m.gt_confidence:.3f}, dist={m.distance:.1f}mm)")
    
    def list_pairs(self):
        """List all fragment pairs with match counts."""
        print("\nFragment pairs with matches:")
        print("-" * 70)
        
        for i, (pair, matches) in enumerate(sorted(self.matches_by_pair.items())):
            n_gt = len(matches['gt'])
            n_non_gt = len(matches['non_gt'])
            total = n_gt + n_non_gt
            
            print(f"{i+1}. {pair[0]} <-> {pair[1]}: "
                  f"{total} total ({n_gt} GT, {n_non_gt} non-GT)")
            
            # Show top GT match if exists
            if matches['gt']:
                top = matches['gt'][0]
                print(f"   Top GT: C{top.cluster_id_1} <-> C{top.cluster_id_2} "
                      f"(conf={top.gt_confidence:.3f})")
    
    def visualize_pair(self, frag1: str, frag2: str,
                      show_gt_only: bool = False,
                      show_top_n: int = 20,
                      n_gt_matches: int = None,
                      n_non_gt_matches: int = None,
                      min_confidence: float = 0.5):
        """Visualize matches for a specific fragment pair."""
        pair = tuple(sorted([frag1, frag2]))
        
        if pair not in self.matches_by_pair:
            logger.error(f"No matches found for {frag1} <-> {frag2}")
            self.list_pairs()
            return
        
        pair_matches = self.matches_by_pair[pair]
        gt_matches = pair_matches['gt']
        non_gt_matches = pair_matches['non_gt']
        
        # Handle match count parameters
        if n_gt_matches is None:
            n_gt_matches = show_top_n
        if n_non_gt_matches is None:
            n_non_gt_matches = 0 if show_gt_only else show_top_n // 2
        
        logger.info(f"\nVisualizing {frag1} <-> {frag2}")
        logger.info(f"Total GT matches: {len(gt_matches)}, Total Non-GT: {len(non_gt_matches)}")
        logger.info(f"Showing: {min(n_gt_matches, len(gt_matches))} GT, {min(n_non_gt_matches, len(non_gt_matches))} non-GT")
        
        # Load fragments
        pcd1 = o3d.io.read_point_cloud(str(self.ply_dir / f"{frag1}.ply"))
        pcd2 = o3d.io.read_point_cloud(str(self.ply_dir / f"{frag2}.ply"))
        
        # Color fragments
        pcd1.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray
        pcd2.paint_uniform_color([0.6, 0.6, 0.6])  # Darker gray
        
        geometries = [pcd1, pcd2]
        
        # Get clusters
        clusters1 = {c['local_id']: c for c in self.clusters_by_fragment[frag1]}
        clusters2 = {c['local_id']: c for c in self.clusters_by_fragment[frag2]}
        
        # Visualize GT matches (green connections)
        gt_shown = 0
        for i, match in enumerate(gt_matches[:n_gt_matches]):
            if match.confidence < min_confidence:
                continue
            
            c1 = clusters1.get(match.cluster_id_1)
            c2 = clusters2.get(match.cluster_id_2)
            
            if c1 and c2:
                # GT match - green spheres and lines
                sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
                sphere1.translate(c1['barycenter'])
                sphere1.paint_uniform_color([0, 1, 0])  # Green
                
                sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
                sphere2.translate(c2['barycenter'])
                sphere2.paint_uniform_color([0, 0.8, 0])  # Darker green
                
                # Connection line
                points = [c1['barycenter'], c2['barycenter']]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
                
                geometries.extend([sphere1, sphere2, line_set])
                gt_shown += 1
        
        # Visualize non-GT matches (red connections)
        non_gt_shown = 0
        if n_non_gt_matches > 0:
            for i, match in enumerate(non_gt_matches[:n_non_gt_matches]):
                if match.confidence < min_confidence:
                    continue
                
                c1 = clusters1.get(match.cluster_id_1)
                c2 = clusters2.get(match.cluster_id_2)
                
                if c1 and c2:
                    # Non-GT match - red spheres and lines
                    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
                    sphere1.translate(c1['barycenter'])
                    sphere1.paint_uniform_color([1, 0, 0])  # Red
                    
                    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
                    sphere2.translate(c2['barycenter'])
                    sphere2.paint_uniform_color([0.8, 0, 0])  # Darker red
                    
                    # Connection line
                    points = [c1['barycenter'], c2['barycenter']]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
                    
                    geometries.extend([sphere1, sphere2, line_set])
                    non_gt_shown += 1
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        geometries.append(coord_frame)
        
        # Add legend
        print("\nVisualization Legend:")
        print("- Green spheres/lines: Ground Truth matches")
        print("- Red spheres/lines: Non-GT matches (algorithm found)")
        print("- Sphere size indicates match type (GT larger)")
        print(f"\nActually showing: {gt_shown} GT matches, {non_gt_shown} non-GT matches")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Assembly Matches: {frag1} <-> {frag2}"
        )
    
    def visualize_all_matches(self, n_gt_matches: int = 50, n_non_gt_matches: int = 50):
        """Visualize top GT and non-GT matches from all pairs."""
        logger.info(f"\nVisualizing top {n_gt_matches} GT and {n_non_gt_matches} non-GT matches...")
        
        # Get top matches
        top_gt = sorted(self.gt_matches, key=lambda m: m.gt_confidence, reverse=True)[:n_gt_matches]
        top_non_gt = sorted(self.non_gt_matches, key=lambda m: m.confidence, reverse=True)[:n_non_gt_matches]
        
        # Group by fragment to load each only once
        fragments_to_load = set()
        for match in top_gt + top_non_gt:
            fragments_to_load.add(match.fragment_1)
            fragments_to_load.add(match.fragment_2)
        
        # Load all fragments
        fragment_pcds = {}
        fragment_colors = {}
        for i, frag in enumerate(sorted(fragments_to_load)):
            pcd = o3d.io.read_point_cloud(str(self.ply_dir / f"{frag}.ply"))
            # Assign unique color to each fragment
            color = np.array([
                (i * 0.1) % 0.6 + 0.3,
                ((i + 3) * 0.15) % 0.6 + 0.3,
                ((i + 7) * 0.2) % 0.6 + 0.3
            ])
            pcd.paint_uniform_color(color)
            fragment_pcds[frag] = pcd
            fragment_colors[frag] = color
        
        geometries = list(fragment_pcds.values())
        
        # Add GT match connections (green)
        logger.info(f"Adding {len(top_gt)} GT matches...")
        for i, match in enumerate(top_gt):
            c1 = None
            c2 = None
            
            # Find clusters
            for c in self.clusters_by_fragment.get(match.fragment_1, []):
                if c['local_id'] == match.cluster_id_1:
                    c1 = c
                    break
            
            for c in self.clusters_by_fragment.get(match.fragment_2, []):
                if c['local_id'] == match.cluster_id_2:
                    c2 = c
                    break
            
            if c1 and c2:
                # Create connection
                sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
                sphere1.translate(c1['barycenter'])
                sphere1.paint_uniform_color([0, 1, 0])  # Green
                
                sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
                sphere2.translate(c2['barycenter'])
                sphere2.paint_uniform_color([0, 1, 0])  # Green
                
                # Connection line with gradient color based on confidence
                points = [c1['barycenter'], c2['barycenter']]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                
                # Color based on confidence (high conf = bright green, low = yellow)
                conf_normalized = min(match.gt_confidence / 3.0, 1.0)
                line_color = [1 - conf_normalized, 1, 0]  # Yellow to green
                line_set.colors = o3d.utility.Vector3dVector([line_color])
                
                geometries.extend([sphere1, sphere2, line_set])
        
        # Add non-GT match connections (red/orange)
        logger.info(f"Adding {len(top_non_gt)} non-GT matches...")
        for i, match in enumerate(top_non_gt):
            c1 = None
            c2 = None
            
            # Find clusters
            for c in self.clusters_by_fragment.get(match.fragment_1, []):
                if c['local_id'] == match.cluster_id_1:
                    c1 = c
                    break
            
            for c in self.clusters_by_fragment.get(match.fragment_2, []):
                if c['local_id'] == match.cluster_id_2:
                    c2 = c
                    break
            
            if c1 and c2:
                # Create connection
                sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
                sphere1.translate(c1['barycenter'])
                sphere1.paint_uniform_color([1, 0, 0])  # Red
                
                sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=2.5)
                sphere2.translate(c2['barycenter'])
                sphere2.paint_uniform_color([1, 0, 0])  # Red
                
                # Connection line with gradient color based on confidence
                points = [c1['barycenter'], c2['barycenter']]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                
                # Color based on confidence (high conf = red, low = orange)
                conf_normalized = min(match.confidence / 0.8, 1.0)
                line_color = [1, 1 - conf_normalized * 0.5, 0]  # Orange to red
                line_set.colors = o3d.utility.Vector3dVector([line_color])
                
                geometries.extend([sphere1, sphere2, line_set])
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        geometries.append(coord_frame)
        
        # Print summary and color legend
        print(f"\nShowing top {len(top_gt)} GT matches and {len(top_non_gt)} non-GT matches")
        print("\nColor Legend:")
        print("- GT matches (green spheres):")
        print("  - Line colors: Yellow (low confidence) -> Bright Green (high confidence)")
        print("- Non-GT matches (red spheres):")
        print("  - Line colors: Orange (low confidence) -> Red (high confidence)")
        print("\nFragment colors:")
        for frag, color in sorted(fragment_colors.items()):
            print(f"  {frag}: RGB{(color * 255).astype(int).tolist()}")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Top {n_gt_matches} GT and {n_non_gt_matches} Non-GT Matches"
        )
    
    def visualize_all_gt_matches(self, show_top_n: int = 50):
        """Visualize top GT matches from all pairs."""
        logger.info(f"\nVisualizing top {show_top_n} GT matches across all pairs...")
        
        # Sort all GT matches by confidence
        top_gt = sorted(self.gt_matches, key=lambda m: m.gt_confidence, reverse=True)[:show_top_n]
        
        # Group by fragment to load each only once
        fragments_to_load = set()
        for match in top_gt:
            fragments_to_load.add(match.fragment_1)
            fragments_to_load.add(match.fragment_2)
        
        # Load all fragments
        fragment_pcds = {}
        for frag in fragments_to_load:
            pcd = o3d.io.read_point_cloud(str(self.ply_dir / f"{frag}.ply"))
            # Assign random color to each fragment
            color = np.random.rand(3) * 0.5 + 0.3  # Avoid too dark/bright
            pcd.paint_uniform_color(color)
            fragment_pcds[frag] = pcd
        
        geometries = list(fragment_pcds.values())
        
        # Add GT match connections
        for i, match in enumerate(top_gt):
            c1 = None
            c2 = None
            
            # Find clusters
            for c in self.clusters_by_fragment[match.fragment_1]:
                if c['local_id'] == match.cluster_id_1:
                    c1 = c
                    break
            
            for c in self.clusters_by_fragment[match.fragment_2]:
                if c['local_id'] == match.cluster_id_2:
                    c2 = c
                    break
            
            if c1 and c2:
                # Create connection
                sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
                sphere1.translate(c1['barycenter'])
                sphere1.paint_uniform_color([0, 1, 0])
                
                sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=4.0)
                sphere2.translate(c2['barycenter'])
                sphere2.paint_uniform_color([0, 1, 0])
                
                # Connection line with gradient color based on confidence
                points = [c1['barycenter'], c2['barycenter']]
                lines = [[0, 1]]
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(points)
                line_set.lines = o3d.utility.Vector2iVector(lines)
                
                # Color based on confidence (high conf = bright green, low = yellow)
                conf_normalized = min(match.gt_confidence / 3.0, 1.0)
                line_color = [1 - conf_normalized, 1, 0]  # Yellow to green
                line_set.colors = o3d.utility.Vector3dVector([line_color])
                
                geometries.extend([sphere1, sphere2, line_set])
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        geometries.append(coord_frame)
        
        print(f"\nShowing top {len(top_gt)} GT matches")
        print("Line colors: Yellow (low confidence) -> Green (high confidence)")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Top {show_top_n} GT Matches Across All Fragments"
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize assembly pipeline matches")
    parser.add_argument("--assembly-h5", default="output/cluster_assembly_with_gt.h5",
                       help="Assembly results HDF5 file")
    parser.add_argument("--clusters", default="output/feature_clusters_fixed.pkl",
                       help="Cluster data file")
    parser.add_argument("--segments", default="output/segmented_fragments_with_indices.pkl",
                       help="Segment data file")
    parser.add_argument("--ply-dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with positioned PLY files")
    parser.add_argument("--pair", nargs=2, metavar=("FRAG1", "FRAG2"),
                       help="Visualize specific fragment pair")
    parser.add_argument("--gt-only", action="store_true",
                       help="Show only GT matches")
    parser.add_argument("--all-gt", action="store_true",
                       help="Show top GT matches across all pairs")
    parser.add_argument("--all", action="store_true",
                       help="Show both GT and non-GT matches across all pairs")
    parser.add_argument("--top-n", type=int, default=20,
                       help="Number of top matches to show (legacy parameter)")
    parser.add_argument("--gt-matches", type=int, default=None,
                       help="Number of GT matches to show")
    parser.add_argument("--non-gt-matches", type=int, default=None,
                       help="Number of non-GT matches to show")
    parser.add_argument("--min-conf", type=float, default=0.5,
                       help="Minimum confidence threshold")
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = AssemblyMatchVisualizer(
        assembly_h5=args.assembly_h5,
        clusters_file=args.clusters,
        segments_file=args.segments,
        ply_dir=args.ply_dir
    )
    
    # Handle the different visualization modes
    if args.all:
        # Show both GT and non-GT matches across all fragments
        n_gt = args.gt_matches if args.gt_matches is not None else 50
        n_non_gt = args.non_gt_matches if args.non_gt_matches is not None else 50
        viz.visualize_all_matches(n_gt_matches=n_gt, n_non_gt_matches=n_non_gt)
    elif args.all_gt:
        # Show only GT matches across all fragments (legacy mode)
        n_gt = args.gt_matches if args.gt_matches is not None else args.top_n
        viz.visualize_all_gt_matches(show_top_n=n_gt)
    elif args.pair:
        # Visualize specific pair
        viz.visualize_pair(
            args.pair[0], args.pair[1],
            show_gt_only=args.gt_only,
            show_top_n=args.top_n,
            n_gt_matches=args.gt_matches,
            n_non_gt_matches=args.non_gt_matches,
            min_confidence=args.min_conf
        )
    else:
        # List available pairs
        viz.list_pairs()
        print("\nUsage examples:")
        print("  # Visualize specific pair with custom match counts:")
        print("  python visualize_assembly_matches.py --pair frag_1 frag_5 --gt-matches 30 --non-gt-matches 20")
        print("  \n  # Show only GT matches:")
        print("  python visualize_assembly_matches.py --pair frag_1 frag_5 --gt-only --gt-matches 50")
        print("  \n  # Show top 50 GT and 50 non-GT matches across all fragments:")
        print("  python visualize_assembly_matches.py --all --gt-matches 50 --non-gt-matches 50")
        print("  \n  # Show 100 GT matches and 30 non-GT matches:")
        print("  python visualize_assembly_matches.py --all --gt-matches 100 --non-gt-matches 30")


if __name__ == "__main__":
    main()