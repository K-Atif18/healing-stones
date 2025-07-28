#!/usr/bin/env python3
"""
Ground Truth Cluster Match Visualizer
Interactive visualization of cluster matches between fragment pairs
"""

import numpy as np
import open3d as o3d
import json
import pickle
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

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

class GTClusterVisualizer:
    """Interactive visualizer for ground truth cluster matches."""
    
    def __init__(self, 
                 gt_file: str = "Ground_Truth/multi_scale_cluster_ground_truth.json",
                 clusters_file: str = "output/feature_clusters.pkl",
                 data_dir: str = "Ground_Truth/reconstructed/artifact_1"):
        
        self.gt_file = Path(gt_file)
        self.clusters_file = Path(clusters_file)
        self.data_dir = Path(data_dir)
        
        # Load data
        self._load_ground_truth()
        self._load_cluster_data()
        
        logger.info(f"Loaded GT with {len(self.contact_pairs)} contact pairs")
        
    def _load_ground_truth(self):
        """Load ground truth data with proper handling of different formats."""
        logger.info(f"Loading ground truth from {self.gt_file}")
        
        with open(self.gt_file, 'r') as f:
            self.gt_data = json.load(f)
        
        # Extract contact pairs - handle both formats
        if 'contact_pairs' in self.gt_data:
            contact_pairs_data = self.gt_data['contact_pairs']
            
            # Check if it's the new detailed format or simple format
            if isinstance(contact_pairs_data, list) and len(contact_pairs_data) > 0:
                if isinstance(contact_pairs_data[0], dict):
                    # New detailed format: [{'fragment_1': 'frag_1', 'fragment_2': 'frag_2', ...}, ...]
                    self.contact_pairs = [(pair['fragment_1'], pair['fragment_2']) for pair in contact_pairs_data]
                else:
                    # Simple format: [['frag_1', 'frag_2'], ...]
                    self.contact_pairs = [tuple(pair) for pair in contact_pairs_data]
            else:
                self.contact_pairs = []
        
        # Fallback to simple format if available
        if not hasattr(self, 'contact_pairs') or not self.contact_pairs:
            if 'fragment_contact_pairs_simple' in self.gt_data:
                self.contact_pairs = [tuple(pair) for pair in self.gt_data['fragment_contact_pairs_simple']]
            else:
                self.contact_pairs = []
        
        # Extract matches by scale
        self.matches_by_scale = self.gt_data.get('cluster_ground_truth_matches_by_scale', {})
        
        # Create lookup for quick access
        self.matches_lookup = {}
        for scale, matches in self.matches_by_scale.items():
            for match in matches:
                pair_key = (match['fragment_1'], match['fragment_2'])
                if pair_key not in self.matches_lookup:
                    self.matches_lookup[pair_key] = {}
                if scale not in self.matches_lookup[pair_key]:
                    self.matches_lookup[pair_key][scale] = []
                self.matches_lookup[pair_key][scale].append(match)
        
        logger.info(f"Processed matches for {len(self.matches_lookup)} fragment pairs")
        logger.info(f"Contact pairs format: {type(self.contact_pairs[0]) if self.contact_pairs else 'empty'}")
    
    def _load_cluster_data(self):
        """Load cluster data for point mapping with fallback options."""
        logger.info(f"Loading cluster data from {self.clusters_file}")
        
        try:
            with open(self.clusters_file, 'rb') as f:
                self.cluster_data = pickle.load(f)
            logger.info("Successfully loaded hierarchical cluster data")
        except (AttributeError, ImportError) as e:
            logger.warning(f"Failed to load hierarchical clusters: {e}")
            
            # Try flat format as fallback
            flat_file = self.clusters_file.parent / "feature_clusters_flat.pkl"
            if flat_file.exists():
                logger.info("Trying flat cluster format...")
                self._load_flat_clusters(flat_file)
            else:
                logger.error("No compatible cluster file found!")
                raise FileNotFoundError(f"Neither hierarchical nor flat cluster file found!")
        
        logger.info("Cluster data loaded successfully")
    
    def _load_flat_clusters(self, flat_file: Path):
        """Load flat cluster data and convert to hierarchical."""
        with open(flat_file, 'rb') as f:
            flat_data = pickle.load(f)
        
        if isinstance(flat_data, dict) and 'clusters' in flat_data:
            clusters = flat_data['clusters']
        else:
            clusters = flat_data
        
        # Convert to hierarchical structure
        self.cluster_data = {}
        
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
                    'local_id': cluster_data.local_id
                }
            else:
                # Dict format
                fragment = cluster_data.get('fragment', 'unknown')
                scale = cluster_data.get('scale', '1k')
                cluster_dict = cluster_data
            
            if fragment not in self.cluster_data:
                self.cluster_data[fragment] = {}
            if scale not in self.cluster_data[fragment]:
                self.cluster_data[fragment][scale] = []
            
            self.cluster_data[fragment][scale].append(cluster_dict)
        
        logger.info("Successfully converted flat clusters to hierarchical structure")
    
    def list_contact_pairs(self):
        """List all available contact pairs with match counts."""
        print("\n" + "="*70)
        print("AVAILABLE FRAGMENT CONTACT PAIRS")
        print("="*70)
        
        for i, contact_pair in enumerate(self.contact_pairs):
            # Handle tuple format
            if isinstance(contact_pair, (list, tuple)) and len(contact_pair) >= 2:
                frag1, frag2 = contact_pair[0], contact_pair[1]
            else:
                print(f"Error: Invalid contact pair format at index {i}: {contact_pair}")
                continue
                
            print(f"\n{i+1:2d}. {frag1} ↔ {frag2}")
            
            pair_matches = self.matches_lookup.get((frag1, frag2), {})
            total_matches = sum(len(matches) for matches in pair_matches.values())
            
            if total_matches > 0:
                print(f"     Total matches: {total_matches}")
                for scale, matches in pair_matches.items():
                    if matches:
                        primary_count = sum(1 for m in matches if m['is_primary_contact'])
                        print(f"     {scale.upper()}: {len(matches)} matches ({primary_count} primary)")
            else:
                print("     No cluster matches found")
        
        print("\n" + "="*70)
    
    def visualize_pair_matches(self, pair_index: int, scale: str = "1k", 
                             top_n: int = 5, show_all: bool = False,
                             min_confidence: float = 0.0):
        """Visualize cluster matches for a specific fragment pair."""
        
        if pair_index < 1 or pair_index > len(self.contact_pairs):
            print(f"Invalid pair index. Choose 1-{len(self.contact_pairs)}")
            return
        
        # Handle contact pair format safely
        contact_pair = self.contact_pairs[pair_index - 1]
        if isinstance(contact_pair, (list, tuple)) and len(contact_pair) >= 2:
            frag1, frag2 = contact_pair[0], contact_pair[1]
        else:
            print(f"Error: Invalid contact pair format: {contact_pair}")
            return
            
        logger.info(f"Visualizing matches between {frag1} and {frag2} at scale {scale}")
        
        # Get matches for this pair and scale
        pair_matches = self.matches_lookup.get((frag1, frag2), {})
        scale_matches = pair_matches.get(scale, [])
        
        if not scale_matches:
            print(f"No matches found for {frag1} ↔ {frag2} at scale {scale}")
            return
        
        # Filter by confidence
        filtered_matches = [m for m in scale_matches if m['confidence'] >= min_confidence]
        
        if not filtered_matches:
            print(f"No matches above confidence threshold {min_confidence}")
            return
        
        # Sort by confidence
        sorted_matches = sorted(filtered_matches, key=lambda m: m['confidence'], reverse=True)
        
        # Select matches to show
        if show_all:
            matches_to_show = sorted_matches
        else:
            matches_to_show = sorted_matches[:top_n]
        
        print(f"\nShowing top {len(matches_to_show)} matches:")
        for i, match in enumerate(matches_to_show):
            print(f"{i+1:2d}. {match['cluster_id_1']} ↔ {match['cluster_id_2']} "
                  f"(conf: {match['confidence']:.3f}, type: {match['contact_type']})")
        
        # Load and visualize
        self._create_pair_visualization(frag1, frag2, matches_to_show, scale)
    
    def _create_pair_visualization(self, frag1: str, frag2: str, 
                                 matches: List[Dict], scale: str):
        """Create Open3D visualization for fragment pair with cluster matches."""
        
        # Load point clouds
        pcd1 = self._load_fragment_pointcloud(frag1)
        pcd2 = self._load_fragment_pointcloud(frag2)
        
        if pcd1 is None or pcd2 is None:
            print(f"Failed to load point clouds for {frag1} or {frag2}")
            return
        
        # Position fragments side by side for better viewing
        bbox1 = pcd1.get_axis_aligned_bounding_box()
        bbox2 = pcd2.get_axis_aligned_bounding_box()
        
        # Calculate separation distance
        width1 = bbox1.max_bound[0] - bbox1.min_bound[0]
        width2 = bbox2.max_bound[0] - bbox2.min_bound[0]
        separation = max(width1, width2) * 1.5
        
        # Move second fragment to the right
        translation = np.array([separation, 0, 0])
        pcd2.translate(translation)
        
        # Create visualization geometries
        geometries = []
        
        # Add base point clouds (in gray)
        pcd1_viz = o3d.geometry.PointCloud(pcd1)
        pcd2_viz = o3d.geometry.PointCloud(pcd2)
        
        # Color base clouds in light gray
        pcd1_viz.paint_uniform_color([0.7, 0.7, 0.7])
        pcd2_viz.paint_uniform_color([0.6, 0.6, 0.6])
        
        geometries.extend([pcd1_viz, pcd2_viz])
        
        # Highlight matched clusters
        colors = self._generate_match_colors(len(matches))
        
        for i, match in enumerate(matches):
            color = colors[i]
            
            # Create cluster point clouds
            cluster_pcd1 = self._create_cluster_pointcloud(frag1, match, scale, 1, color)
            cluster_pcd2 = self._create_cluster_pointcloud(frag2, match, scale, 2, color, translation)
            
            if cluster_pcd1 is not None:
                geometries.append(cluster_pcd1)
            if cluster_pcd2 is not None:
                geometries.append(cluster_pcd2)
            
            # Add cluster centers as spheres
            center1 = np.array(match['cluster_center_1'])
            center2 = np.array(match['cluster_center_2']) + translation
            
            sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
            sphere1.translate(center1)
            sphere1.paint_uniform_color(color)
            
            sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
            sphere2.translate(center2)
            sphere2.paint_uniform_color(color)
            
            geometries.extend([sphere1, sphere2])
            
            # Add connecting line between cluster centers
            line_points = [center1, center2]
            line_lines = [[0, 1]]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(line_lines)
            line_set.paint_uniform_color(color)
            
            geometries.append(line_set)
        
        # Add fragment labels
        self._add_fragment_labels(geometries, frag1, frag2, bbox1, bbox2, translation)
        
        # Launch visualization
        window_title = (f"GT Cluster Matches: {frag1} ↔ {frag2} "
                       f"({scale.upper()} scale, {len(matches)} matches)")
        
        print(f"\nLaunching visualization...")
        print(f"Controls:")
        print(f"  - Mouse: Rotate, zoom, pan")
        print(f"  - R: Reset view")
        print(f"  - Q/ESC: Close")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_title,
            width=1400,
            height=800
        )
    
    def _load_fragment_pointcloud(self, fragment_name: str) -> Optional[o3d.geometry.PointCloud]:
        """Load point cloud for a fragment."""
        ply_file = self.data_dir / f"{fragment_name}.ply"
        
        if not ply_file.exists():
            # Try alternative location
            ply_file = Path(f"Ground_Truth/artifact_1/{fragment_name}.ply")
        
        if ply_file.exists():
            return o3d.io.read_point_cloud(str(ply_file))
        else:
            logger.error(f"Point cloud not found: {fragment_name}.ply")
            return None
    
    def _create_cluster_pointcloud(self, fragment_name: str, match: Dict, 
                                 scale: str, cluster_num: int, color: List[float],
                                 translation: np.ndarray = None) -> Optional[o3d.geometry.PointCloud]:
        """Create point cloud for a specific cluster."""
        
        # Get cluster info
        if cluster_num == 1:
            local_id = match['local_id_1']
        else:
            local_id = match['local_id_2']
        
        # Get cluster from hierarchical data
        fragment_clusters = self.cluster_data.get(fragment_name, {})
        scale_clusters = fragment_clusters.get(scale, [])
        
        if local_id >= len(scale_clusters):
            logger.warning(f"Cluster {local_id} not found in {fragment_name} {scale}")
            return None
        
        cluster = scale_clusters[local_id]
        
        # Get point indices
        if isinstance(cluster, dict):
            point_indices = cluster.get('original_point_indices', cluster.get('point_indices', []))
        else:
            point_indices = getattr(cluster, 'original_point_indices', 
                                  getattr(cluster, 'point_indices', []))
        
        if point_indices is None or len(point_indices) == 0:
            return None
        
        # Load full point cloud
        pcd_full = self._load_fragment_pointcloud(fragment_name)
        if pcd_full is None:
            return None
        
        # Extract cluster points
        all_points = np.asarray(pcd_full.points)
        
        # Handle point indices safely
        if isinstance(point_indices, (list, np.ndarray)):
            valid_indices = [idx for idx in point_indices if 0 <= idx < len(all_points)]
        else:
            valid_indices = []
        
        if len(valid_indices) == 0:
            return None
        
        cluster_points = all_points[valid_indices]
        
        # Create cluster point cloud
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.paint_uniform_color(color)
        
        # Apply translation if needed
        if translation is not None:
            cluster_pcd.translate(translation)
        
        return cluster_pcd
    
    def _generate_match_colors(self, n_matches: int) -> List[List[float]]:
        """Generate distinct colors for matches."""
        if n_matches <= 10:
            # Predefined colors for better distinction
            base_colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
                [0.0, 0.5, 0.0],  # Dark Green
                [0.5, 0.5, 0.0],  # Olive
            ]
            return base_colors[:n_matches]
        else:
            # Generate colors using HSV space for many matches
            colors = []
            for i in range(n_matches):
                hue = (i * 360 / n_matches) % 360
                # Convert HSV to RGB (simplified)
                import colorsys
                rgb = colorsys.hsv_to_rgb(hue/360, 0.8, 0.9)
                colors.append(list(rgb))
            return colors
    
    def _add_fragment_labels(self, geometries: List, frag1: str, frag2: str,
                           bbox1, bbox2, translation: np.ndarray):
        """Add text labels for fragments (simplified as coordinate frames)."""
        
        # Add coordinate frames to indicate fragment centers
        frame_size = 10.0
        
        # Frame for fragment 1
        center1 = bbox1.get_center()
        frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame1.translate(center1)
        geometries.append(frame1)
        
        # Frame for fragment 2
        center2 = bbox2.get_center() + translation
        frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame2.translate(center2)
        geometries.append(frame2)
    
    def interactive_viewer(self):
        """Interactive command-line interface for viewing matches."""
        
        while True:
            print("\n" + "="*70)
            print("INTERACTIVE GT CLUSTER MATCH VIEWER")
            print("="*70)
            print("1. List all contact pairs")
            print("2. Visualize specific pair")
            print("3. Quick visualization (pair, scale, top N)")
            print("4. Exit")
            
            try:
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == "1":
                    self.list_contact_pairs()
                
                elif choice == "2":
                    self._interactive_pair_selection()
                
                elif choice == "3":
                    self._quick_visualization()
                
                elif choice == "4":
                    print("Goodbye!")
                    break
                
                else:
                    print("Invalid choice. Please select 1-4.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    def _interactive_pair_selection(self):
        """Interactive pair selection with detailed options."""
        
        self.list_contact_pairs()
        
        try:
            pair_idx = int(input(f"\nSelect pair (1-{len(self.contact_pairs)}): "))
            scale = input("Select scale (1k/5k/10k) [default: 1k]: ").strip() or "1k"
            top_n = int(input("Number of top matches to show [default: 5]: ") or "5")
            min_conf = float(input("Minimum confidence [default: 0.0]: ") or "0.0")
            show_all = input("Show all matches? (y/n) [default: n]: ").strip().lower() == 'y'
            
            self.visualize_pair_matches(pair_idx, scale, top_n, show_all, min_conf)
            
        except ValueError:
            print("Invalid input. Please enter numbers where required.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _quick_visualization(self):
        """Quick visualization with minimal input."""
        try:
            params = input("\nEnter: pair_index scale top_n (e.g., '1 1k 5'): ").strip().split()
            
            if len(params) >= 1:
                pair_idx = int(params[0])
                scale = params[1] if len(params) >= 2 else "1k"
                top_n = int(params[2]) if len(params) >= 3 else 5
                
                self.visualize_pair_matches(pair_idx, scale, top_n)
            else:
                print("Invalid input format")
                
        except (ValueError, IndexError):
            print("Invalid input. Use format: pair_index scale top_n")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Visualize Ground Truth Cluster Matches")
    parser.add_argument("--gt-file", default="Ground_Truth/multi_scale_cluster_ground_truth.json",
                       help="Path to ground truth JSON file")
    parser.add_argument("--clusters", default="output/feature_clusters.pkl",
                       help="Path to cluster data file")
    parser.add_argument("--data-dir", default="Ground_Truth/reconstructed/artifact_1",
                       help="Directory with PLY files")
    parser.add_argument("--pair", type=int, default=None,
                       help="Fragment pair index to visualize")
    parser.add_argument("--scale", default="1k", choices=["1k", "5k", "10k"],
                       help="Cluster scale to visualize")
    parser.add_argument("--top", type=int, default=5,
                       help="Number of top matches to show")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                       help="Minimum confidence threshold")
    parser.add_argument("--list-pairs", action="store_true",
                       help="Just list available pairs and exit")
    parser.add_argument("--interactive", action="store_true", default=True,
                       help="Launch interactive viewer")
    
    args = parser.parse_args()
    
    try:
        # Initialize visualizer
        visualizer = GTClusterVisualizer(
            gt_file=args.gt_file,
            clusters_file=args.clusters,
            data_dir=args.data_dir
        )
        
        if args.list_pairs:
            # Just list pairs
            visualizer.list_contact_pairs()
            
        elif args.pair is not None:
            # Direct visualization
            visualizer.visualize_pair_matches(
                pair_index=args.pair,
                scale=args.scale,
                top_n=args.top,
                min_confidence=args.min_confidence
            )
            
        else:
            # Interactive mode
            visualizer.interactive_viewer()
            
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()