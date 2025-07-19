#!/usr/bin/env python3
"""
Interactive 3D visualization of contact regions between fragment pairs.
Shows fragments in different colors with contact regions highlighted.
"""

import numpy as np
import open3d as o3d
import pickle
import json
import h5py
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContactRegionVisualizer:
    def __init__(self,
                 ply_dir: str = "Ground_Truth/artifact_1",
                 segments_file: str = "output/segmented_fragments.pkl",
                 clusters_file: str = "output/feature_clusters_fixed.pkl",
                 ground_truth_file: str = "Ground_Truth/ground_truth_from_positioned.json",
                 assembly_file: str = "output/cluster_assembly_priors_improved.h5"):
        
        self.ply_dir = Path(ply_dir)
        
        # Load all data
        logger.info("Loading data...")
        self._load_data(segments_file, clusters_file, ground_truth_file, assembly_file)
        
        # Visualization parameters
        self.contact_threshold = 10.0  # mm
        self.point_size = 1.0
        
        # Check if we're using pre-positioned fragments
        self.using_positioned = False
        if 'extraction_info' in self.ground_truth:
            self.using_positioned = self.ground_truth['extraction_info'].get('fragments_are_pre_positioned', False)
            if self.using_positioned:
                # Update PLY directory to positioned fragments
                source_dir = self.ground_truth['extraction_info'].get('source_directory', str(self.ply_dir))
                self.ply_dir = Path(source_dir)
                logger.info(f"Using pre-positioned fragments from: {self.ply_dir}")
        
    def _load_data(self, segments_file, clusters_file, ground_truth_file, assembly_file):
        """Load all required data."""
        # Load segmentation data
        with open(segments_file, 'rb') as f:
            self.segment_data = pickle.load(f)
        
        # Load cluster data - try fixed file first
        if Path(clusters_file).exists():
            with open(clusters_file, 'rb') as f:
                self.cluster_data = pickle.load(f)
        elif Path("output/feature_clusters_fixed.pkl").exists():
            with open("output/feature_clusters_fixed.pkl", 'rb') as f:
                self.cluster_data = pickle.load(f)
        else:
            with open("output/feature_clusters.pkl", 'rb') as f:
                self.cluster_data = pickle.load(f)
        
        # Load ground truth
        with open(ground_truth_file, 'r') as f:
            self.ground_truth = json.load(f)
        
        # Load assembly results
        self.assembly_data = {}
        if Path(assembly_file).exists():
            with h5py.File(assembly_file, 'r') as f:
                if 'cluster_matches' in f:
                    matches = f['cluster_matches']
                    self.assembly_data = {
                        'cluster_id_1': matches['cluster_id_1'][:],
                        'cluster_id_2': matches['cluster_id_2'][:],
                        'is_ground_truth': matches['is_ground_truth'][:],
                        'confidences': matches['confidences'][:],
                        'fragment_1': [s.decode() if isinstance(s, bytes) else s for s in matches['fragment_1'][:]],
                        'fragment_2': [s.decode() if isinstance(s, bytes) else s for s in matches['fragment_2'][:]]
                    }
        
        # Get contact pairs
        self.contact_pairs = self.ground_truth['contact_pairs']
        self.contact_details = {
            tuple(sorted([c['fragment_1'], c['fragment_2']])): c 
            for c in self.ground_truth.get('contact_details', [])
        }
        
        # Organize clusters by fragment (matching the assembly script logic)
        self._organize_clusters_by_fragment()
        
        logger.info(f"Loaded {len(self.contact_pairs)} contact pairs")
        if self.assembly_data:
            gt_count = sum(1 for x in self.assembly_data['is_ground_truth'] if x)
            logger.info(f"Loaded {len(self.assembly_data['cluster_id_1'])} matches ({gt_count} GT)")
    
    def _organize_clusters_by_fragment(self):
        """Organize clusters by fragment matching the assembly script."""
        self.clusters_by_fragment = {}
        self.cluster_lookup = {}
        
        # Get fragment processing order (sorted)
        fragment_names = sorted(self.segment_data.keys())
        
        # Track cluster assignment
        cluster_idx = 0
        
        for frag_name in fragment_names:
            n_clusters = self.segment_data[frag_name].get('n_clusters', 0)
            fragment_clusters = []
            
            # Assign next n_clusters to this fragment
            for i in range(n_clusters):
                if cluster_idx < len(self.cluster_data['clusters']):
                    cluster = self.cluster_data['clusters'][cluster_idx].copy()
                    cluster['fragment'] = frag_name
                    cluster['global_id'] = cluster_idx
                    
                    self.cluster_lookup[cluster['cluster_id']] = cluster
                    fragment_clusters.append(cluster)
                    cluster_idx += 1
            
            self.clusters_by_fragment[frag_name] = fragment_clusters
    
    def list_contact_pairs(self):
        """List all available contact pairs."""
        print("\nAvailable contact pairs:")
        print("-" * 40)
        for i, (frag1, frag2) in enumerate(self.contact_pairs):
            contact_key = tuple(sorted([frag1, frag2]))
            details = self.contact_details.get(contact_key, {})
            n_points = details.get('contact_point_count', 0)
            area = details.get('estimated_contact_area', 0)
            
            # Count GT matches for this pair
            gt_matches = 0
            if self.assembly_data:
                for j in range(len(self.assembly_data['cluster_id_1'])):
                    if (self.assembly_data['fragment_1'][j] == frag1 and 
                        self.assembly_data['fragment_2'][j] == frag2 and
                        self.assembly_data['is_ground_truth'][j]):
                        gt_matches += 1
            
            print(f"{i+1}. {frag1} <-> {frag2}")
            print(f"   Contact points: {n_points}")
            print(f"   Contact area: {area:.1f} mm²")
            print(f"   GT cluster matches: {gt_matches}")
            print()
    
    def visualize_contact_pair(self, frag1_name: str, frag2_name: str, 
                             show_clusters: bool = True,
                             show_gt_matches: bool = True):
        """Visualize a specific contact pair with contact regions highlighted."""
        logger.info(f"Visualizing contact between {frag1_name} and {frag2_name}")
        
        # Load point clouds
        pcd1 = self._load_fragment_pcd(frag1_name)
        pcd2 = self._load_fragment_pcd(frag2_name)
        
        if pcd1 is None or pcd2 is None:
            logger.error("Failed to load point clouds")
            return
        
        # Transform to world coordinates
        pcd1_world = self._transform_to_world(pcd1, frag1_name)
        pcd2_world = self._transform_to_world(pcd2, frag2_name)
        
        # Color fragments (base colors)
        self._color_fragment_base(pcd1_world, [0.8, 0.2, 0.2])  # Red
        self._color_fragment_base(pcd2_world, [0.2, 0.8, 0.2])  # Green
        
        # Identify and highlight contact regions
        contact_pcd1, contact_pcd2 = self._identify_contact_regions(
            pcd1_world, pcd2_world, frag1_name, frag2_name
        )
        
        # Create visualization list
        vis_list = [pcd1_world, pcd2_world]
        
        if contact_pcd1 is not None:
            vis_list.append(contact_pcd1)
        if contact_pcd2 is not None:
            vis_list.append(contact_pcd2)
        
        # Add cluster visualizations if requested
        if show_clusters:
            cluster_spheres = self._create_cluster_spheres(frag1_name, frag2_name, show_gt_matches)
            vis_list.extend(cluster_spheres)
        
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
        vis_list.append(coord_frame)
        
        # Create window and visualize
        self._visualize_geometries(vis_list, f"Contact: {frag1_name} <-> {frag2_name}")
    
    def _load_fragment_pcd(self, fragment_name: str) -> Optional[o3d.geometry.PointCloud]:
        """Load fragment point cloud."""
        ply_file = self.ply_dir / f"{fragment_name}.ply"
        
        if not ply_file.exists():
            logger.error(f"PLY file not found: {ply_file}")
            return None
        
        pcd = o3d.io.read_point_cloud(str(ply_file))
        
        # Keep only break surface points
        if fragment_name in self.segment_data:
            break_indices = self.segment_data[fragment_name]['surface_patches'].get('break_0', [])
            if break_indices:
                pcd = pcd.select_by_index(break_indices)
                logger.info(f"Loaded {fragment_name}: {len(pcd.points)} break surface points")
        
        return pcd
    
    def _transform_to_world(self, pcd: o3d.geometry.PointCloud, 
                           fragment_name: str) -> o3d.geometry.PointCloud:
        """Transform point cloud to world coordinates."""
        if self.using_positioned:
            # Fragments are already in correct positions
            return pcd
            
        if fragment_name in self.ground_truth['fragments']:
            transform = np.array(self.ground_truth['fragments'][fragment_name]['transform_matrix'])
            pcd.transform(transform)
        else:
            logger.warning(f"No transform found for {fragment_name}")
        
        return pcd
    
    def _color_fragment_base(self, pcd: o3d.geometry.PointCloud, base_color: List[float]):
        """Apply base color to fragment."""
        colors = np.tile(base_color, (len(pcd.points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    def _identify_contact_regions(self, pcd1: o3d.geometry.PointCloud, 
                                 pcd2: o3d.geometry.PointCloud,
                                 frag1_name: str, frag2_name: str) -> Tuple:
        """Identify and highlight contact regions."""
        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)
        
        # Build KD-trees
        pcd1_tree = o3d.geometry.KDTreeFlann(pcd1)
        pcd2_tree = o3d.geometry.KDTreeFlann(pcd2)
        
        # Find contact points
        contact_mask1 = np.zeros(len(points1), dtype=bool)
        contact_mask2 = np.zeros(len(points2), dtype=bool)
        
        # Check each point in pcd1
        for i in range(len(points1)):
            [k, idx, _] = pcd2_tree.search_radius_vector_3d(pcd1.points[i], self.contact_threshold)
            if k > 0:
                contact_mask1[i] = True
        
        # Check each point in pcd2
        for i in range(len(points2)):
            [k, idx, _] = pcd1_tree.search_radius_vector_3d(pcd2.points[i], self.contact_threshold)
            if k > 0:
                contact_mask2[i] = True
        
        logger.info(f"Contact points: {np.sum(contact_mask1)} in {frag1_name}, "
                   f"{np.sum(contact_mask2)} in {frag2_name}")
        
        # Create contact point clouds with special colors
        contact_pcd1 = None
        contact_pcd2 = None
        
        if np.any(contact_mask1):
            contact_pcd1 = pcd1.select_by_index(np.where(contact_mask1)[0])
            # Color contact region in yellow for fragment 1
            contact_colors1 = np.tile([1.0, 1.0, 0.0], (len(contact_pcd1.points), 1))
            contact_pcd1.colors = o3d.utility.Vector3dVector(contact_colors1)
        
        if np.any(contact_mask2):
            contact_pcd2 = pcd2.select_by_index(np.where(contact_mask2)[0])
            # Color contact region in cyan for fragment 2
            contact_colors2 = np.tile([0.0, 1.0, 1.0], (len(contact_pcd2.points), 1))
            contact_pcd2.colors = o3d.utility.Vector3dVector(contact_colors2)
        
        return contact_pcd1, contact_pcd2
    
    def _create_cluster_spheres(self, frag1_name: str, frag2_name: str, 
                               show_gt_only: bool = True) -> List:
        """Create spheres to visualize cluster positions."""
        spheres = []
        
        if not self.assembly_data:
            return spheres
        
        # Get cluster positions
        cluster_positions = self._get_cluster_positions()
        
        # Find matches for this fragment pair
        for i in range(len(self.assembly_data['cluster_id_1'])):
            if (self.assembly_data['fragment_1'][i] == frag1_name and 
                self.assembly_data['fragment_2'][i] == frag2_name):
                
                is_gt = self.assembly_data['is_ground_truth'][i]
                
                if show_gt_only and not is_gt:
                    continue
                
                # Get cluster IDs
                c1_id = self.assembly_data['cluster_id_1'][i]
                c2_id = self.assembly_data['cluster_id_2'][i]
                
                # Create spheres at cluster positions
                if c1_id in cluster_positions:
                    pos1 = cluster_positions[c1_id]['world_pos']
                    sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
                    sphere1.translate(pos1)
                    
                    # Color based on match type
                    if is_gt:
                        sphere1.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta for GT
                    else:
                        sphere1.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-GT
                    
                    spheres.append(sphere1)
                
                if c2_id in cluster_positions:
                    pos2 = cluster_positions[c2_id]['world_pos']
                    sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=5.0)
                    sphere2.translate(pos2)
                    
                    if is_gt:
                        sphere2.paint_uniform_color([1.0, 0.0, 1.0])  # Magenta for GT
                    else:
                        sphere2.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-GT
                    
                    spheres.append(sphere2)
                
                # Draw line between matching clusters
                if c1_id in cluster_positions and c2_id in cluster_positions:
                    if is_gt:
                        line_points = [pos1, pos2]
                        line = o3d.geometry.LineSet()
                        line.points = o3d.utility.Vector3dVector(line_points)
                        line.lines = o3d.utility.Vector2iVector([[0, 1]])
                        line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0]])  # Magenta
                        spheres.append(line)
        
        logger.info(f"Created {len(spheres)} cluster visualizations")
        return spheres
    
    def _get_cluster_positions(self) -> Dict:
        """Get world positions of all clusters matching the assembly script logic."""
        cluster_positions = {}
        
        # Use the organized clusters by fragment
        for frag_name, clusters in self.clusters_by_fragment.items():
            if frag_name not in self.ground_truth['fragments']:
                continue
                
            transform = np.array(self.ground_truth['fragments'][frag_name]['transform_matrix'])
            
            for cluster in clusters:
                # Transform to world coordinates
                bary_homo = np.append(cluster['barycenter'], 1.0)
                world_pos = (transform @ bary_homo)[:3]
                
                cluster_positions[cluster['cluster_id']] = {
                    'fragment': frag_name,
                    'local_pos': cluster['barycenter'],
                    'world_pos': world_pos,
                    'scale': cluster['scale']
                }
        
        return cluster_positions
    
    def _visualize_geometries(self, geometries: List, window_name: str = "3D Visualization"):
        """Visualize geometries with proper settings."""
        # Use the simpler draw_geometries function which is more compatible
        o3d.visualization.draw_geometries(
            geometries,
            window_name=window_name,
            width=1200,
            height=800,
            left=50,
            top=50,
            point_show_normal=False
        )
    
    def visualize_all_contacts(self, save_images: bool = False):
        """Visualize all contact pairs sequentially."""
        for i, (frag1, frag2) in enumerate(self.contact_pairs):
            print(f"\nVisualizing pair {i+1}/{len(self.contact_pairs)}: {frag1} <-> {frag2}")
            self.visualize_contact_pair(frag1, frag2)
            
            if save_images:
                # TODO: Implement image saving
                pass
    
    def create_summary_visualization(self):
        """Create a summary visualization showing all contacts."""
        logger.info("Creating summary visualization...")
        
        # Load all fragments in world coordinates
        all_pcds = []
        fragment_colors = {}
        color_palette = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for i, frag_name in enumerate(sorted(self.segment_data.keys())):
            pcd = self._load_fragment_pcd(frag_name)
            if pcd is not None:
                pcd_world = self._transform_to_world(pcd, frag_name)
                
                # Assign unique color
                color = color_palette[i % 20][:3]
                fragment_colors[frag_name] = color
                self._color_fragment_base(pcd_world, color)
                
                all_pcds.append(pcd_world)
        
        # Merge all point clouds
        if all_pcds:
            merged_pcd = all_pcds[0]
            for pcd in all_pcds[1:]:
                merged_pcd += pcd
            
            # Add coordinate frame
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
            
            self._visualize_geometries([merged_pcd, coord_frame], "All Fragments Assembly")


def main():
    parser = argparse.ArgumentParser(description="Visualize contact regions between fragments")
    parser.add_argument("--ply_dir", default="Ground_Truth/artifact_1",
                       help="Directory containing PLY files")
    parser.add_argument("--segments", default="output/segmented_fragments.pkl",
                       help="Path to segmented fragments file")
    parser.add_argument("--clusters", default="output/feature_clusters_fixed.pkl",
                       help="Path to cluster file")
    parser.add_argument("--ground_truth", default="Ground_Truth/blender/ground_truth_assembly.json",
                       help="Path to ground truth file")
    parser.add_argument("--assembly", default="output/cluster_assembly_priors_improved.h5",
                       help="Path to assembly results")
    parser.add_argument("--list", action="store_true",
                       help="List all contact pairs")
    parser.add_argument("--pair", nargs=2, metavar=('FRAG1', 'FRAG2'),
                       help="Visualize specific fragment pair")
    parser.add_argument("--all", action="store_true",
                       help="Visualize all contact pairs")
    parser.add_argument("--summary", action="store_true",
                       help="Show summary visualization")
    parser.add_argument("--no-clusters", action="store_true",
                       help="Don't show cluster spheres")
    parser.add_argument("--debug", action="store_true",
                       help="Show debug information")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ContactRegionVisualizer(
        ply_dir=args.ply_dir,
        segments_file=args.segments,
        clusters_file=args.clusters,
        ground_truth_file=args.ground_truth,
        assembly_file=args.assembly
    )
    
    # Debug mode
    if args.debug:
        visualizer.debug_cluster_mapping()
        return
    
    # Execute requested action
    if args.list:
        visualizer.list_contact_pairs()
    elif args.pair:
        visualizer.visualize_contact_pair(
            args.pair[0], args.pair[1], 
            show_clusters=not args.no_clusters
        )
    elif args.all:
        visualizer.visualize_all_contacts()
    elif args.summary:
        visualizer.create_summary_visualization()
    else:
        # Default: list pairs and show instructions
        visualizer.list_contact_pairs()
        print("\nUsage examples:")
        print("  Visualize specific pair:  python visualize_contacts.py --pair frag_1 frag_2")
        print("  Show all pairs:          python visualize_contacts.py --all")
        print("  Summary view:            python visualize_contacts.py --summary")
        print("  Without clusters:        python visualize_contacts.py --pair frag_1 frag_2 --no-clusters")
        print("  Debug mapping:           python visualize_contacts.py --debug")
        print("\nControls:")
        print("  P - Toggle point size")
        print("  Mouse - Rotate/zoom/pan")
        print("  Q/ESC - Close window")


if __name__ == "__main__":
    main()