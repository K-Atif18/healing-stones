#!/usr/bin/env python3
"""
Comprehensive verification and fix for the cluster-fragment mapping pipeline.
This script identifies and fixes the core issues preventing GT matching.
"""

import numpy as np
import pickle
import json
import open3d as o3d
from pathlib import Path
import logging
from typing import Dict, List, Tuple
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineVerifier:
    def __init__(self):
        self.issues = []
        self.fixes_applied = []
        
    def verify_all(self):
        """Run all verification checks."""
        logger.info("Running comprehensive pipeline verification...")
        
        # 1. Check cluster data
        self.verify_cluster_data()
        
        # 2. Check segment data
        self.verify_segment_data()
        
        # 3. Check cluster-segment alignment
        self.verify_cluster_segment_alignment()
        
        # 4. Check coordinate systems
        self.verify_coordinate_systems()
        
        # 5. Check ground truth
        self.verify_ground_truth()
        
        # Report findings
        self.report_findings()
        
    def verify_cluster_data(self):
        """Verify cluster data structure and content."""
        logger.info("\n1. Verifying cluster data...")
        
        try:
            with open('output/feature_clusters.pkl', 'rb') as f:
                cluster_data = pickle.load(f)
            
            n_clusters = len(cluster_data['clusters'])
            logger.info(f"   Total clusters: {n_clusters}")
            
            # Check if clusters have fragment info
            has_fragment_info = any('fragment' in c for c in cluster_data['clusters'])
            if not has_fragment_info:
                self.issues.append("Clusters missing fragment associations")
                logger.error("   ✗ No fragment associations in clusters")
            else:
                logger.info("   ✓ Clusters have fragment associations")
            
            # Check cluster bounds
            if n_clusters > 0:
                barycenters = np.array([c['barycenter'] for c in cluster_data['clusters']])
                logger.info(f"   Cluster bounds: min={barycenters.min(axis=0)}, max={barycenters.max(axis=0)}")
                
        except Exception as e:
            self.issues.append(f"Failed to load cluster data: {e}")
            logger.error(f"   ✗ Failed to load cluster data: {e}")
    
    def verify_segment_data(self):
        """Verify segmentation data."""
        logger.info("\n2. Verifying segmentation data...")
        
        try:
            with open('output/segmented_fragments.pkl', 'rb') as f:
                segment_data = pickle.load(f)
            
            logger.info(f"   Fragments: {len(segment_data)}")
            
            total_clusters_reported = 0
            for frag_name, data in segment_data.items():
                n_clusters = data.get('n_clusters', 0)
                total_clusters_reported += n_clusters
                
                # Check for segment_indices
                if 'segment_indices' not in data:
                    logger.warning(f"   ⚠ {frag_name}: No segment_indices field")
                else:
                    logger.info(f"   ✓ {frag_name}: Has segment_indices")
            
            logger.info(f"   Total clusters reported: {total_clusters_reported}")
            
        except Exception as e:
            self.issues.append(f"Failed to load segment data: {e}")
            logger.error(f"   ✗ Failed to load segment data: {e}")
    
    def verify_cluster_segment_alignment(self):
        """Check if clusters align with segment assignments."""
        logger.info("\n3. Verifying cluster-segment alignment...")
        
        try:
            # Load fixed clusters if available
            cluster_file = 'output/feature_clusters_fixed.pkl'
            if not Path(cluster_file).exists():
                cluster_file = 'output/feature_clusters.pkl'
                
            with open(cluster_file, 'rb') as f:
                cluster_data = pickle.load(f)
                
            with open('output/segmented_fragments.pkl', 'rb') as f:
                segment_data = pickle.load(f)
            
            # Check cluster count alignment
            total_clusters = len(cluster_data['clusters'])
            total_reported = sum(data.get('n_clusters', 0) for data in segment_data.values())
            
            if total_clusters != total_reported:
                self.issues.append(f"Cluster count mismatch: {total_clusters} actual vs {total_reported} reported")
                logger.error(f"   ✗ Cluster count mismatch!")
            else:
                logger.info(f"   ✓ Cluster counts match: {total_clusters}")
                
        except Exception as e:
            logger.error(f"   ✗ Failed to verify alignment: {e}")
    
    def verify_coordinate_systems(self):
        """Check if cluster coordinates match PLY coordinates."""
        logger.info("\n4. Verifying coordinate systems...")
        
        try:
            # Test with first fragment
            fragment_name = "frag_1"
            ply_file = Path(f"Ground_Truth/artifact_1/{fragment_name}.ply")
            
            if not ply_file.exists():
                ply_file = Path(f"Ground_Truth/reconstructed/artifact_1/{fragment_name}.ply")
            
            if ply_file.exists():
                pcd = o3d.io.read_point_cloud(str(ply_file))
                points = np.asarray(pcd.points)
                
                ply_bounds = {
                    'min': points.min(axis=0),
                    'max': points.max(axis=0),
                    'center': points.mean(axis=0)
                }
                logger.info(f"   PLY bounds: {ply_bounds['min']} to {ply_bounds['max']}")
                
                # Check cluster bounds for this fragment
                with open('output/feature_clusters_fixed.pkl', 'rb') as f:
                    cluster_data = pickle.load(f)
                
                frag_clusters = [c for c in cluster_data['clusters'] if c.get('fragment') == fragment_name]
                if frag_clusters:
                    cluster_centers = np.array([c['barycenter'] for c in frag_clusters])
                    cluster_bounds = {
                        'min': cluster_centers.min(axis=0),
                        'max': cluster_centers.max(axis=0)
                    }
                    logger.info(f"   Cluster bounds: {cluster_bounds['min']} to {cluster_bounds['max']}")
                    
                    # Check if clusters are within PLY bounds (with margin)
                    margin = 50  # 50mm margin
                    if (np.all(cluster_bounds['min'] >= ply_bounds['min'] - margin) and
                        np.all(cluster_bounds['max'] <= ply_bounds['max'] + margin)):
                        logger.info("   ✓ Clusters within PLY bounds")
                    else:
                        self.issues.append("Clusters outside PLY bounds - coordinate mismatch")
                        logger.error("   ✗ Clusters outside PLY bounds!")
                        
        except Exception as e:
            logger.error(f"   ✗ Failed to verify coordinates: {e}")
    
    def verify_ground_truth(self):
        """Verify ground truth data availability."""
        logger.info("\n5. Verifying ground truth data...")
        
        gt_files = [
            "Ground_Truth/ground_truth_from_positioned.json",
            "Ground_Truth/cluster_level_ground_truth.json"
        ]
        
        for gt_file in gt_files:
            if Path(gt_file).exists():
                with open(gt_file, 'r') as f:
                    gt_data = json.load(f)
                
                logger.info(f"   Found: {gt_file}")
                
                if 'cluster_ground_truth_matches' in gt_data:
                    n_matches = len(gt_data['cluster_ground_truth_matches'])
                    logger.info(f"   ✓ Has cluster-level GT: {n_matches} matches")
                else:
                    logger.info(f"   ⚠ Only fragment-level GT available")
    
    def report_findings(self):
        """Report all findings and suggest fixes."""
        logger.info("\n" + "="*70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*70)
        
        if self.issues:
            logger.error(f"\nFound {len(self.issues)} issues:")
            for i, issue in enumerate(self.issues, 1):
                logger.error(f"  {i}. {issue}")
        else:
            logger.info("\n✓ No critical issues found!")
        
        logger.info("\nRECOMMENDED FIXES:")
        logger.info("1. Ensure feature_clusters_fixed.pkl is used (with fragment associations)")
        logger.info("2. Generate segment_indices for point-to-cluster mapping")
        logger.info("3. Extract cluster-level ground truth")
        logger.info("4. Verify coordinate system alignment")


class SegmentIndexGenerator:
    """Generate segment indices mapping points to clusters."""
    
    def generate_segment_indices(self):
        """Generate segment_indices for all fragments."""
        logger.info("\nGenerating segment indices for point-to-cluster mapping...")
        
        with open('output/segmented_fragments.pkl', 'rb') as f:
            segment_data = pickle.load(f)
        
        with open('output/feature_clusters_fixed.pkl', 'rb') as f:
            cluster_data = pickle.load(f)
        
        # Organize clusters by fragment
        clusters_by_fragment = {}
        for cluster in cluster_data['clusters']:
            frag = cluster.get('fragment')
            if frag:
                if frag not in clusters_by_fragment:
                    clusters_by_fragment[frag] = []
                clusters_by_fragment[frag].append(cluster)
        
        # Process each fragment
        for frag_name in sorted(segment_data.keys()):
            logger.info(f"\nProcessing {frag_name}...")
            
            # Load PLY file
            ply_file = Path(f"Ground_Truth/artifact_1/{frag_name}.ply")
            if not ply_file.exists():
                ply_file = Path(f"Ground_Truth/reconstructed/artifact_1/{frag_name}.ply")
            
            if not ply_file.exists():
                logger.warning(f"  PLY file not found for {frag_name}")
                continue
            
            pcd = o3d.io.read_point_cloud(str(ply_file))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            # Get break surface points
            green_mask = (colors[:, 1] > 0.6) & (colors[:, 0] < 0.4) & (colors[:, 2] < 0.4)
            break_indices = np.where(green_mask)[0]
            break_points = points[break_indices]
            
            logger.info(f"  Break surface: {len(break_points)} points")
            
            # Initialize segment indices (-1 = unassigned)
            segment_indices = np.full(len(points), -1, dtype=int)
            
            # Get clusters for this fragment
            frag_clusters = clusters_by_fragment.get(frag_name, [])
            logger.info(f"  Clusters: {len(frag_clusters)}")
            
            if len(frag_clusters) > 0 and len(break_points) > 0:
                # Build KD-tree of break points
                tree = cKDTree(break_points)
                
                # Assign break points to nearest cluster
                for local_id, cluster in enumerate(frag_clusters):
                    center = cluster['barycenter']
                    scale = cluster['scale']
                    
                    # Find points within cluster's influence
                    indices = tree.query_ball_point(center, scale * 1.5)
                    
                    # Map back to original indices
                    for idx in indices:
                        original_idx = break_indices[idx]
                        segment_indices[original_idx] = local_id
                
                # Count assignments
                assigned = np.sum(segment_indices >= 0)
                logger.info(f"  Assigned {assigned}/{len(break_points)} break points to clusters")
            
            # Update segment data
            segment_data[frag_name]['segment_indices'] = segment_indices
            segment_data[frag_name]['n_assigned'] = int(np.sum(segment_indices >= 0))
        
        # Save updated segment data
        with open('output/segmented_fragments_with_indices.pkl', 'wb') as f:
            pickle.dump(segment_data, f)
        
        logger.info("\nSaved segmented_fragments_with_indices.pkl")
        return segment_data


def create_simple_cluster_gt():
    """Create a simple cluster-level ground truth for testing."""
    logger.info("\nCreating simple cluster-level ground truth...")
    
    # Load required data
    with open('output/feature_clusters_fixed.pkl', 'rb') as f:
        cluster_data = pickle.load(f)
    
    with open('Ground_Truth/ground_truth_from_positioned.json', 'r') as f:
        gt_data = json.load(f)
    
    # Organize clusters by fragment
    clusters_by_fragment = {}
    for cluster in cluster_data['clusters']:
        frag = cluster.get('fragment')
        if frag:
            if frag not in clusters_by_fragment:
                clusters_by_fragment[frag] = []
            clusters_by_fragment[frag].append(cluster)
    
    # Create cluster matches based on proximity
    cluster_matches = []
    
    for contact in gt_data.get('contact_details', []):
        frag1 = contact['fragment_1']
        frag2 = contact['fragment_2']
        
        clusters1 = clusters_by_fragment.get(frag1, [])
        clusters2 = clusters_by_fragment.get(frag2, [])
        
        if not clusters1 or not clusters2:
            continue
        
        # Find closest cluster pairs
        contact_center1 = np.array(contact['contact_center_1'])
        contact_center2 = np.array(contact['contact_center_2'])
        
        # Find clusters near contact centers
        near_clusters1 = []
        near_clusters2 = []
        
        for i, c in enumerate(clusters1):
            dist = np.linalg.norm(c['barycenter'] - contact_center1)
            if dist < 50:  # 50mm threshold
                near_clusters1.append((i, dist))
        
        for i, c in enumerate(clusters2):
            dist = np.linalg.norm(c['barycenter'] - contact_center2)
            if dist < 50:
                near_clusters2.append((i, dist))
        
        # Create matches for nearest clusters
        if near_clusters1 and near_clusters2:
            # Sort by distance
            near_clusters1.sort(key=lambda x: x[1])
            near_clusters2.sort(key=lambda x: x[1])
            
            # Create match for closest pair
            match = {
                'fragment_1': frag1,
                'fragment_2': frag2,
                'cluster_id_1': near_clusters1[0][0],
                'cluster_id_2': near_clusters2[0][0],
                'confidence': 0.8,
                'distance': near_clusters1[0][1] + near_clusters2[0][1]
            }
            cluster_matches.append(match)
    
    # Create simple GT file
    simple_gt = {
        'fragments': gt_data['fragments'],
        'contact_pairs': gt_data['contact_pairs'],
        'cluster_ground_truth_matches': cluster_matches,
        'extraction_info': {
            'method': 'simple_proximity',
            'threshold': 50.0
        }
    }
    
    with open('Ground_Truth/simple_cluster_gt.json', 'w') as f:
        json.dump(simple_gt, f, indent=2)
    
    logger.info(f"Created simple cluster GT with {len(cluster_matches)} matches")
    return simple_gt


def main():
    """Run verification and fixes."""
    logger.info("Starting pipeline verification and fixes...")
    
    # 1. Verify current state
    verifier = PipelineVerifier()
    verifier.verify_all()
    
    # 2. Apply fixes if needed
    fixes_needed = input("\nApply fixes? (y/n): ").lower() == 'y'
    
    if fixes_needed:
        # Generate segment indices
        generator = SegmentIndexGenerator()
        generator.generate_segment_indices()
        
        # Create simple cluster GT for testing
        create_simple_cluster_gt()
        
        logger.info("\n" + "="*70)
        logger.info("FIXES APPLIED")
        logger.info("="*70)
        logger.info("\n1. Generated segment_indices in segmented_fragments_with_indices.pkl")
        logger.info("2. Created simple_cluster_gt.json for testing")
        logger.info("\nNext steps:")
        logger.info("1. Update assembly script to use segmented_fragments_with_indices.pkl")
        logger.info("2. Test with simple_cluster_gt.json first")
        logger.info("3. Then run full cluster GT extraction")


if __name__ == "__main__":
    main()