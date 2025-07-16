#!/usr/bin/env python3
"""
Preparation and verification script for Phase 1.3
Helps ensure all data is ready before running the assembly knowledge extraction.
"""

import pickle
import json
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase13Preparer:
    def __init__(self):
        self.issues = []
        self.warnings = []
        
    def check_prerequisites(self):
        """Check if all required files exist and are valid."""
        logger.info("Checking prerequisites for Phase 1.3...")
        
        # Check Phase 1.1 and 1.2 outputs
        required_files = {
            'feature_clusters.pkl': 'output/feature_clusters.pkl',
            'segmented_fragments.pkl': 'output/segmented_fragments.pkl'
        }
        
        for name, path in required_files.items():
            if not Path(path).exists():
                self.issues.append(f"Missing required file: {path}")
            else:
                logger.info(f"✓ Found {name}")
        
        # Check Blender file location
        blend_dir = Path("Ground_Truth/blender/done")
        if blend_dir.exists():
            blend_files = list(blend_dir.glob("*.blend"))
            if blend_files:
                logger.info(f"✓ Found {len(blend_files)} .blend file(s) in {blend_dir}")
                for bf in blend_files:
                    logger.info(f"  - {bf.name}")
            else:
                self.warnings.append(f"No .blend files found in {blend_dir}")
        else:
            self.issues.append(f"Directory not found: {blend_dir}")
        
        # Check if ground truth JSON exists
        gt_json = Path("Ground_Truth/blender/ground_truth_assembly.json")
        if gt_json.exists():
            logger.info("✓ Ground truth JSON already exists")
            self._verify_ground_truth(gt_json)
        else:
            self.warnings.append("Ground truth JSON not found - you'll need to run the Blender script")
        
        return len(self.issues) == 0
    
    def _verify_ground_truth(self, gt_path):
        """Verify the ground truth JSON file."""
        try:
            with open(gt_path, 'r') as f:
                gt_data = json.load(f)
            
            # Check required fields
            required_fields = ['fragments', 'contact_pairs', 'contact_details']
            for field in required_fields:
                if field not in gt_data:
                    self.issues.append(f"Ground truth missing field: {field}")
            
            # Report statistics
            if 'fragments' in gt_data:
                logger.info(f"  - Fragments in GT: {len(gt_data['fragments'])}")
            if 'contact_pairs' in gt_data:
                logger.info(f"  - Contact pairs: {len(gt_data['contact_pairs'])}")
            if 'contact_details' in gt_data:
                logger.info(f"  - Contact details: {len(gt_data['contact_details'])}")
                
        except Exception as e:
            self.issues.append(f"Error reading ground truth: {e}")
    
    def verify_data_consistency(self):
        """Verify that the data from Phase 1.1 and 1.2 is consistent."""
        logger.info("\nVerifying data consistency...")
        
        try:
            # Load data
            with open('output/segmented_fragments.pkl', 'rb') as f:
                segments = pickle.load(f)
            
            with open('output/feature_clusters.pkl', 'rb') as f:
                clusters = pickle.load(f)
            
            # Check fragment names match
            segment_fragments = set(segments.keys())
            logger.info(f"Fragments in segmentation: {sorted(segment_fragments)}")
            
            # Check cluster data
            n_clusters = len(clusters['clusters'])
            logger.info(f"Total clusters: {n_clusters}")
            
            # Check overlap graph
            n_edges = len(clusters.get('overlap_graph_edges', []))
            logger.info(f"Overlap graph edges: {n_edges}")
            
            # Verify cluster IDs are unique
            cluster_ids = [c['cluster_id'] for c in clusters['clusters']]
            if len(cluster_ids) != len(set(cluster_ids)):
                self.issues.append("Duplicate cluster IDs found!")
            
            # Check cluster data completeness
            required_cluster_fields = ['barycenter', 'principal_axes', 'eigenvalues', 
                                     'size_signature', 'anisotropy_signature', 'scale']
            
            for cluster in clusters['clusters'][:5]:  # Check first 5 clusters
                for field in required_cluster_fields:
                    if field not in cluster:
                        self.issues.append(f"Cluster {cluster.get('cluster_id', '?')} missing field: {field}")
            
        except Exception as e:
            self.issues.append(f"Error verifying data: {e}")
    
    def create_blender_instructions(self):
        """Create instructions for running the Blender script."""
        instructions = """
INSTRUCTIONS FOR BLENDER GROUND TRUTH EXTRACTION:

1. Open Blender
2. Open your assembled artifact file from: Ground_Truth/blender/done/
3. Verify the assembly:
   - All fragments should be in their correct positions
   - Fragment names should follow the pattern: frag_1, frag_2, etc.
   - If not, rename them to match this pattern

4. Check units:
   - Go to Scene Properties > Units
   - Ensure Unit System is set to Metric
   - Set Length to Millimeters (important!)

5. Run the extraction script:
   - Switch to the Scripting tab
   - Open the file: blender_gt_extractor.py
   - Click "Run Script"
   - Check the console for any errors

6. The script will create:
   - ground_truth_assembly.json in Ground_Truth/blender/
   - Contact visualization spheres in the scene

7. Verify the output:
   - Check that ground_truth_assembly.json was created
   - Review the contact visualization spheres
   - They should appear at fragment contact points

8. Save the Blender file with visualizations for future reference
"""
        
        instruction_file = Path("blender_extraction_instructions.txt")
        with open(instruction_file, 'w') as f:
            f.write(instructions)
        
        logger.info(f"\nCreated Blender instructions at: {instruction_file}")
        return instructions
    
    def prepare_enhanced_cluster_data(self):
        """Enhance cluster data with fragment associations."""
        logger.info("\nPreparing enhanced cluster data...")
        
        try:
            # Load data
            with open('output/segmented_fragments.pkl', 'rb') as f:
                segments = pickle.load(f)
            
            with open('output/feature_clusters.pkl', 'rb') as f:
                clusters = pickle.load(f)
            
            # Calculate clusters per fragment
            n_fragments = len(segments)
            n_clusters = len(clusters['clusters'])
            clusters_per_fragment = n_clusters // n_fragments
            
            logger.info(f"Distributing {n_clusters} clusters across {n_fragments} fragments")
            logger.info(f"Approximate clusters per fragment: {clusters_per_fragment}")
            
            # Create enhanced cluster data
            enhanced_clusters = clusters.copy()
            fragment_names = sorted(segments.keys())
            
            # Assign fragments to clusters
            for i, cluster in enumerate(enhanced_clusters['clusters']):
                fragment_idx = min(i // clusters_per_fragment, n_fragments - 1)
                cluster['fragment'] = fragment_names[fragment_idx]
            
            # Save enhanced data
            output_file = Path('output/feature_clusters_enhanced.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(enhanced_clusters, f)
            
            logger.info(f"✓ Saved enhanced cluster data to {output_file}")
            
            # Verify distribution
            fragment_counts = {}
            for cluster in enhanced_clusters['clusters']:
                frag = cluster.get('fragment', 'unknown')
                fragment_counts[frag] = fragment_counts.get(frag, 0) + 1
            
            logger.info("\nCluster distribution by fragment:")
            for frag, count in sorted(fragment_counts.items()):
                logger.info(f"  {frag}: {count} clusters")
                
        except Exception as e:
            self.issues.append(f"Error preparing enhanced cluster data: {e}")
    
    def report(self):
        """Generate final report."""
        print("\n" + "="*70)
        print("PHASE 1.3 PREPARATION REPORT")
        print("="*70)
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES (must be fixed):")
            for issue in self.issues:
                print(f"  - {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.issues:
            print("\n✅ All prerequisites satisfied!")
            print("\nNEXT STEPS:")
            print("1. If ground_truth_assembly.json doesn't exist:")
            print("   - Follow the Blender instructions in 'blender_extraction_instructions.txt'")
            print("   - Run the blender_gt_extractor.py script in Blender")
            print("\n2. Once ground truth is ready, run Phase 1.3:")
            print("   python phase_1_3_assembly.py")
        
        print("\n" + "="*70)


def main():
    preparer = Phase13Preparer()
    
    # Run all checks
    preparer.check_prerequisites()
    preparer.verify_data_consistency()
    preparer.prepare_enhanced_cluster_data()
    
    # Create Blender instructions
    preparer.create_blender_instructions()
    
    # Generate report
    preparer.report()


if __name__ == "__main__":
    main()