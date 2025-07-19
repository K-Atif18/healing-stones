#!/usr/bin/env python3
"""
Run assembly extraction with full ground truth only (skip test)
"""

import subprocess
import sys
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and show output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    else:
        print("FAILED!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False

def main():
    logger.info("Running assembly with full ground truth...")
    
    # Check if cluster GT exists
    cluster_gt_file = "Ground_Truth/cluster_level_ground_truth.json"
    
    if not Path(cluster_gt_file).exists():
        logger.info("Cluster GT not found. Extracting it first...")
        
        # Extract cluster-level ground truth
        gt_cmd = [
            sys.executable, "gt_extractor.py",
            "--positioned-dir", "Ground_Truth/reconstructed/artifact_1",
            "--clusters", "output/feature_clusters_fixed.pkl",
            "--segments", "output/segmented_fragments_with_indices.pkl",
            "--output-dir", "Ground_Truth",
            "--contact-threshold", "3.0",
            "--cluster-threshold", "10.0"
        ]
        
        if not run_command(gt_cmd, "Extract cluster-level ground truth"):
            logger.error("Failed to extract ground truth")
            return 1
    else:
        logger.info(f"Using existing cluster GT: {cluster_gt_file}")
        with open(cluster_gt_file, 'r') as f:
            gt_data = json.load(f)
            n_matches = len(gt_data.get('cluster_ground_truth_matches', []))
            logger.info(f"Contains {n_matches} cluster GT matches")
    
    # Run assembly with full cluster GT
    logger.info("\nRunning assembly with full cluster GT...")
    
    assembly_cmd = [
        sys.executable, "assembly.py",
        "--clusters", "output/feature_clusters_fixed.pkl",
        "--segments", "output/segmented_fragments_with_indices.pkl",
        "--ground_truth", cluster_gt_file,
        "--output_dir", "output"
    ]
    
    if run_command(assembly_cmd, "Assembly extraction with cluster GT"):
        # Show results summary
        report_file = "output/assembly_with_cluster_gt_report.txt"
        if Path(report_file).exists():
            logger.info("\n" + "="*60)
            logger.info("FINAL RESULTS SUMMARY")
            logger.info("="*60)
            
            with open(report_file, 'r') as f:
                lines = f.readlines()
                
                # Print summary section
                in_summary = False
                for line in lines:
                    if "SUMMARY:" in line:
                        in_summary = True
                    if in_summary and line.strip():
                        print(line.rstrip())
                    if "MATCHES BY FRAGMENT PAIR:" in line:
                        break
                
                # Count fragment pairs with GT matches
                print("\nFragment pairs with GT matches:")
                pair_count = 0
                for line in lines:
                    if " matches (" in line and " GT)" in line and "0 GT" not in line:
                        print(f"  {line.strip()}")
                        pair_count += 1
                
                print(f"\nTotal pairs with GT matches: {pair_count}")
        
        logger.info("\n✓ Assembly extraction complete!")
        logger.info(f"  Report: {report_file}")
        logger.info(f"  HDF5 output: output/cluster_assembly_with_gt.h5")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())