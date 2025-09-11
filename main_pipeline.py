#!/usr/bin/env python3
"""
Main pipeline for Mayan Stele Fragment Reconstruction

This script orchestrates the entire reconstruction process from PLY files
with colored break surfaces to a final reconstructed artifact.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import time

# Import our custom modules
from ply_loader import PLYColorExtractor
from feature_extractor import BreakSurfaceFeatureExtractor
from surface_matcher import SurfaceMatcher
from fragment_aligner import FragmentAligner
from reconstruction_visualizer import ReconstructionVisualizer

class ReconstructionPipeline:
    """
    Main pipeline for fragment reconstruction
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'color_tolerance': 0.3,
            'min_cluster_size': 50,
            'min_similarity': 0.6,
            'use_optimal_matching': True,
            'visualize_steps': False,
            'save_intermediate': True,
            'output_reports': True
        }
        
        if config:
            self.config.update(config)
        
        # Initialize components
        self.ply_extractor = PLYColorExtractor()
        self.feature_extractor = BreakSurfaceFeatureExtractor()
        self.surface_matcher = SurfaceMatcher()
        self.fragment_aligner = FragmentAligner()
        self.visualizer = ReconstructionVisualizer()
        
        # Pipeline data
        self.fragments = []
        self.enhanced_fragments = []
        self.all_matches = {}
        self.transformations = {}
        self.quality_metrics = {}
    
    def load_and_process_fragments(self, input_directory):
        """Step 1: Load PLY files and extract break surfaces"""
        print("="*60)
        print("STEP 1: Loading PLY files and extracting break surfaces")
        print("="*60)
        
        start_time = time.time()
        
        # Load all fragments
        print(f"Loading fragments from: {input_directory}")
        self.fragments = self.ply_extractor.process_all_fragments(input_directory)
        
        if not self.fragments:
            raise ValueError(f"No valid PLY files found in {input_directory}")
        
        # Print summary
        print(f"\nLoaded {len(self.fragments)} fragments:")
        total_surfaces = {'blue': 0, 'green': 0, 'red': 0}
        
        for i, fragment in enumerate(self.fragments):
            print(f"  Fragment {i}: {Path(fragment['filepath']).name}")
            for color, surfaces in fragment['break_surfaces'].items():
                count = len(surfaces)
                total_surfaces[color] += count
                if count > 0:
                    print(f"    {color}: {count} surfaces")
        
        print(f"\nTotal break surfaces found:")
        for color, count in total_surfaces.items():
            print(f"  {color}: {count}")
        
        elapsed = time.time() - start_time
        print(f"\nStep 1 completed in {elapsed:.2f} seconds")
        
        return self.fragments
    
    def extract_features(self):
        """Step 2: Extract geometric features from break surfaces"""
        print("\n" + "="*60)
        print("STEP 2: Extracting geometric features")
        print("="*60)
        
        start_time = time.time()
        
        print("Computing geometric features for all break surfaces...")
        self.enhanced_fragments = self.feature_extractor.extract_all_features(self.fragments)
        
        # Print feature summary
        total_features = 0
        for fragment in self.enhanced_fragments:
            for color, features in fragment['features'].items():
                total_features += len(features)
        
        print(f"Extracted features for {total_features} break surfaces")
        
        elapsed = time.time() - start_time
        print(f"Step 2 completed in {elapsed:.2f} seconds")
        
        return self.enhanced_fragments
    
    def find_surface_matches(self):
        """Step 3: Find matching break surfaces between fragments"""
        print("\n" + "="*60)
        print("STEP 3: Finding surface matches")
        print("="*60)
        
        start_time = time.time()
        
        print(f"Searching for surface matches (min similarity: {self.config['min_similarity']})...")
        
        self.all_matches = self.surface_matcher.find_all_matches(
            self.enhanced_fragments,
            min_similarity=self.config['min_similarity'],
            use_optimal=self.config['use_optimal_matching']
        )
        
        # Print match summary
        total_matches = 0
        match_summary = {'blue': 0, 'green': 0, 'red': 0}
        
        for pair_key, color_matches in self.all_matches.items():
            for color, matches in color_matches.items():
                count = len(matches)
                total_matches += count
                match_summary[color] += count
        
        print(f"\nFound {total_matches} total matches:")
        for color, count in match_summary.items():
            print(f"  {color}: {count}")
        
        if total_matches == 0:
            print("WARNING: No matches found! Consider lowering min_similarity threshold.")
        
        elapsed = time.time() - start_time
        print(f"Step 3 completed in {elapsed:.2f} seconds")
        
        return self.all_matches
    
    def align_fragments(self):
        """Step 4: Align fragments based on surface matches"""
        print("\n" + "="*60)
        print("STEP 4: Aligning fragments")
        print("="*60)
        
        start_time = time.time()
        
        if not self.all_matches:
            print("No matches found - skipping alignment")
            return {}
        
        # Build alignment graph and find connected components
        alignment_graph = self.build_alignment_graph()
        connected_components = self.find_connected_components(alignment_graph)
        
        print(f"Found {len(connected_components)} connected component(s)")
        
        # Process each connected component
        self.transformations = {}
        component_metrics = []
        
        for i, component in enumerate(connected_components):
            print(f"\nProcessing component {i+1} with {len(component)} fragments:")
            print(f"  Fragments: {component}")
            
            if len(component) == 1:
                # Single fragment - use identity transformation
                self.transformations[component[0]] = np.eye(4)
                continue
            
            # Align fragments in this component
            component_transforms, metrics = self.align_component(component)
            
            # Add to global transformations
            self.transformations.update(component_transforms)
            component_metrics.append(metrics)
            
            print(f"  Component alignment quality: {metrics.get('mean_distance_error', 'N/A')}")
        
        # Compute overall quality metrics
        self.quality_metrics = self.compute_overall_quality(component_metrics)
        
        elapsed = time.time() - start_time
        print(f"\nStep 4 completed in {elapsed:.2f} seconds")
        print(f"Aligned {len(self.transformations)} fragments")
        
        return self.transformations
    
    def build_alignment_graph(self):
        """Build graph of fragment connections based on matches"""
        graph = {}
        
        # Initialize graph with all fragments
        for i in range(len(self.enhanced_fragments)):
            graph[i] = set()
        
        # Add edges for matched fragments
        for pair_key, color_matches in self.all_matches.items():
            # Extract fragment indices from pair key
            parts = pair_key.split('_')
            frag1_idx = int(parts[1])
            frag2_idx = int(parts[3])
            
            # Check if there are any matches for this pair
            has_matches = any(len(matches) > 0 for matches in color_matches.values())
            
            if has_matches:
                graph[frag1_idx].add(frag2_idx)
                graph[frag2_idx].add(frag1_idx)
        
        return graph
    
    def find_connected_components(self, graph):
        """Find connected components in the alignment graph"""
        visited = set()
        components = []
        
        for node in graph:
            if node not in visited:
                component = []
                stack = [node]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        stack.extend(graph[current] - visited)
                
                components.append(sorted(component))
        
        return components
    
    def align_component(self, fragment_indices):
        """Align fragments within a connected component"""
        if len(fragment_indices) < 2:
            return {fragment_indices[0]: np.eye(4)}, {}
        
        # Start with first fragment as reference (identity transform)
        transforms = {fragment_indices[0]: np.eye(4)}
        
        # Process remaining fragments
        remaining = set(fragment_indices[1:])
        aligned = {fragment_indices[0]}
        
        while remaining:
            best_match = None
            best_similarity = 0
            best_alignment = None
            
            # Find best match between aligned and remaining fragments
            for aligned_idx in aligned:
                for remaining_idx in remaining:
                    # Get matches between these fragments
                    pair_key = f"fragment_{min(aligned_idx, remaining_idx)}_to_{max(aligned_idx, remaining_idx)}"
                    
                    if pair_key not in self.all_matches:
                        continue
                    
                    # Collect all matches for this pair
                    all_pair_matches = []
                    for color_matches in self.all_matches[pair_key].values():
                        all_pair_matches.extend(color_matches)
                    
                    if not all_pair_matches:
                        continue
                    
                    # Get best similarity
                    max_sim = max(match['similarity'] for match in all_pair_matches)
                    
                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_match = (aligned_idx, remaining_idx)
                        best_alignment = all_pair_matches
            
            if best_match is None:
                # No more connections - add remaining fragments with identity
                for idx in remaining:
                    transforms[idx] = np.eye(4)
                break
            
            # Align the best match
            aligned_idx, remaining_idx = best_match
            fragment1 = self.enhanced_fragments[aligned_idx]
            fragment2 = self.enhanced_fragments[remaining_idx]
            
            # Compute alignment
            transform, metrics = self.fragment_aligner.optimize_multi_surface_alignment(
                fragment1, fragment2, best_alignment
            )
            
            # Apply relative to already aligned fragment
            if aligned_idx in transforms:
                absolute_transform = transforms[aligned_idx] @ transform
            else:
                absolute_transform = transform
            
            transforms[remaining_idx] = absolute_transform
            
            # Update sets
            aligned.add(remaining_idx)
            remaining.remove(remaining_idx)
        
        # Compute component quality metrics
        component_metrics = self.evaluate_component_alignment(fragment_indices, transforms)
        
        return transforms, component_metrics
    
    def evaluate_component_alignment(self, fragment_indices, transforms):
        """Evaluate alignment quality for a component"""
        if len(fragment_indices) < 2:
            return {}
        
        total_errors = []
        
        # Evaluate all pairwise alignments in component
        for i in range(len(fragment_indices)):
            for j in range(i + 1, len(fragment_indices)):
                idx1, idx2 = fragment_indices[i], fragment_indices[j]
                
                # Get matches between these fragments
                pair_key = f"fragment_{min(idx1, idx2)}_to_{max(idx1, idx2)}"
                
                if pair_key in self.all_matches:
                    all_matches = []
                    for color_matches in self.all_matches[pair_key].values():
                        all_matches.extend(color_matches)
                    
                    if all_matches:
                        fragment1 = self.enhanced_fragments[idx1]
                        fragment2 = self.enhanced_fragments[idx2]
                        
                        # Compute relative transform
                        relative_transform = np.linalg.inv(transforms[idx1]) @ transforms[idx2]
                        
                        # Evaluate alignment
                        metrics = self.fragment_aligner.evaluate_alignment_quality(
                            fragment1, fragment2, relative_transform, all_matches
                        )
                        
                        if 'mean_distance_error' in metrics:
                            total_errors.append(metrics['mean_distance_error'])
        
        if total_errors:
            return {
                'mean_distance_error': np.mean(total_errors),
                'std_distance_error': np.std(total_errors),
                'max_distance_error': np.max(total_errors)
            }
        else:
            return {}
    
    def compute_overall_quality(self, component_metrics):
        """Compute overall reconstruction quality"""
        if not component_metrics:
            return {}
        
        all_errors = []
        for metrics in component_metrics:
            if 'mean_distance_error' in metrics:
                all_errors.append(metrics['mean_distance_error'])
        
        if all_errors:
            return {
                'overall_mean_error': np.mean(all_errors),
                'overall_std_error': np.std(all_errors),
                'overall_max_error': np.max(all_errors),
                'num_aligned_fragments': len(self.transformations),
                'total_fragments': len(self.fragments)
            }
        else:
            return {'num_aligned_fragments': len(self.transformations)}
    
    def visualize_and_report(self, output_directory):
        """Step 5: Create visualizations and reports"""
        print("\n" + "="*60)
        print("STEP 5: Creating visualizations and reports")
        print("="*60)
        
        start_time = time.time()
        
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)
        
        if self.config['visualize_steps']:
            print("Showing step-by-step visualizations...")
            self.visualizer.visualize_original_fragments(self.fragments)
            
            if self.all_matches:
                # Show some example matches
                for pair_key, color_matches in list(self.all_matches.items())[:2]:
                    for color, matches in color_matches.items():
                        if matches:
                            parts = pair_key.split('_')
                            idx1, idx2 = int(parts[1]), int(parts[3])
                            self.visualizer.visualize_surface_matches(
                                self.enhanced_fragments[idx1],
                                self.enhanced_fragments[idx2],
                                matches[:3]  # Show top 3 matches
                            )
                            break
                    break
        
        # Always show final reconstruction
        if self.transformations:
            print("Showing final reconstruction...")
            final_mesh = self.visualizer.visualize_final_reconstruction(
                self.fragments, self.transformations
            )
        
        if self.config['output_reports']:
            print("Generating reports...")
            
            # Match quality report
            if self.all_matches:
                self.visualizer.create_match_quality_report(
                    self.all_matches, 
                    output_dir / "match_quality_report.png"
                )
            
            # Reconstruction report
            if self.transformations:
                self.visualizer.create_reconstruction_report(
                    self.fragments, self.transformations, self.quality_metrics,
                    output_dir / "reconstruction_report.png"
                )
        
        # Save reconstruction
        if self.transformations:
            print("Saving reconstruction...")
            self.visualizer.save_reconstruction(
                self.fragments, self.transformations, output_dir / "reconstruction"
            )
        
        elapsed = time.time() - start_time
        print(f"Step 5 completed in {elapsed:.2f} seconds")
    
    def save_pipeline_data(self, output_directory):
        """Save intermediate pipeline data"""
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)
        
        if self.config['save_intermediate']:
            print("Saving intermediate data...")
            
            # Save fragment data
            self.ply_extractor.save_fragment_data(
                self.fragments, output_dir / "fragment_data.json"
            )
            
            # Save matches
            if self.all_matches:
                # Convert matches to serializable format
                serializable_matches = {}
                for pair_key, color_matches in self.all_matches.items():
                    serializable_matches[pair_key] = {}
                    for color, matches in color_matches.items():
                        serializable_matches[pair_key][color] = []
                        for match in matches:
                            # Remove non-serializable data
                            clean_match = {k: v for k, v in match.items() 
                                         if k not in ['surface1_features', 'surface2_features']}
                            serializable_matches[pair_key][color].append(clean_match)
                
                with open(output_dir / "surface_matches.json", 'w') as f:
                    json.dump(serializable_matches, f, indent=2)
            
            # Save transformations
            if self.transformations:
                transform_data = {}
                for i, transform in self.transformations.items():
                    transform_data[f"fragment_{i}"] = transform.tolist()
                
                with open(output_dir / "transformations.json", 'w') as f:
                    json.dump(transform_data, f, indent=2)
            
            # Save quality metrics
            with open(output_dir / "quality_metrics.json", 'w') as f:
                json.dump(self.quality_metrics, f, indent=2)
    
    def run_full_pipeline(self, input_directory, output_directory):
        """Run the complete reconstruction pipeline"""
        print("MAYAN STELE FRAGMENT RECONSTRUCTION PIPELINE")
        print("=" * 60)
        print(f"Input directory: {input_directory}")
        print(f"Output directory: {output_directory}")
        print(f"Configuration: {self.config}")
        print()
        
        overall_start_time = time.time()
        
        try:
            # Step 1: Load and process fragments
            self.load_and_process_fragments(input_directory)
            
            # Step 2: Extract features
            self.extract_features()
            
            # Step 3: Find surface matches
            self.find_surface_matches()
            
            # Step 4: Align fragments
            self.align_fragments()
            
            # Step 5: Visualize and report
            self.visualize_and_report(output_directory)
            
            # Save all data
            self.save_pipeline_data(output_directory)
            
            # Final summary
            total_time = time.time() - overall_start_time
            print("\n" + "="*60)
            print("RECONSTRUCTION PIPELINE COMPLETED")
            print("="*60)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Fragments processed: {len(self.fragments)}")
            print(f"Fragments aligned: {len(self.transformations)}")
            print(f"Overall quality: {self.quality_metrics.get('overall_mean_error', 'N/A')}")
            print(f"Results saved to: {output_directory}")
            
            return True
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Command line interface for the reconstruction pipeline"""
    parser = argparse.ArgumentParser(
        description="Mayan Stele Fragment Reconstruction Pipeline"
    )
    
    parser.add_argument(
        "input_dir",
        help="Directory containing PLY files with colored break surfaces"
    )
    
    parser.add_argument(
        "output_dir",
        help="Directory to save reconstruction results"
    )
    
    parser.add_argument(
        "--min-similarity", 
        type=float, 
        default=0.6,
        help="Minimum similarity threshold for surface matching (default: 0.6)"
    )
    
    parser.add_argument(
        "--color-tolerance",
        type=float,
        default=0.3,
        help="Color matching tolerance (default: 0.3)"
    )
    
    parser.add_argument(
        "--visualize-steps",
        action="store_true",
        help="Show step-by-step visualizations"
    )
    
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip generating detailed reports"
    )
    
    parser.add_argument(
        "--config",
        help="JSON configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'min_similarity': args.min_similarity,
        'color_tolerance': args.color_tolerance,
        'visualize_steps': args.visualize_steps,
        'output_reports': not args.no_reports
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    # Run pipeline
    pipeline = ReconstructionPipeline(config)
    success = pipeline.run_full_pipeline(args.input_dir, args.output_dir)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()