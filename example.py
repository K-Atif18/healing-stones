# example_basic_usage.py
"""
Basic usage example for Mayan Stele Reconstruction System
This script demonstrates the simplest way to reconstruct fragments
"""

import sys
from pathlib import Path
from main_pipeline import ReconstructionPipeline

def main():
    # Input and output directories
    input_dir = "data/input"  # Directory containing PLY files
    output_dir = "data/output"  # Where to save results
    
    print("Mayan Stele Reconstruction - Basic Example")
    print("=" * 50)
    
    # Check if input directory exists
    if not Path(input_dir).exists():
        print(f"Error: Input directory '{input_dir}' not found")
        print("Please create the directory and add your PLY files")
        return False
    
    # Check for PLY files
    ply_files = list(Path(input_dir).glob("*.ply"))
    if not ply_files:
        print(f"Error: No PLY files found in '{input_dir}'")
        print("Please add PLY files with colored break surfaces")
        return False
    
    print(f"Found {len(ply_files)} PLY files:")
    for ply_file in ply_files:
        print(f"  - {ply_file.name}")
    
    # Create reconstruction pipeline with default settings
    print("\\nInitializing reconstruction pipeline...")
    pipeline = ReconstructionPipeline()
    
    # Run the complete reconstruction
    print("\\nStarting reconstruction...")
    success = pipeline.run_full_pipeline(input_dir, output_dir)
    
    if success:
        print("\\n✓ Reconstruction completed successfully!")
        print(f"Results saved to: {output_dir}")
        
        # Print summary
        print("\\nReconstruction Summary:")
        print(f"  Total fragments: {len(pipeline.fragments)}")
        print(f"  Aligned fragments: {len(pipeline.transformations)}")
        
        if pipeline.quality_metrics:
            if 'overall_mean_error' in pipeline.quality_metrics:
                print(f"  Mean alignment error: {pipeline.quality_metrics['overall_mean_error']:.4f}")
        
        # List output files
        output_path = Path(output_dir)
        if (output_path / "reconstruction").exists():
            print("\\nOutput files:")
            for file in (output_path / "reconstruction").iterdir():
                print(f"  - {file.name}")
        
        return True
    else:
        print("\\n✗ Reconstruction failed")
        print("Check the error messages above for details")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# example_advanced_usage.py
"""
Advanced usage example with custom configuration
"""

from main_pipeline import ReconstructionPipeline
from config_manager import ConfigurationManager, ReconstructionConfig
import numpy as np

def create_custom_config():
    """Create a custom configuration for challenging fragments"""
    
    config_manager = ConfigurationManager()
    
    # Start with high quality template
    config = config_manager.get_template_config('high_quality')
    
    # Customize for specific needs
    config.color_extraction.color_tolerance = 0.25  # Moderate color tolerance
    config.color_extraction.min_cluster_size = 30   # Smaller surface clusters
    
    # Emphasize surface normals and curvature for matching
    config.matching.similarity_weights = {
        'normal_similarity': 0.35,      # Higher weight on normals
        'curvature_similarity': 0.25,   # Higher weight on curvature
        'area_similarity': 0.15,
        'shape_similarity': 0.15,
        'boundary_similarity': 0.05,
        'size_similarity': 0.05
    }
    
    # More aggressive alignment
    config.alignment.icp_max_iterations = 150
    config.alignment.enable_global_optimization = True
    
    # Enhanced visualization
    config.visualization.show_step_by_step = True
    config.visualization.save_images = True
    
    return config

def main():
    input_dir = "data/input"
    output_dir = "data/advanced_output"
    
    print("Advanced Reconstruction Example")
    print("=" * 40)
    
    # Create custom configuration
    print("Creating custom configuration...")
    custom_config = create_custom_config()
    
    # Save configuration for reference
    config_manager = ConfigurationManager()
    config_manager.save_config(custom_config, "configs/custom_advanced.json")
    print("Custom configuration saved to: configs/custom_advanced.json")
    
    # Create pipeline with custom config
    pipeline = ReconstructionPipeline(custom_config.__dict__)
    
    # Run reconstruction with detailed progress reporting
    success = pipeline.run_full_pipeline(input_dir, output_dir)
    
    if success:
        print("\\n✓ Advanced reconstruction completed!")
        
        # Detailed analysis
        if pipeline.all_matches:
            total_matches = sum(
                len(matches) for color_matches in pipeline.all_matches.values()
                for matches in color_matches.values()
            )
            print(f"Total surface matches found: {total_matches}")
            
            # Match quality distribution
            all_similarities = []
            for color_matches in pipeline.all_matches.values():
                for matches in color_matches.values():
                    for match in matches:
                        all_similarities.append(match['similarity'])
            
            if all_similarities:
                print(f"Match quality - Mean: {np.mean(all_similarities):.3f}, "
                      f"Std: {np.std(all_similarities):.3f}")
        
        print(f"Results saved to: {output_dir}")
    else:
        print("\\n✗ Advanced reconstruction failed")

if __name__ == "__main__":
    main()

# example_batch_comparison.py
"""
Example: Compare different configurations on the same dataset
"""

from batch_processor import BatchProcessor, BatchJob
from config_manager import ConfigurationManager
from reconstruction_evaluator import ReconstructionEvaluator

def main():
    input_dir = "data/input"
    batch_output_dir = "data/batch_comparison"
    
    print("Batch Configuration Comparison Example")
    print("=" * 45)
    
    # Setup batch processor
    processor = BatchProcessor(max_workers=2)  # Adjust based on your system
    config_manager = ConfigurationManager()
    
    # Define configurations to compare
    configs_to_test = {
        'default': config_manager.get_template_config('default'),
        'high_quality': config_manager.get_template_config('high_quality'),
        'fast_processing': config_manager.get_template_config('fast_processing'),
        'conservative': config_manager.get_template_config('conservative')
    }
    
    print(f"Comparing {len(configs_to_test)} configurations:")
    for config_name in configs_to_test.keys():
        print(f"  - {config_name}")
    
    # Add comparative jobs
    processor.add_comparative_jobs(
        input_directory=input_dir,
        base_output_dir=batch_output_dir,
        configs=configs_to_test,
        description_prefix="config_comparison"
    )
    
    print(f"\\nRunning {len(processor.job_queue)} reconstruction jobs...")
    
    # Run batch processing
    results = processor.run_batch_parallel()
    
    # Save results
    results_file = f"{batch_output_dir}/batch_results.json"
    processor.save_batch_results(results, results_file)
    
    print(f"\\nBatch processing completed:")
    print(f"  Successful jobs: {len(processor.completed_jobs)}")
    print(f"  Failed jobs: {len(processor.failed_jobs)}")
    
    # Generate comparison report
    if len(processor.completed_jobs) >= 2:
        print("\\nGenerating comparison report...")
        evaluator = ReconstructionEvaluator()
        batch_data = evaluator.load_batch_results(results_file)
        evaluator.compare_configurations(batch_data, f"{batch_output_dir}/comparison")
        print("Comparison report saved to: data/batch_comparison/comparison/")
    
    return len(processor.failed_jobs) == 0

if __name__ == "__main__":
    main()

# example_interactive.py
"""
Example: Launch interactive reconstruction tool
"""

from interactive_tools import InteractiveReconstructionTool

def main():
    print("Launching Interactive Reconstruction Tool...")
    print("=" * 45)
    print("Features:")
    print("  - Load PLY fragments")
    print("  - Visual fragment inspection")
    print("  - Interactive surface matching")
    print("  - Manual alignment refinement")
    print("  - Real-time 3D visualization")
    print("  - Export reconstruction results")
    
    # Create and run interactive tool
    tool = InteractiveReconstructionTool()
    tool.run()

if __name__ == "__main__":
    main()

# create_example_data.py
"""
Create example PLY files for testing (synthetic data)
"""

import numpy as np
import open3d as o3d
from pathlib import Path

def create_synthetic_fragment(fragment_id, break_colors, noise_level=0.01):
    """Create a synthetic fragment with colored break surfaces"""
    
    # Create base shape (part of a cylinder/column)
    if fragment_id == 0:
        # Bottom fragment
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=0.3)
        mesh.translate([0, 0, 0])
    elif fragment_id == 1:
        # Middle fragment  
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=0.3)
        mesh.translate([0, 0, 0.25])
    else:
        # Top fragment
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.5, height=0.3)
        mesh.translate([0, 0, 0.5])
    
    # Add some noise to make it more realistic
    vertices = np.asarray(mesh.vertices)
    vertices += np.random.normal(0, noise_level, vertices.shape)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Create vertex colors (mostly gray with colored break surfaces)
    num_vertices = len(mesh.vertices)
    colors = np.ones((num_vertices, 3)) * 0.7  # Gray base color
    
    # Add colored break surfaces
    vertices = np.asarray(mesh.vertices)
    
    # Top surface (for fragments 0, 1)
    if fragment_id < 2:
        top_mask = vertices[:, 2] > (vertices[:, 2].max() - 0.05)
        colors[top_mask] = break_colors[0]  # First break color
    
    # Bottom surface (for fragments 1, 2)  
    if fragment_id > 0:
        bottom_mask = vertices[:, 2] < (vertices[:, 2].min() + 0.05)
        colors[bottom_mask] = break_colors[1] if len(break_colors) > 1 else break_colors[0]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    
    return mesh

def main():
    """Create example PLY files for testing"""
    output_dir = Path("data/examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating example PLY files...")
    
    # Define break surface colors (RGB 0-1)
    blue = [0.0, 0.0, 1.0]
    green = [0.0, 1.0, 0.0]
    red = [1.0, 0.0, 0.0]
    
    # Create fragments
    fragments = [
        create_synthetic_fragment(0, [blue], noise_level=0.005),      # Fragment 0: blue top
        create_synthetic_fragment(1, [blue, green], noise_level=0.005), # Fragment 1: blue bottom, green top
        create_synthetic_fragment(2, [green], noise_level=0.005),     # Fragment 2: green bottom
    ]
    
    # Save fragments
    for i, fragment in enumerate(fragments):
        filename = output_dir / f"synthetic_fragment_{i:03d}.ply"
        o3d.io.write_triangle_mesh(str(filename), fragment)
        print(f"Created: {filename}")
    
    print(f"\\nExample files created in: {output_dir}")
    print("These synthetic fragments have:")
    print("  - Fragment 0: Blue break surface on top")
    print("  - Fragment 1: Blue break surface on bottom, green on top")  
    print("  - Fragment 2: Green break surface on bottom")
    print("\\nFragments 0-1 should match via blue surfaces")
    print("Fragments 1-2 should match via green surfaces")
    
    # Create a simple test script
    test_script = output_dir / "test_reconstruction.py"
    with open(test_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""Test reconstruction with synthetic data"""

import sys
sys.path.append('..')  # Add parent directory to path

from main_pipeline import ReconstructionPipeline

def main():
    pipeline = ReconstructionPipeline()
    success = pipeline.run_full_pipeline("{output_dir}", "{output_dir}/output")
    
    if success:
        print("✓ Test reconstruction completed successfully!")
        print(f"Fragments aligned: {{len(pipeline.transformations)}}/{{len(pipeline.fragments)}}")
    else:
        print("✗ Test reconstruction failed")
    
    return success

if __name__ == "__main__":
    main()
''')
    
    print(f"\\nTest script created: {test_script}")
    print("Run it with: python data/examples/test_reconstruction.py")

if __name__ == "__main__":
    main()

# example_config_templates.json
"""
Example batch job configuration file
"""
BATCH_CONFIG_EXAMPLE = {
    "batch_info": {
        "description": "Compare reconstruction configurations",
        "created_by": "user",
        "dataset": "mayan_stele_fragments"
    },
    "jobs": [
        {
            "job_id": "default_reconstruction",
            "description": "Default configuration reconstruction",
            "input_directory": "data/input",
            "output_directory": "data/output/default",
            "config_template": "default",
            "priority": 1
        },
        {
            "job_id": "high_quality_reconstruction", 
            "description": "High quality configuration reconstruction",
            "input_directory": "data/input",
            "output_directory": "data/output/high_quality",
            "config_template": "high_quality",
            "priority": 2
        },
        {
            "job_id": "fast_reconstruction",
            "description": "Fast processing configuration reconstruction", 
            "input_directory": "data/input",
            "output_directory": "data/output/fast",
            "config_template": "fast_processing",
            "priority": 1
        }
    ]
}

def create_batch_config_file():
    """Create example batch configuration file"""
    import json
    
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    batch_config_file = config_dir / "example_batch_config.json"
    
    with open(batch_config_file, 'w') as f:
        json.dump(BATCH_CONFIG_EXAMPLE, f, indent=2)
    
    print(f"Example batch configuration created: {batch_config_file}")
    print("Use it with: python batch_processor.py batch configs/example_batch_config.json --output batch_results")

if __name__ == "__main__":
    create_batch_config_file()