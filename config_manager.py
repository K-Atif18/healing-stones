import json
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

@dataclass
class ColorExtractionConfig:
    """Configuration for color-based break surface extraction"""
    blue_range: Dict[str, List[int]] = None
    green_range: Dict[str, List[int]] = None
    red_range: Dict[str, List[int]] = None
    color_tolerance: float = 0.3
    min_cluster_size: int = 50
    clustering_eps: float = 0.01
    clustering_min_samples: int = 10
    
    def __post_init__(self):
        if self.blue_range is None:
            self.blue_range = {'min': [0, 0, 100], 'max': [100, 100, 255]}
        if self.green_range is None:
            self.green_range = {'min': [0, 100, 0], 'max': [100, 255, 100]}
        if self.red_range is None:
            self.red_range = {'min': [100, 0, 0], 'max': [255, 100, 100]}

@dataclass
class FeatureExtractionConfig:
    """Configuration for geometric feature extraction"""
    curvature_radius: float = 0.02
    normal_estimation_neighbors: int = 30
    boundary_detection_method: str = "convex_hull"  # or "alpha_shape"
    alpha_value: float = 0.05
    moment_order: int = 2
    enable_texture_features: bool = False
    
@dataclass
class MatchingConfig:
    """Configuration for surface matching"""
    min_similarity: float = 0.6
    use_optimal_matching: bool = True
    similarity_weights: Dict[str, float] = None
    distance_threshold: float = 0.02
    angle_threshold: float = 30.0  # degrees
    size_ratio_threshold: float = 0.5
    
    def __post_init__(self):
        if self.similarity_weights is None:
            self.similarity_weights = {
                'normal_similarity': 0.25,
                'area_similarity': 0.15,
                'shape_similarity': 0.20,
                'curvature_similarity': 0.15,
                'boundary_similarity': 0.15,
                'size_similarity': 0.10
            }

@dataclass
class AlignmentConfig:
    """Configuration for fragment alignment"""
    icp_max_iterations: int = 100
    icp_tolerance: float = 1e-6
    icp_threshold: float = 0.02
    optimization_method: str = "BFGS"  # or "L-BFGS-B", "Powell"
    max_optimization_iterations: int = 100
    alignment_quality_threshold: float = 0.05
    enable_global_optimization: bool = True
    
@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    show_original_fragments: bool = True
    show_break_surfaces: bool = True
    show_matches: bool = True
    show_step_by_step: bool = False
    show_final_reconstruction: bool = True
    save_images: bool = True
    image_dpi: int = 300
    window_size: Tuple[int, int] = (1200, 800)
    background_color: List[float] = None
    
    def __post_init__(self):
        if self.background_color is None:
            self.background_color = [0.1, 0.1, 0.1]

@dataclass
class OutputConfig:
    """Configuration for output generation"""
    save_intermediate_results: bool = True
    save_aligned_fragments: bool = True
    save_combined_mesh: bool = True
    save_transformation_matrices: bool = True
    save_quality_reports: bool = True
    save_feature_data: bool = False
    output_format: str = "ply"  # or "obj", "stl"
    compression_level: int = 0

@dataclass
class ReconstructionConfig:
    """Master configuration for the entire reconstruction pipeline"""
    # Sub-configurations
    color_extraction: ColorExtractionConfig = None
    feature_extraction: FeatureExtractionConfig = None
    matching: MatchingConfig = None
    alignment: AlignmentConfig = None
    visualization: VisualizationConfig = None
    output: OutputConfig = None
    
    # Global settings
    verbose: bool = True
    debug_mode: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    random_seed: int = 42
    
    def __post_init__(self):
        if self.color_extraction is None:
            self.color_extraction = ColorExtractionConfig()
        if self.feature_extraction is None:
            self.feature_extraction = FeatureExtractionConfig()
        if self.matching is None:
            self.matching = MatchingConfig()
        if self.alignment is None:
            self.alignment = AlignmentConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.output is None:
            self.output = OutputConfig()

class ConfigurationManager:
    """
    Manager for loading, saving, and validating reconstruction configurations
    """
    
    def __init__(self):
        self.config_templates = {
            'default': self._create_default_config(),
            'high_quality': self._create_high_quality_config(),
            'fast_processing': self._create_fast_processing_config(),
            'conservative': self._create_conservative_config(),
            'aggressive': self._create_aggressive_config()
        }
    
    def _create_default_config(self) -> ReconstructionConfig:
        """Create default configuration"""
        return ReconstructionConfig()
    
    def _create_high_quality_config(self) -> ReconstructionConfig:
        """Create high-quality reconstruction configuration"""
        config = ReconstructionConfig()
        
        # More stringent color extraction
        config.color_extraction.color_tolerance = 0.2
        config.color_extraction.min_cluster_size = 30
        
        # Enhanced feature extraction
        config.feature_extraction.curvature_radius = 0.015
        config.feature_extraction.normal_estimation_neighbors = 50
        
        # Stricter matching
        config.matching.min_similarity = 0.7
        config.matching.similarity_weights['normal_similarity'] = 0.30
        config.matching.similarity_weights['curvature_similarity'] = 0.20
        
        # More iterations for alignment
        config.alignment.icp_max_iterations = 200
        config.alignment.max_optimization_iterations = 200
        config.alignment.icp_tolerance = 1e-8
        
        return config
    
    def _create_fast_processing_config(self) -> ReconstructionConfig:
        """Create fast processing configuration"""
        config = ReconstructionConfig()
        
        # Relaxed color extraction
        config.color_extraction.color_tolerance = 0.4
        config.color_extraction.min_cluster_size = 100
        
        # Simplified feature extraction
        config.feature_extraction.curvature_radius = 0.03
        config.feature_extraction.normal_estimation_neighbors = 20
        
        # Relaxed matching
        config.matching.min_similarity = 0.5
        config.matching.use_optimal_matching = False
        
        # Fewer iterations
        config.alignment.icp_max_iterations = 50
        config.alignment.max_optimization_iterations = 50
        
        # Minimal visualization
        config.visualization.show_step_by_step = False
        config.visualization.save_images = False
        
        return config
    
    def _create_conservative_config(self) -> ReconstructionConfig:
        """Create conservative matching configuration"""
        config = ReconstructionConfig()
        
        # Conservative matching
        config.matching.min_similarity = 0.8
        config.matching.distance_threshold = 0.01
        config.matching.angle_threshold = 15.0
        config.matching.size_ratio_threshold = 0.7
        
        # Conservative alignment
        config.alignment.alignment_quality_threshold = 0.02
        
        return config
    
    def _create_aggressive_config(self) -> ReconstructionConfig:
        """Create aggressive matching configuration"""
        config = ReconstructionConfig()
        
        # Aggressive matching
        config.matching.min_similarity = 0.4
        config.matching.distance_threshold = 0.05
        config.matching.angle_threshold = 45.0
        config.matching.size_ratio_threshold = 0.3
        
        # More flexible alignment
        config.alignment.alignment_quality_threshold = 0.1
        config.alignment.enable_global_optimization = True
        
        return config
    
    def load_config(self, config_path: str) -> ReconstructionConfig:
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file format
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return self._dict_to_config(config_dict)
    
    def save_config(self, config: ReconstructionConfig, config_path: str, format: str = 'json'):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_dict = asdict(config)
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif format.lower() in ['yml', 'yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Configuration saved to: {config_path}")
    
    def _dict_to_config(self, config_dict: dict) -> ReconstructionConfig:
        """Convert dictionary to ReconstructionConfig"""
        # Handle nested configurations
        if 'color_extraction' in config_dict and isinstance(config_dict['color_extraction'], dict):
            config_dict['color_extraction'] = ColorExtractionConfig(**config_dict['color_extraction'])
        
        if 'feature_extraction' in config_dict and isinstance(config_dict['feature_extraction'], dict):
            config_dict['feature_extraction'] = FeatureExtractionConfig(**config_dict['feature_extraction'])
        
        if 'matching' in config_dict and isinstance(config_dict['matching'], dict):
            config_dict['matching'] = MatchingConfig(**config_dict['matching'])
        
        if 'alignment' in config_dict and isinstance(config_dict['alignment'], dict):
            config_dict['alignment'] = AlignmentConfig(**config_dict['alignment'])
        
        if 'visualization' in config_dict and isinstance(config_dict['visualization'], dict):
            config_dict['visualization'] = VisualizationConfig(**config_dict['visualization'])
        
        if 'output' in config_dict and isinstance(config_dict['output'], dict):
            config_dict['output'] = OutputConfig(**config_dict['output'])
        
        return ReconstructionConfig(**config_dict)
    
    def get_template_config(self, template_name: str) -> ReconstructionConfig:
        """Get a predefined configuration template"""
        if template_name not in self.config_templates:
            available = list(self.config_templates.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
        
        return self.config_templates[template_name]
    
    def create_custom_config(self, base_template: str = 'default', **overrides) -> ReconstructionConfig:
        """Create custom configuration based on template with overrides"""
        config = self.get_template_config(base_template)
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Handle nested attributes
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        raise ValueError(f"Invalid configuration key: {key}")
                
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
                else:
                    raise ValueError(f"Invalid configuration key: {key}")
        
        return config
    
    def validate_config(self, config: ReconstructionConfig) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Validate color extraction
        if config.color_extraction.color_tolerance < 0 or config.color_extraction.color_tolerance > 1:
            warnings.append("Color tolerance should be between 0 and 1")
        
        if config.color_extraction.min_cluster_size < 10:
            warnings.append("Minimum cluster size very small, may include noise")
        
        # Validate matching
        if config.matching.min_similarity < 0 or config.matching.min_similarity > 1:
            warnings.append("Minimum similarity should be between 0 and 1")
        
        weights_sum = sum(config.matching.similarity_weights.values())
        if abs(weights_sum - 1.0) > 0.01:
            warnings.append(f"Similarity weights sum to {weights_sum:.3f}, should sum to 1.0")
        
        # Validate alignment
        if config.alignment.icp_threshold <= 0:
            warnings.append("ICP threshold should be positive")
        
        if config.alignment.icp_max_iterations < 10:
            warnings.append("ICP max iterations very low, may not converge")
        
        # Validate visualization
        if len(config.visualization.background_color) != 3:
            warnings.append("Background color should have 3 components (RGB)")
        
        for color_val in config.visualization.background_color:
            if color_val < 0 or color_val > 1:
                warnings.append("Background color values should be between 0 and 1")
                break
        
        return warnings
    
    def print_config_summary(self, config: ReconstructionConfig):
        """Print a human-readable summary of the configuration"""
        print("=== RECONSTRUCTION CONFIGURATION SUMMARY ===")
        print()
        
        print("Color Extraction:")
        print(f"  Tolerance: {config.color_extraction.color_tolerance}")
        print(f"  Min cluster size: {config.color_extraction.min_cluster_size}")
        print()
        
        print("Feature Extraction:")
        print(f"  Curvature radius: {config.feature_extraction.curvature_radius}")
        print(f"  Normal neighbors: {config.feature_extraction.normal_estimation_neighbors}")
        print()
        
        print("Surface Matching:")
        print(f"  Min similarity: {config.matching.min_similarity}")
        print(f"  Optimal matching: {config.matching.use_optimal_matching}")
        print(f"  Distance threshold: {config.matching.distance_threshold}")
        print()
        
        print("Fragment Alignment:")
        print(f"  ICP max iterations: {config.alignment.icp_max_iterations}")
        print(f"  ICP threshold: {config.alignment.icp_threshold}")
        print(f"  Optimization method: {config.alignment.optimization_method}")
        print()
        
        print("Visualization:")
        print(f"  Show step-by-step: {config.visualization.show_step_by_step}")
        print(f"  Save images: {config.visualization.save_images}")
        print()
        
        print("Output:")
        print(f"  Save intermediate: {config.output.save_intermediate_results}")
        print(f"  Output format: {config.output.output_format}")
        print()
        
        print("Global:")
        print(f"  Verbose: {config.verbose}")
        print(f"  Parallel processing: {config.parallel_processing}")
        print(f"  Max workers: {config.max_workers}")

# Example usage and CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Manager for Stele Reconstruction")
    parser.add_argument("action", choices=['create', 'validate', 'show'], 
                       help="Action to perform")
    parser.add_argument("--template", default="default",
                       choices=['default', 'high_quality', 'fast_processing', 'conservative', 'aggressive'],
                       help="Configuration template to use")
    parser.add_argument("--output", "-o", help="Output configuration file")
    parser.add_argument("--input", "-i", help="Input configuration file")
    parser.add_argument("--format", choices=['json', 'yaml'], default='json',
                       help="Configuration file format")
    
    args = parser.parse_args()
    
    manager = ConfigurationManager()
    
    if args.action == 'create':
        config = manager.get_template_config(args.template)
        
        if args.output:
            manager.save_config(config, args.output, args.format)
        else:
            manager.print_config_summary(config)
    
    elif args.action == 'validate':
        if not args.input:
            print("Error: --input required for validation")
            exit(1)
        
        config = manager.load_config(args.input)
        warnings = manager.validate_config(config)
        
        if warnings:
            print("Configuration warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        else:
            print("Configuration is valid!")
    
    elif args.action == 'show':
        if args.input:
            config = manager.load_config(args.input)
        else:
            config = manager.get_template_config(args.template)
        
        manager.print_config_summary(config)