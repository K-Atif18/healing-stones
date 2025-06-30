from feature_extractor import plot_mesh_with_boundary
import numpy as np

# Example: visualize one mesh
file_path = "dataset_root/data/frag_1.ply"
# Load boundary indices from saved features if needed
# Or extract directly:
from feature_extractor import FeatureExtractor
extractor = FeatureExtractor()
boundary_info = extractor.extract_mesh_boundary(file_path)
if boundary_info is not None:
    plot_mesh_with_boundary(file_path, boundary_info['boundary_indices'])