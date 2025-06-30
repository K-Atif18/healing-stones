import os
import torch
import numpy as np
from typing import List, Tuple, Optional, Callable, Any, Sequence
from torch.serialization import safe_globals
from GNN_model import GNNModel, GNNConfig
from graph_constructor import GraphConstructor, GraphConfig
from feature_extractor import FeatureExtractor, FeatureConfig

def load_model(model_path: str, input_dim: int, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> GNNModel:
    """Load the trained model with the correct input_dim."""
    model_config = GNNConfig(
        input_dim=input_dim,
        hidden_dim=32,
        embedding_dim=16,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
        use_edge_attr=True,
        edge_dim=None,
        use_contrastive=False,
        predict_similarity=False,
        predict_transformation=False,
        gradient_checkpointing=False
    )
    model = GNNModel(model_config)
    try:
        with safe_globals([
            (lambda x: x, 'torch_geometric.data.data.DataEdgeAttr')
        ]):
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
    except Exception as e1:
        error_msg = f"Failed to load model weights from {model_path}.\nError: {str(e1)}"
        raise RuntimeError(error_msg)
    model = model.to(device)
    model.eval()
    return model

def predict_pair(
    model_path: str,
    fragment_a_path: str,
    fragment_b_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[float, float]:
    """
    Predict if two fragments match.
    Returns:
        Tuple[float, float]: (probability of match, probability of no match)
    """
    # Use a lower target_points and higher max_nodes for prediction
    feature_extractor = FeatureExtractor(FeatureConfig(target_points=15000))  # Lower to keep total nodes < max_nodes
    graph_constructor = GraphConstructor(GraphConfig(max_nodes=80000))  # Increase to allow large pairs
    # Extract features
    features_a = feature_extractor.extract_features(fragment_a_path)
    features_b = feature_extractor.extract_features(fragment_b_path)
    if features_a is None or features_b is None:
        raise ValueError("Failed to extract features from one or both fragments")
    # Get input_dim from features
    input_dim = features_a['point_features'].shape[1]
    # Build model with correct input_dim
    model = load_model(model_path, input_dim, device)
    # Build graph
    graph = graph_constructor.build_graph_from_features(
        features_a=features_a,
        features_b=features_b,
        label=0,  # Dummy label for prediction
        metadata={
            "fragment_a": os.path.basename(fragment_a_path),
            "fragment_b": os.path.basename(fragment_b_path)
        }
    )
    if graph is None:
        raise ValueError("Failed to construct graph from features")
    graph = graph.to(device)
    with torch.no_grad():
        output = model(graph)
        logits = output["logits"]
        probabilities = torch.exp(logits)
    return tuple(probabilities[0].cpu().numpy().tolist())

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Predict if two fragments match")
    parser.add_argument("--model", type=str, default="checkpoints/best.pt",
                      help="Path to the trained model checkpoint")
    parser.add_argument("--fragment1", type=str, required=True,
                      help="Path to first fragment PLY file")
    parser.add_argument("--fragment2", type=str, required=True,
                      help="Path to second fragment PLY file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    print(f"Loading model from {args.model}...")
    print(f"Processing fragments:")
    print(f"  Fragment 1: {args.fragment1}")
    print(f"  Fragment 2: {args.fragment2}")
    try:
        no_match_prob, match_prob = predict_pair(
            args.model, args.fragment1, args.fragment2, args.device
        )
        print("\nPrediction Results:")
        print(f"  Match probability: {match_prob:.4f}")
        print(f"  No match probability: {no_match_prob:.4f}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise e

if __name__ == "__main__":
    main() 