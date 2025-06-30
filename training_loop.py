import os
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as GeometricDataLoader
from trainer import Trainer, TrainerConfig
from GNN_model import GNNModel, GNNConfig
from torch.serialization import safe_globals

# --- Utility to load all graphs from batches ---
def load_all_graphs(graph_dir):
    all_graphs = []
    for fname in sorted(os.listdir(graph_dir)):
        if fname.endswith('.pt'):
            with safe_globals([
                (lambda x: x, 'torch_geometric.data.data.DataEdgeAttr')
            ]):
                batch = torch.load(os.path.join(graph_dir, fname), weights_only=False)
                all_graphs.extend(batch)
    return all_graphs

class GraphDataset(torch.utils.data.Dataset):
    """Dataset for graph data."""
    def __init__(self, data_list):
        super().__init__()
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    print("Starting main function...")
    # Check for CUDA
    if torch.cuda.is_available():
        print("CUDA is available, setting up GPU...")
        torch.cuda.empty_cache()
    # Load all graphs from batches
    graph_dir = "dataset_graphs"
    data_list = load_all_graphs(graph_dir)
    print(f"Loaded {len(data_list)} graphs.")
    if len(data_list) == 0:
        raise RuntimeError("No graphs found in dataset_graphs/ directory!")
    # Automatically set input_dim
    input_dim = data_list[0].x.shape[1]
    print(f"Detected input_dim: {input_dim}")
    # Model config
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
    # Trainer config
    trainer_config = TrainerConfig(
        batch_size=2,  # You may want to increase this if you have enough RAM/GPU
        num_workers=0,  # Set >0 for faster loading if you have CPU cores
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        loss_type="bce",
        class_weights=[1.0, 3.0],
        focal_alpha=0.25,
        focal_gamma=2.0,
        temperature=0.07,
        margin=0.5,
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=0.01,
        scheduler="plateau",
        patience=10,
        factor=0.5,
        min_lr=1e-6,
        max_epochs=200,
        early_stopping_patience=20,
        gradient_clipping=1.0,
        gradient_checkpointing=False
    )
    # Split dataset
    dataset = GraphDataset(data_list)
    train_size = int(trainer_config.train_ratio * len(dataset))
    val_size = int(trainer_config.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    # Data loaders
    train_loader = GeometricDataLoader(
        [dataset[i] for i in train_dataset.indices],
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.num_workers
    )
    val_loader = GeometricDataLoader(
        [dataset[i] for i in val_dataset.indices],
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers
    )
    test_loader = GeometricDataLoader(
        [dataset[i] for i in test_dataset.indices],
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers
    )
    print("\nDataset split information:")
    print(f"Train batches: {len(train_loader)} ({len(train_dataset)} samples)")
    print(f"Val batches: {len(val_loader)} ({len(val_dataset)} samples)")
    print(f"Test batches: {len(test_loader)} ({len(test_dataset)} samples)")
    # Model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNModel(model_config)
    trainer = Trainer(model, trainer_config, device)
    # Train model
    print("\nStarting training...")
    try:
        trainer.train(train_loader, val_loader, test_loader)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
