from typing import Dict, Optional, Union, cast
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

__all__ = ['GNNConfig', 'GNNModel']

class GNNConfig:
    """Configuration for GNN model."""
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        edge_dim: Optional[int] = None,
        use_contrastive: bool = False,
        predict_similarity: bool = False,
        predict_transformation: bool = False,
        gradient_checkpointing: bool = False
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.edge_dim = edge_dim
        self.use_contrastive = use_contrastive
        self.predict_similarity = predict_similarity
        self.predict_transformation = predict_transformation
        self.gradient_checkpointing = gradient_checkpointing

class AttentionPooling(nn.Module):
    """Attention-based graph pooling"""
    
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.1): 
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Compute attention weights
        attention_weights = self.attention(x)  # [num_nodes, 1]
        
        # Apply softmax per graph
        attention_weights = torch.softmax(attention_weights, dim=0)
        
        # Apply attention pooling
        pooled = global_add_pool(x * attention_weights, batch)
        
        return pooled

class MultiHeadAttentionPooling(nn.Module):
    """Multi-head attention pooling for graphs"""
    
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size = int(batch.max().item()) + 1
        num_nodes = x.size(0)
        
        # Project to Q, K, V
        Q = self.query(x).view(num_nodes, self.num_heads, self.head_dim)
        K = self.key(x).view(num_nodes, self.num_heads, self.head_dim)
        V = self.value(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.view(num_nodes, -1)
        
        # Output projection
        output = self.output_proj(attended)
        
        # Pool per graph
        pooled = global_mean_pool(output, batch)
        
        return pooled

class GNNModel(nn.Module):
    """Graph Neural Network for fragment matching."""
    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for _ in range(config.num_layers):
            conv = GCNConv(
                config.hidden_dim,
                config.hidden_dim,
                improved=True,
                add_self_loops=True
            )
            self.convs.append(conv)
        
        # Batch normalization layers
        self.bns = nn.ModuleList()
        for _ in range(config.num_layers):
            self.bns.append(nn.BatchNorm1d(config.hidden_dim))
        
        # Dropout layer
        self.dropout = nn.Dropout(config.dropout)
        
        # Final projection to embedding dimension
        self.proj = nn.Linear(config.hidden_dim, config.embedding_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, 2),  # Two outputs for binary classification
            nn.LogSoftmax(dim=1)  # Use LogSoftmax for better numerical stability
        )
        
        # Initialize weights with Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass to get node embeddings."""
        # Initial projection
        x = self.input_proj(x)
        
        # Graph convolution layers
        for conv, bn in zip(self.convs, self.bns):
            # Convolution
            x_conv = conv(x, edge_index)
            
            # Batch normalization
            x_conv = bn(x_conv)
            
            # Non-linearity and dropout
            x_conv = F.relu(x_conv)
            x_conv = self.dropout(x_conv)
            
            # Residual connection
            x = x + x_conv
        
        # Final projection
        x = self.proj(x)
        
        return x
    
    def forward_fragment_embeddings(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass to get fragment embeddings through pooling."""
        # Use mean pooling to get fragment embeddings
        fragment_embeddings = global_mean_pool(node_embeddings, batch)
        return fragment_embeddings
    
    def forward(self, data: Union[Data, Batch]) -> Dict[str, torch.Tensor]:
        """Forward pass through the model"""
        # Extract features from data object
        if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
            raise ValueError("Data object must have 'x' and 'edge_index' attributes")
        
        # Get data attributes
        x = getattr(data, 'x')
        edge_index = getattr(data, 'edge_index')
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', None)
        
        # Create batch tensor if not present
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Get node embeddings
        node_embeddings = self.forward_node_embeddings(x, edge_index, edge_attr)
        
        # Get fragment embeddings through pooling
        fragment_embeddings = self.forward_fragment_embeddings(node_embeddings, batch)
        
        # Get logits for each fragment
        logits = self.classifier(fragment_embeddings)  # Shape: [num_fragments, 2]
        
        # Return outputs
        outputs = {
            'logits': logits,  # Shape: [num_fragments, 2]
            'embeddings': fragment_embeddings  # Shape: [num_fragments, embedding_dim]
        }
        
        return outputs

# Example usage
if __name__ == "__main__":
    # Model configuration
    model_config = GNNConfig(
        input_dim=10,
        hidden_dim=128,
        embedding_dim=64,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        use_edge_attr=True,
        edge_dim=None,
        use_contrastive=True,
        predict_similarity=True,
        predict_transformation=False,
        gradient_checkpointing=False
    )
    
    # Create model
    model = GNNModel(model_config)
    
    # Example forward pass
    print("Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)