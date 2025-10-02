#!/usr/bin/env python3
"""
Graph Neural Network Models for Cell2Gene

author: Jingkun Yu
"""

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

# PyTorch Geometric imports for GNN
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, GCNConv
    GNN_AVAILABLE = True
    print(f"✓ PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError as e:
    GNN_AVAILABLE = False
    print(f"警告: PyTorch Geometric不可用: {e}")


class StaticGraphGNN(nn.Module):
    """
    Static Graph Neural Network supporting GAT and GCN
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, gnn_type='GAT', dropout=0.3):
        super(StaticGraphGNN, self).__init__()
        
        if not GNN_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN functionality")
        
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = nn.Dropout(dropout)
        
        self.convs = nn.ModuleList()
        
        if gnn_type == 'GAT':
            # GAT layers
            if num_layers == 1:
                self.convs.append(GATConv(input_dim, output_dim, heads=1, concat=False))
            else:
                # First layer
                self.convs.append(GATConv(input_dim, hidden_dim, heads=1, concat=False))
                # Hidden layers
                for _ in range(num_layers - 2):
                    self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False))
                # Output layer
                self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
                
        elif gnn_type == 'GCN':
            # GCN layers
            if num_layers == 1:
                self.convs.append(GCNConv(input_dim, output_dim))
            else:
                # First layer
                self.convs.append(GCNConv(input_dim, hidden_dim))
                # Hidden layers
                for _ in range(num_layers - 2):
                    self.convs.append(GCNConv(hidden_dim, hidden_dim))
                # Output layer
                self.convs.append(GCNConv(hidden_dim, output_dim))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # 不在最后一层应用激活和dropout
                x = torch.relu(x)
                x = self.dropout(x)
            
        return x