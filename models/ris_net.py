import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class BaseModel(nn.Module):
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPModel(BaseModel):
    def __init__(self, input_dim, num_elements, hidden_dim, num_layers, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_elements))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RISNet(MLPModel):
    """Backward-compatible name for the default MLP phase predictor."""

    def __init__(self, input_dim, num_elements, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__(input_dim, num_elements, hidden_dim, num_layers, dropout)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, based on https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = self.W(h) # (batch, N, out_features)
        
        a1 = self.a[:self.out_features, :]
        a2 = self.a[self.out_features:, :]
        
        e1 = torch.matmul(Wh, a1) # (batch, N, 1)
        e2 = torch.matmul(Wh, a2) # (batch, N, 1)
        
        e = e1 + e2.transpose(1, 2) # (batch, N, N)
        e = self.leakyrelu(e)
        
        zero_vec = -9e15 * torch.ones_like(e)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
            
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.bmm(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGraphAttentionLayer(nn.Module):
    """Multi-head GAT block with a stable output feature dimension."""

    def __init__(self, in_features, out_features, num_heads, dropout, alpha, concat=True):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be >= 1")

        head_dim = max(1, out_features // num_heads)
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, head_dim, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
        merged_dim = head_dim * num_heads
        self.out_proj = (
            nn.Linear(merged_dim, out_features)
            if merged_dim != out_features else nn.Identity()
        )
        self.concat = concat

    def forward(self, h, adj):
        h = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        h = self.out_proj(h)
        return F.elu(h) if self.concat else h


class GNNModel(BaseModel):
    def __init__(self, input_dim, num_elements, hidden_dim, num_layers, dropout, config=None):
        super().__init__()
        self.num_elements = num_elements
        
        # Use GNN hidden dimension from config if available
        self.node_dim = getattr(config, 'GNN_HIDDEN_DIM', hidden_dim) if config else hidden_dim
        
        # Initial projection to node features
        self.feature_proj = nn.Linear(input_dim, num_elements * self.node_dim)
        
        # Create adjacency matrix for grid
        grid_rows = getattr(config, 'PIXEL_GRID_ROWS', 8) if config else 8
        grid_cols = getattr(config, 'PIXEL_GRID_COLS', 8) if config else 8
        if grid_rows * grid_cols != num_elements:
            raise ValueError(
                f"GNN grid ({grid_rows}x{grid_cols}) must match num_elements={num_elements}"
            )
        self.register_buffer('adj', self._build_grid_adj(grid_rows, grid_cols))
        
        # GAT Layers
        num_gnn_layers = getattr(config, 'GNN_NUM_LAYERS', 3) if config else num_layers
        num_heads = getattr(config, 'GNN_NUM_HEADS', 1) if config else 1
        self.gats = nn.ModuleList()
        for i in range(num_gnn_layers):
            concat = (i < num_gnn_layers - 1)
            self.gats.append(
                MultiHeadGraphAttentionLayer(
                    self.node_dim,
                    self.node_dim,
                    num_heads,
                    dropout,
                    0.2,
                    concat=concat,
                )
            )
        
        # Output projection
        self.out_proj = nn.Linear(self.node_dim, 1)

    def _build_grid_adj(self, rows, cols):
        num_nodes = rows * cols
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                # Self connection
                adj[idx, idx] = 1
                # Neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        n_idx = ni * cols + nj
                        adj[idx, n_idx] = 1
        return adj

    def forward(self, x):
        # x: (batch, input_dim)
        batch_size = x.size(0)
        h = self.feature_proj(x)
        h = h.view(batch_size, self.num_elements, self.node_dim)
        
        for gat in self.gats:
            h = gat(h, self.adj)
            
        out = self.out_proj(h).squeeze(-1) # (batch, num_elements)
        return out


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNNModel(BaseModel):
    def __init__(self, input_dim, num_elements, hidden_dim, num_layers, dropout, config=None):
        super().__init__()
        self.num_elements = num_elements
        self.grid_rows = getattr(config, 'PIXEL_GRID_ROWS', 8) if config else 8
        self.grid_cols = getattr(config, 'PIXEL_GRID_COLS', 8) if config else 8
        if self.grid_rows * self.grid_cols != num_elements:
            raise ValueError(
                f"CNN grid ({self.grid_rows}x{self.grid_cols}) must match num_elements={num_elements}"
            )
        
        channels = getattr(config, 'CNN_HIDDEN_CHANNELS', 64) if config else 64
        se_reduction = getattr(config, 'CNN_SE_REDUCTION', 16) if config else 16
        
        self.feature_proj = nn.Linear(input_dim, channels * self.grid_rows * self.grid_cols)
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SEBlock(channels, se_reduction),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            SEBlock(channels, se_reduction)
        )
        
        self.out_conv = nn.Conv2d(channels, 1, kernel_size=1)
        
    def forward(self, x):
        b = x.size(0)
        h = self.feature_proj(x)
        h = h.view(b, -1, self.grid_rows, self.grid_cols)
        h = self.conv_net(h)
        out = self.out_conv(h).view(b, self.num_elements)
        return out


class RISNetCNN(BaseModel):
    """Backward-compatible CNN for scripts that pass image-like RIS grids."""

    def __init__(self, input_channels=4, grid_size=(8, 8), hidden_channels=64, se_reduction=16):
        super().__init__()
        self.grid_size = grid_size
        self.num_elements = grid_size[0] * grid_size[1]
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            SEBlock(hidden_channels, se_reduction),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            SEBlock(hidden_channels, se_reduction),
        )
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        b = x.size(0)
        return self.out_conv(self.conv_net(x)).view(b, self.num_elements)


class TransformerModel(BaseModel):
    def __init__(self, input_dim, num_elements, hidden_dim, num_layers, dropout, config=None):
        super().__init__()
        self.num_elements = num_elements
        
        d_model = getattr(config, 'TRANSFORMER_D_MODEL', 256) if config else 256
        nhead = getattr(config, 'TRANSFORMER_NUM_HEADS', 8) if config else 8
        num_layers_tf = getattr(config, 'TRANSFORMER_NUM_LAYERS', 4) if config else 4
        dim_feedforward = getattr(config, 'TRANSFORMER_FF_DIM', 512) if config else 512
        
        self.d_model = d_model
        self.feature_proj = nn.Linear(input_dim, num_elements * d_model)
        
        # Position encoding
        self.pos_emb = nn.Parameter(torch.randn(1, num_elements, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_tf)
        
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x):
        b = x.size(0)
        h = self.feature_proj(x)
        h = h.view(b, self.num_elements, self.d_model)
        
        h = h + self.pos_emb
        h = self.transformer(h)
        out = self.out_proj(h).squeeze(-1)
        return out


def create_model(
    model_type,
    input_dim,
    num_elements,
    hidden_dim=None,
    num_layers=None,
    dropout=None,
    config=None
):
    """
    Factory function to create the requested neural network model.
    """
    if config is None:
        config = Config

    hidden_dim = getattr(config, 'HIDDEN_DIM', 256) if hidden_dim is None else hidden_dim
    num_layers = getattr(config, 'NUM_LAYERS', 3) if num_layers is None else num_layers
    dropout = getattr(config, 'DROPOUT', 0.1) if dropout is None else dropout

    normalized_type = model_type.replace("-", "_").upper()
    aliases = {
        "RISNET": "MLP",
        "MLP": "MLP",
        "GNN": "GNN",
        "CNN": "CNN",
        "CNN_ATTENTION": "CNN",
        "CNN+ATTENTION": "CNN",
        "TRANSFORMER": "TRANSFORMER",
    }
    model_key = aliases.get(normalized_type)

    if model_key == "MLP":
        return MLPModel(input_dim, num_elements, hidden_dim, num_layers, dropout)
    elif model_key == "GNN":
        return GNNModel(input_dim, num_elements, hidden_dim, num_layers, dropout, config)
    elif model_key == "CNN":
        return CNNModel(input_dim, num_elements, hidden_dim, num_layers, dropout, config)
    elif model_key == "TRANSFORMER":
        return TransformerModel(input_dim, num_elements, hidden_dim, num_layers, dropout, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
