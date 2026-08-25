import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MLPModel(BaseModel):
    def __init__(self, input_dim, num_elements, hidden_dim, num_layers, dropout):
        super().__init__()
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

class GNNModel(BaseModel):
    def __init__(self, input_dim, num_elements, hidden_dim, num_layers, dropout, config=None):
        super().__init__()
        self.num_elements = num_elements
        
        # Use GNN hidden dimension from config if available
        self.node_dim = getattr(config, 'GNN_HIDDEN_DIM', 256) if config else 256
        
        # Initial projection to node features
        self.feature_proj = nn.Linear(input_dim, num_elements * self.node_dim)
        
        # Create adjacency matrix for grid
        grid_rows = getattr(config, 'PIXEL_GRID_ROWS', 8) if config else 8
        grid_cols = getattr(config, 'PIXEL_GRID_COLS', 8) if config else 8
        self.register_buffer('adj', self._build_grid_adj(grid_rows, grid_cols))
        
        # GAT Layers
        # We can implement multi-layer GAT as requested
        num_gnn_layers = getattr(config, 'GNN_NUM_LAYERS', 3) if config else num_layers
        self.gats = nn.ModuleList()
        for i in range(num_gnn_layers):
            concat = (i < num_gnn_layers - 1)
            self.gats.append(GraphAttentionLayer(self.node_dim, self.node_dim, dropout, 0.2, concat=concat))
        
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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
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


def create_model(model_type, input_dim, num_elements, hidden_dim, num_layers, dropout, config=None):
    """
    Factory function to create the requested neural network model.
    """
    if model_type == "MLP":
        return MLPModel(input_dim, num_elements, hidden_dim, num_layers, dropout)
    elif model_type == "GNN":
        return GNNModel(input_dim, num_elements, hidden_dim, num_layers, dropout, config)
    elif model_type == "CNN":
        return CNNModel(input_dim, num_elements, hidden_dim, num_layers, dropout, config)
    elif model_type == "Transformer":
        return TransformerModel(input_dim, num_elements, hidden_dim, num_layers, dropout, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
