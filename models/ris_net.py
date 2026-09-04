import torch
import torch.nn.functional as F
from torch import nn

from config import Config


class BaseModel(nn.Module):
    """Abstract base class for all RIS phase-prediction models.

    Phase is a circular quantity, so every model in this module predicts each
    element's phase as a point on the unit circle: two unconstrained outputs
    (cos, sin) per element, from which the angle is recovered with ``atan2``.

    Regressing the angle directly and penalising the wrapped difference makes
    the objective periodic in the network output. That surface has no single
    descent direction -- a target near 0 and a target near 2*pi pull the same
    output in opposite directions -- and training stalls at the random-guess
    error of 90 degrees regardless of how long it runs. The (cos, sin) target
    is smooth and non-periodic, so ordinary MSE has a well-behaved gradient.

    Subclasses emit ``2 * num_elements`` values from their head and return
    ``(batch, num_elements, 2)`` from :meth:`forward_components`. Callers that
    want angles use :meth:`forward`, which stays ``(batch, num_elements)``.
    """

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_components(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return the raw (cos, sin) pair per element, shape ``(B, N, 2)``.

        This is what the training loss should consume; it is differentiable
        everywhere and free of the 2*pi wrap discontinuity.
        """
        raise NotImplementedError

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Return predicted phase shifts in ``(-pi, pi]``, shape ``(B, N)``."""
        comp = self.forward_components(x)
        return torch.atan2(comp[..., 1], comp[..., 0])

    @staticmethod
    def phase_targets(phases: "torch.Tensor") -> "torch.Tensor":
        """Map target angles ``(B, N)`` onto the unit circle ``(B, N, 2)``."""
        return torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1)


class MLPModel(BaseModel):
    """Multi-Layer Perceptron for RIS phase prediction.

    A fully-connected feed-forward network that maps concatenated channel
    state features directly to per-element phase shifts.  Each hidden layer
    uses ReLU activation; optional dropout is applied after every hidden layer.

    Args:
        input_dim: Dimensionality of the input feature vector.
        num_elements: Number of RIS elements (output dimension).
        hidden_dim: Width of each hidden layer.
        num_layers: Number of hidden layers (must be >= 1).
        dropout: Dropout probability applied after each hidden layer (0 = off).
    """

    def __init__(self, input_dim: int, num_elements: int, hidden_dim: int,
                 num_layers: int, dropout: float = 0.0):
        """Initialise the MLP by building a sequential stack of linear layers."""
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_elements = num_elements

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        # Two outputs per element: the (cos, sin) pair (see BaseModel).
        layers.append(nn.Linear(hidden_dim, 2 * num_elements))
        self.net = nn.Sequential(*layers)

    def forward_components(self, x):
        """Map input features to per-element (cos, sin) pairs.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Tensor of shape ``(batch, num_elements, 2)``.
        """
        return self.net(x).view(x.size(0), self.num_elements, 2)


class RISNet(MLPModel):
    """Backward-compatible name for the default MLP phase predictor."""

    def __init__(self, input_dim, num_elements, hidden_dim=256, num_layers=3, dropout=0.1):
        """Initialise with the same signature as :class:`MLPModel` plus sensible defaults."""
        super().__init__(input_dim, num_elements, hidden_dim, num_layers, dropout)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, based on https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features: int, out_features: int, dropout: float,
                 alpha: float, concat: bool = True):
        """Initialise linear projection and attention weight parameters.

        Args:
            in_features: Input node feature dimension.
            out_features: Output node feature dimension per head.
            dropout: Attention coefficient dropout probability.
            alpha: Negative slope for the LeakyReLU activation.
            concat: If ``True``, apply ELU activation after aggregation
                (used for all layers except the final one).
        """
        super().__init__()
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
        """Compute attended node features for one graph attention head.

        Args:
            h: Node feature tensor of shape ``(batch, N, in_features)``.
            adj: Adjacency matrix of shape ``(N, N)`` or ``(batch, N, N)``;
                non-zero entries indicate edges.

        Returns:
            Attended node features of shape ``(batch, N, out_features)``
            after ELU if ``concat=True``, else raw pre-activation values.
        """
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

    def __init__(self, in_features: int, out_features: int, num_heads: int,
                 dropout: float, alpha: float, concat: bool = True):
        """Initialise *num_heads* GAT heads and an output projection.

        Args:
            in_features: Input node feature dimension.
            out_features: Desired output feature dimension (after projection).
            num_heads: Number of parallel attention heads (must be ≥ 1).
            dropout: Attention dropout probability passed to each head.
            alpha: LeakyReLU negative slope passed to each head.
            concat: If ``True``, apply ELU after the output projection.
        """
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
        """Run all heads in parallel, concatenate, and project to *out_features*.

        Args:
            h: Node feature tensor of shape ``(batch, N, in_features)``.
            adj: Adjacency matrix passed through to each head.

        Returns:
            Node feature tensor of shape ``(batch, N, out_features)``.
        """
        h = torch.cat([head(h, adj) for head in self.heads], dim=-1)
        h = self.out_proj(h)
        return F.elu(h) if self.concat else h


class GNNModel(BaseModel):
    """Graph Neural Network model for RIS phase prediction using GAT layers.

    Models RIS element interactions as a spatial graph where nodes are pixels
    and edges connect 4-neighbours on a rectangular grid.  A shared feature
    encoder produces a context vector that is added to learnable per-node
    embeddings; stacked Graph Attention (GAT) layers then perform message
    passing before a final per-node linear projection yields phase shifts.

    Args:
        input_dim: Dimensionality of the input feature vector.
        num_elements: Number of RIS elements; must equal ``grid_rows × grid_cols``.
        hidden_dim: Default node feature dimension (overridden by ``config.GNN_HIDDEN_DIM``).
        num_layers: Default number of GAT layers (overridden by ``config.GNN_NUM_LAYERS``).
        dropout: Dropout probability in the feature encoder.
        config: Optional :class:`Config` instance for architecture hyper-parameters.
    """

    def __init__(self, input_dim: int, num_elements: int, hidden_dim: int,
                 num_layers: int, dropout: float, config=None):
        """Build the feature encoder, adjacency matrix, node embeddings, and GAT stack."""
        super().__init__()
        self.num_elements = num_elements
        
        # Use GNN hidden dimension from config if available
        self.node_dim = getattr(config, 'GNN_HIDDEN_DIM', hidden_dim) if config else hidden_dim
        
        # Compact shared feature encoder. The previous direct projection from
        # input_dim to num_elements * node_dim made communication cost scale
        # as O(users * elements * hidden_dim). A shared context plus learned
        # element embeddings keeps the model size practical while preserving
        # per-element message passing.
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, self.node_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.node_dim, self.node_dim),
        )
        
        # Create adjacency matrix for grid
        grid_rows = getattr(config, 'PIXEL_GRID_ROWS', 8) if config else 8
        grid_cols = getattr(config, 'PIXEL_GRID_COLS', 8) if config else 8
        if grid_rows * grid_cols != num_elements:
            raise ValueError(
                f"GNN grid ({grid_rows}x{grid_cols}) must match num_elements={num_elements}"
            )
        self.register_buffer('adj', self._build_grid_adj(grid_rows, grid_cols))
        self.node_embedding = nn.Parameter(torch.randn(1, num_elements, self.node_dim) * 0.02)
        
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
        self.out_proj = nn.Linear(self.node_dim, 2)  # (cos, sin) per node

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

    def forward_components(self, x):
        """Forward pass through encoder, node-embedding fusion, and GAT layers.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Per-element (cos, sin) pairs of shape ``(batch, num_elements, 2)``;
            use :meth:`forward` to obtain angles.
        """
        # x: (batch, input_dim)
        batch_size = x.size(0)
        context = self.feature_encoder(x)
        h = context.unsqueeze(1) + self.node_embedding.expand(batch_size, -1, -1)
        
        for gat in self.gats:
            h = gat(h, self.adj)
            
        return self.out_proj(h)  # (batch, num_elements, 2)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Recalibrates channel-wise feature responses by modelling inter-channel
    dependencies.  Applies global average pooling to produce channel
    descriptors, then uses a bottleneck FC network to produce per-channel
    scaling weights in [0, 1].

    Args:
        channel: Number of input channels.
        reduction: Bottleneck reduction ratio for the FC layers.

    Reference:
        Hu et al., "Squeeze-and-Excitation Networks," CVPR 2018.
    """

    def __init__(self, channel: int, reduction: int = 16):
        """Initialise the squeeze-and-excitation block."""
        super().__init__()
        reduced_channels = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply channel-wise squeeze-and-excitation recalibration.

        Args:
            x: Feature map of shape ``(batch, channel, H, W)``.

        Returns:
            Rescaled feature map of the same shape.
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNNModel(BaseModel):
    """CNN with Squeeze-and-Excitation attention for RIS phase prediction.

    Interprets the RIS pixel grid as a 2-D image and applies convolutional
    layers interleaved with SE channel attention.  A shared feature encoder
    maps the input CSI vector to a context tensor that is added to a
    learnable spatial embedding before the convolutional trunk.

    Args:
        input_dim: Dimensionality of the input feature vector.
        num_elements: Number of RIS elements; must equal ``grid_rows × grid_cols``.
        hidden_dim: Width of the feature encoder projection.
        num_layers: Unused (reserved for API consistency with other models).
        dropout: Dropout probability in the feature encoder.
        config: Optional :class:`Config` instance for ``PIXEL_GRID_ROWS``,
            ``PIXEL_GRID_COLS``, ``CNN_HIDDEN_CHANNELS``, and ``CNN_SE_REDUCTION``.
    """

    def __init__(self, input_dim: int, num_elements: int, hidden_dim: int,
                 num_layers: int, dropout: float, config=None):
        """Build feature encoder, spatial embedding, and convolutional trunk."""
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
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, channels),
        )
        self.spatial_embedding = nn.Parameter(
            torch.randn(1, channels, self.grid_rows, self.grid_cols) * 0.02
        )
        
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
        
        self.out_conv = nn.Conv2d(channels, 2, kernel_size=1)  # (cos, sin)
        
    def forward_components(self, x):
        """Forward pass: encode CSI, fuse spatial embedding, apply CNN trunk.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Per-element (cos, sin) pairs of shape ``(batch, num_elements, 2)``;
            use :meth:`forward` to obtain angles.
        """
        b = x.size(0)
        context = self.feature_encoder(x).view(b, -1, 1, 1)
        h = context + self.spatial_embedding.expand(b, -1, -1, -1)
        h = self.conv_net(h)
        # (B, 2, R, C) -> (B, N, 2)
        return self.out_conv(h).view(b, 2, self.num_elements).permute(0, 2, 1)


class RISNetCNN(BaseModel):
    """Backward-compatible CNN for scripts that pass image-like RIS grids."""

    def __init__(self, input_channels: int = 4, grid_size: tuple = (8, 8),
                 hidden_channels: int = 64, se_reduction: int = 16):
        """Initialise the backward-compatible image-input CNN.

        Args:
            input_channels: Number of channels in the input image tensor
                (e.g. 4 for a stacked amplitude/phase representation).
            grid_size: ``(rows, cols)`` spatial size of each RIS tile grid.
            hidden_channels: Number of feature maps in both conv layers.
            se_reduction: SE bottleneck reduction ratio.
        """
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
        self.out_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)  # (cos, sin)

    def forward_components(self, x):
        """Apply the convolutional trunk to an image-format input.

        Args:
            x: Input image tensor of shape
                ``(batch, input_channels, grid_rows, grid_cols)``.

        Returns:
            Per-element (cos, sin) pairs of shape ``(batch, num_elements, 2)``;
            use :meth:`forward` to obtain angles.
        """
        b = x.size(0)
        return (self.out_conv(self.conv_net(x))
                .view(b, 2, self.num_elements).permute(0, 2, 1))


class TransformerModel(BaseModel):
    """Transformer encoder for RIS phase prediction.

    Treats each RIS element as a token in a sequence.  A shared feature
    encoder maps the input CSI vector to a single context embedding; learnable
    positional embeddings are added per element before the transformer encoder
    processes inter-element attention.  A final per-token linear projection
    yields per-element phase shifts.

    Args:
        input_dim: Dimensionality of the input feature vector.
        num_elements: Number of RIS elements (sequence length).
        hidden_dim: Unused (overridden by ``config.TRANSFORMER_D_MODEL``).
        num_layers: Unused (overridden by ``config.TRANSFORMER_NUM_LAYERS``).
        dropout: Dropout probability in encoder layers.
        config: Optional :class:`Config` for ``TRANSFORMER_D_MODEL``,
            ``TRANSFORMER_NUM_HEADS``, ``TRANSFORMER_NUM_LAYERS``,
            ``TRANSFORMER_FF_DIM``.
    """

    def __init__(self, input_dim: int, num_elements: int, hidden_dim: int,
                 num_layers: int, dropout: float, config=None):
        """Build the feature encoder, positional embedding, and transformer encoder stack."""
        super().__init__()
        self.num_elements = num_elements
        
        d_model = getattr(config, 'TRANSFORMER_D_MODEL', 256) if config else 256
        nhead = getattr(config, 'TRANSFORMER_NUM_HEADS', 8) if config else 8
        num_layers_tf = getattr(config, 'TRANSFORMER_NUM_LAYERS', 4) if config else 4
        dim_feedforward = getattr(config, 'TRANSFORMER_FF_DIM', 512) if config else 512
        
        self.d_model = d_model
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_model, d_model),
        )
        
        # Position encoding
        self.pos_emb = nn.Parameter(torch.randn(1, num_elements, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_tf)
        
        self.out_proj = nn.Linear(d_model, 2)  # (cos, sin) per element

    def forward_components(self, x):
        """Forward pass: encode CSI, add positional embeddings, apply transformer.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Per-element (cos, sin) pairs of shape ``(batch, num_elements, 2)``;
            use :meth:`forward` to obtain angles.
        """
        b = x.size(0)
        context = self.feature_encoder(x)
        h = context.unsqueeze(1) + self.pos_emb.expand(b, -1, -1)
        h = self.transformer(h)
        return self.out_proj(h)  # (batch, num_elements, 2)


def create_model(
    model_type: str,
    input_dim: int,
    num_elements: int,
    hidden_dim: int | None = None,
    num_layers: int | None = None,
    dropout: float | None = None,
    config = None
) -> nn.Module:
    """
    Factory function to create the requested neural network model.
    
    Args:
        model_type: Type of the model (e.g., 'MLP', 'GNN', 'CNN', 'Transformer')
        input_dim: Dimension of the input features
        num_elements: Number of elements in the RIS tile
        hidden_dim: Number of hidden units
        num_layers: Number of layers
        dropout: Dropout probability
        config: Configuration object containing hyperparameters
        
    Returns:
        PyTorch model instance inheriting from nn.Module
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
