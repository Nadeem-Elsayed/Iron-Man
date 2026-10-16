import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class JesterSequenceLSTM(nn.Module):
    """
    DirectML-compatible hybrid CNN + 1D-Conv temporal model for 4-class Jester gesture recognition.

    Architecture constraints (DO NOT VIOLATE — see CONTEXT.md §3):
      - No Conv3d: DirectML fails on 5D volumetric tensors.
      - No nn.LSTM: DirectML lacks the fused LSTM kernel operator.

    Temporal pooling is configurable via `pool_mode`:
      - 'avg'     : AdaptiveAvgPool1d(1) collapses T → 1 vector.
                    Simpler, but averages out directional motion cues.
      - 'flatten' : Concatenates all T hidden-dim vectors → fc input of size hidden_dim * T.
                    Preserves Left-to-Right vs Right-to-Left trajectory order,
                    which is critical for discriminating Swiping Left from Swiping Right.
                    Default and recommended for maximising swipe precision/recall.

    The attribute `self.lstm` holds the Conv1d layer; the name is intentional so that
    the optimizer parameter groups in train.py (`model.lstm.parameters()`) do not break.
    """

    VALID_POOL_MODES = ('avg', 'flatten')

    def __init__(self, num_classes: int = 4, hidden_dim: int = 256,
                 pool_mode: str = 'flatten', sequence_length: int = 15,
                 dropout: float = 0.5, train_layer4: bool = False,
                 spatial_zones: int = 3):
        """
        Args:
            num_classes:     Number of output gesture classes (default 4 for Jester subset).
            hidden_dim:      Output channels of the temporal Conv1d layer.
            pool_mode:       'avg' or 'flatten' — controls how temporal features are aggregated.
            sequence_length: Number of frames per clip; only used when pool_mode='flatten'
                             to compute the FC input dimension (hidden_dim * sequence_length).
            dropout:         Dropout probability applied before the FC head (default 0.5).
                             Prevents the model from memorising dominant-class (Doing other things)
                             feature patterns instead of learning genuine swipe trajectories.
                             Set to 0.0 to disable.
            train_layer4:    Whether to unfreeze and train layer4. Kept False by default
                             for fast, stable execution on DirectML (RX 7800 XT).
            spatial_zones:   Number of horizontal zones across width (default 3: Left, Center, Right).
                             Fixes the fundamental Left vs Right ambiguity of global avgpool(1, 1).
        """
        if pool_mode not in self.VALID_POOL_MODES:
            raise ValueError(f"pool_mode must be one of {self.VALID_POOL_MODES}, got '{pool_mode}'")

        super().__init__()

        self.pool_mode = pool_mode
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.train_layer4 = train_layer4
        self.spatial_zones = spatial_zones

        # ------------------------------------------------------------------
        # 1. Standard 2D ResNet50 backbone (fully supported by DirectML).
        #    If spatial_zones > 1: drop both FC and avgpool to preserve 7x7 spatial grid.
        #    Then apply AdaptiveAvgPool2d((1, spatial_zones)) to keep Left, Center, Right.
        # ------------------------------------------------------------------
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        if spatial_zones > 1:
            modules = list(resnet.children())[:-2]  # drop FC and original 1x1 avgpool
            self.spatial_pool = nn.AdaptiveAvgPool2d((1, spatial_zones))
            in_channels = 2048 * spatial_zones      # e.g., 2048 * 3 = 6144
        else:
            modules = list(resnet.children())[:-1]  # drop only FC
            self.spatial_pool = None
            in_channels = 2048

        self.cnn = nn.Sequential(*modules)

        # Freeze all layers by default
        for param in self.cnn.parameters():
            param.requires_grad = False

        # Optionally unfreeze layer4 (index 7 in modules)
        if self.train_layer4:
            for param in self.cnn[7].parameters():
                param.requires_grad = True

        # ------------------------------------------------------------------
        # 2. DirectML-compatible temporal layer (replaces nn.LSTM).
        #    Named 'lstm' for train.py optimizer-group compatibility.
        # ------------------------------------------------------------------
        self.lstm = nn.Conv1d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # ------------------------------------------------------------------
        # 3. Temporal aggregation + regularisation + classifier head.
        # ------------------------------------------------------------------
        if pool_mode == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
            fc_in_features = hidden_dim
        else:  # 'flatten'
            self.pool = None
            fc_in_features = hidden_dim * sequence_length

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(fc_in_features, num_classes)

    def train(self, mode: bool = True):
        """
        Override train() to keep the frozen ResNet backbone in eval mode at all times,
        preventing BatchNorm running-stat drift on the frozen layers (see CONTEXT.md §3).
        """
        super().train(mode)
        self.cnn.eval()  # always keep backbone in eval — BatchNorm must not drift
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Float tensor of shape [B, C, T, H, W]
               (B=batch, C=3 RGB channels, T=time/sequence, H=height, W=width)
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        batch_size, channels, time_steps, h, w = x.size()

        # Merge B and T so 2D CNN processes each frame
        c_in = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time_steps, channels, h, w)
        
        if self.train_layer4:
            c_out = self.cnn(c_in)
        else:
            with torch.no_grad():
                c_out = self.cnn(c_in)

        # Apply 3-zone spatial pooling: Left, Center, Right
        if self.spatial_zones > 1:
            # c_out is [B*T, 2048, 7, 7] -> pool to [B*T, 2048, 1, spatial_zones] -> [B, T, 2048 * spatial_zones]
            c_out = self.spatial_pool(c_out).view(batch_size, time_steps, 2048 * self.spatial_zones)
        else:
            c_out = c_out.view(batch_size, time_steps, 2048)

        # Reshape for Conv1d: [B, Channels, Time=T]
        r_in = c_out.permute(0, 2, 1)

        # Temporal convolution
        t_out = self.relu(self.lstm(r_in))  # [B, hidden_dim, T]

        # Temporal aggregation
        if self.pool_mode == 'avg':
            t_out = self.pool(t_out).view(batch_size, -1)      # [B, hidden_dim]
        else:
            t_out = t_out.permute(0, 2, 1).reshape(batch_size, -1)  # [B, hidden_dim * T]

        t_out = self.dropout(t_out)
        return self.fc(t_out)