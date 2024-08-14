import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels):
        super().__init__()

        # (batch_size, in_channels, img_sz, img_sz) -> (batch_size, embed_dim, num_patches)
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size),
            nn.Flatten(2))
        
        # Learnable embedding (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.positional_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (1, 1, embed_dim) -> (batch_size, 1, embed_dim)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # (batch_size, embed_dim, num_patches) -> (batch_size, num_patches, embed_dim)
        x = self.patcher(x).permute(0, 2, 1)
        # Concatenate with the cls token (batch_size, num_patches + 1, embed_dim)
        x = torch.cat([cls_token, x], dim=1)
        x = self.positional_embeddings + x 
        x = self.dropout(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_channels, nhead, activation, num_layers, num_classes):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        # (batch_size, in_channels, img_sz, img_sz) -> (batch_size, num_patches + 1, embed_dim)
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        # Classifying on the cls token only (batch_size, num_patches + 1, embed_dim) -> (batch_size, embed_dim)
        x = self.mlp_head(x[:, 0, :])
        return x
