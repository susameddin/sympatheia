"""
Valence-Arousal Embedding Module

Projects continuous [valence, arousal] values to the model's embedding space.
This allows the model to condition on emotion in a continuous, interpolatable way.
"""

import torch
import torch.nn as nn


class VAEmbedding(nn.Module):
    """
    Projects continuous [valence, arousal] values to embedding space.

    The embedding can be prepended to the input sequence, allowing the model
    to attend to emotion information throughout generation.

    Args:
        embed_dim: Output embedding dimension (should match model's hidden size)
        hidden_dim: Hidden layer dimension in the projection MLP
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP with GELU activation
        # Input: [valence, arousal] (2D)
        # Output: embedding vector (embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        # Initialize with small weights for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, va: torch.Tensor) -> torch.Tensor:
        """
        Project valence-arousal values to embedding space.

        Args:
            va: Tensor of shape [batch_size, 2] containing [valence, arousal] pairs.
                Values should be in range [-1, 1].

        Returns:
            Tensor of shape [batch_size, 1, embed_dim] - the VA embedding
            ready to be prepended to the input sequence.
        """
        # Project to embedding space
        emb = self.proj(va)  # [batch_size, embed_dim]

        # Add sequence dimension for concatenation with input embeddings
        return emb.unsqueeze(1)  # [batch_size, 1, embed_dim]

    def save(self, path: str):
        """Save the VA embedding weights."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, embed_dim: int, hidden_dim: int = 64, device: str = "cpu"):
        """Load VA embedding weights from file."""
        model = cls(embed_dim=embed_dim, hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        return model


if __name__ == "__main__":
    # Quick test
    print("Testing VAEmbedding module...")

    # GLM-4-Voice uses 4096 hidden dimension
    embed_dim = 4096
    va_embed = VAEmbedding(embed_dim=embed_dim, hidden_dim=64)

    # Test with batch of VA values
    batch_size = 4
    va_values = torch.tensor([
        [0.8, 0.6],    # happy
        [0.0, 0.0],    # neutral
        [-0.7, -0.3],  # sad
        [0.4, 0.3],    # interpolated
    ])

    output = va_embed(va_values)
    print(f"Input shape: {va_values.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: [{batch_size}, 1, {embed_dim}]")
    assert output.shape == (batch_size, 1, embed_dim), "Shape mismatch!"
    print("Test passed!")
