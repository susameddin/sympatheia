"""
VA Model Wrapper for GLM-4-Voice

Wraps the GLM-4-Voice model to inject valence-arousal embeddings
by adding a placeholder token at the start of the sequence and replacing
its embedding with the VA embedding.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from transformers import AutoModel

try:
    from .va_embedding import VAEmbedding
except ImportError:
    from va_embedding import VAEmbedding


class VAGLMWrapper(nn.Module):
    """
    Wrapper for GLM-4-Voice that injects VA embeddings into the input sequence.

    Approach:
    1. Prepend a placeholder token (pad_token) to input_ids
    2. Use a forward hook to REPLACE that token's embedding with VA embedding
    3. This gives VA its own position without mixing with other token meanings

    The model sees: [VA_embed, <|system|>, ..., token_n]
    where VA_embed completely replaces the placeholder token's embedding.
    """

    def __init__(
        self,
        base_model: nn.Module,
        va_hidden_dim: int = 64,
        freeze_va_embedding: bool = False,
        pad_token_id: int = 0,  # Token ID to use as placeholder
    ):
        """
        Args:
            base_model: The base GLM-4-Voice model (can be PEFT-wrapped)
            va_hidden_dim: Hidden dimension for VA projection MLP
            freeze_va_embedding: If True, don't train the VA embedding
            pad_token_id: Token ID to use as placeholder for VA position
        """
        super().__init__()

        self.base_model = base_model
        self.pad_token_id = pad_token_id

        # Get embedding dimension from model config
        if hasattr(base_model, "config"):
            self.embed_dim = base_model.config.hidden_size
        else:
            # PEFT model - access base model config
            self.embed_dim = base_model.base_model.config.hidden_size

        # Create VA embedding layer
        self.va_embedding = VAEmbedding(
            embed_dim=self.embed_dim,
            hidden_dim=va_hidden_dim
        )

        if freeze_va_embedding:
            for param in self.va_embedding.parameters():
                param.requires_grad = False

        # Store config for compatibility
        self.config = base_model.config if hasattr(base_model, "config") else base_model.base_model.config

        # Storage for VA values during forward pass (used by hook)
        self._current_va_values = None
        self._hook_handle = None

    def _get_embedding_layer(self):
        """Get the word embedding layer from the base model."""
        # Navigate through PEFT wrapper if present
        model = self.base_model
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model"):
            model = model.model

        # ChatGLM structure: model.transformer.embedding
        if hasattr(model, "transformer"):
            if hasattr(model.transformer, "embedding"):
                return model.transformer.embedding

        # Fallback: try to find embedding layer
        if hasattr(model, "get_input_embeddings"):
            return model.get_input_embeddings()

        raise AttributeError("Cannot find embedding layer in base model")

    def _embedding_hook(self, module, input, output):
        """
        Forward hook that REPLACES the first token's embedding with VA embedding.

        The first token is a placeholder (pad token) that we added.
        We completely replace its embedding with the computed VA embedding.
        """
        if self._current_va_values is None:
            return output

        # output shape: [batch, seq_len, embed_dim] or could be a tuple
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure VA embedding module is on the correct device
        if next(self.va_embedding.parameters()).device != device:
            self.va_embedding = self.va_embedding.to(device)

        # Get VA embedding: [batch, 1, embed_dim]
        va_embed = self.va_embedding(self._current_va_values.to(device))
        va_embed = va_embed.to(dtype)

        # REPLACE the first token's embedding with VA embedding
        hidden_states = hidden_states.clone()
        hidden_states[:, 0:1, :] = va_embed

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states

    def _register_hook(self):
        """Register the embedding hook."""
        if self._hook_handle is not None:
            return  # Already registered

        embedding_layer = self._get_embedding_layer()
        self._hook_handle = embedding_layer.register_forward_hook(self._embedding_hook)

    def _remove_hook(self):
        """Remove the embedding hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _prepend_placeholder(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        """
        Prepend a placeholder token to the inputs.

        This token's embedding will be replaced with VA embedding by the hook.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Create placeholder token (using pad_token_id)
        placeholder = torch.full(
            (batch_size, 1),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=device
        )

        # Prepend placeholder to input_ids
        input_ids = torch.cat([placeholder, input_ids], dim=1)

        # Extend attention mask
        if attention_mask is not None:
            va_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([va_mask, attention_mask], dim=1)

        # Extend labels with -100 (ignore in loss calculation)
        if labels is not None:
            ignore_label = torch.full(
                (batch_size, 1),
                -100,
                dtype=labels.dtype,
                device=device
            )
            labels = torch.cat([ignore_label, labels], dim=1)

        return input_ids, attention_mask, labels

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        va_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass with optional VA embedding injection.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            va_values: Valence-arousal values [batch, 2]. If None, no VA injection.
            labels: Labels for language modeling loss [batch, seq_len]
            ... other standard transformer args

        Returns:
            Model outputs with loss, logits, etc.
        """
        # Store VA values for the hook
        self._current_va_values = va_values

        if va_values is not None:
            # Register hook to replace placeholder embedding with VA embedding
            self._register_hook()

            # Prepend placeholder token to inputs
            input_ids, attention_mask, labels = self._prepend_placeholder(
                input_ids, attention_mask, labels
            )

            # Handle position_ids if provided
            if position_ids is not None:
                batch_size = position_ids.shape[0]
                device = position_ids.device
                va_position = torch.zeros(batch_size, 1, dtype=position_ids.dtype, device=device)
                position_ids = torch.cat([va_position, position_ids + 1], dim=1)
        else:
            self._remove_hook()

        # Call base model
        try:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )
        finally:
            # Clear VA values after forward pass
            self._current_va_values = None

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        va_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate with VA conditioning.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            va_values: Valence-arousal values [batch, 2]
            **kwargs: Other generation arguments
        """
        # Store VA values for the hook
        self._current_va_values = va_values

        if va_values is not None:
            self._register_hook()

            # Prepend placeholder token
            input_ids, attention_mask, _ = self._prepend_placeholder(
                input_ids, attention_mask, labels=None
            )
        else:
            self._remove_hook()

        try:
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            self._current_va_values = None

        return outputs

    def save_va_embedding(self, path: str):
        """Save only the VA embedding weights."""
        self.va_embedding.save(path)

    def load_va_embedding(self, path: str, device: str = "cpu"):
        """Load VA embedding weights."""
        state_dict = torch.load(path, map_location=device, weights_only=True)

        # Handle different save formats
        if any('va_embedding.' in k for k in state_dict.keys()):
            # New format: keys like 'va_embedding.proj.0.weight'
            # Strip the 'va_embedding.' prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('va_embedding.'):
                    new_key = k.replace('va_embedding.', '')
                    new_state_dict[new_key] = v
            state_dict = new_state_dict

        self.va_embedding.load_state_dict(state_dict)

    def print_trainable_parameters(self):
        """Print number of trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # VA embedding params
        va_params = sum(p.numel() for p in self.va_embedding.parameters())
        va_trainable = sum(p.numel() for p in self.va_embedding.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"VA embedding parameters: {va_params:,} (trainable: {va_trainable:,})")

        # If base model has print_trainable_parameters (PEFT)
        if hasattr(self.base_model, "print_trainable_parameters"):
            print("\nBase model (LoRA):")
            self.base_model.print_trainable_parameters()


def create_va_model(
    model_path: str,
    va_hidden_dim: int = 64,
    freeze_va_embedding: bool = False,
    device_map: str = "auto",
) -> VAGLMWrapper:
    """
    Create a VA-wrapped GLM-4-Voice model.

    Args:
        model_path: Path to base model or HuggingFace model ID
        va_hidden_dim: Hidden dimension for VA projection
        freeze_va_embedding: Whether to freeze VA embedding during training
        device_map: Device map for model loading

    Returns:
        VAGLMWrapper instance
    """
    # Load base model
    base_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device_map,
    )

    # Wrap with VA embedding
    wrapper = VAGLMWrapper(
        base_model=base_model,
        va_hidden_dim=va_hidden_dim,
        freeze_va_embedding=freeze_va_embedding,
    )

    return wrapper


if __name__ == "__main__":
    print("Testing VAGLMWrapper with placeholder token approach...")

    # Test with a mock model
    class MockEmbedding(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)

        def forward(self, x):
            return self.embedding(x)

    class MockTransformer(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embedding = MockEmbedding(1000, embed_dim)

        def forward(self, inputs_embeds, attention_mask=None):
            return inputs_embeds

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = 256
            self.transformer = MockTransformer(self.embed_dim)

        class Config:
            hidden_size = 256

        config = Config()

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            embeds = self.transformer.embedding(input_ids)
            batch_size, seq_len, _ = embeds.shape
            return type('Output', (), {
                'loss': torch.tensor(0.0) if labels is not None else None,
                'logits': embeds,
                'shape': embeds.shape,
                'input_ids_shape': input_ids.shape,
                'attention_mask_shape': attention_mask.shape if attention_mask is not None else None,
                'labels_shape': labels.shape if labels is not None else None,
            })()

    # Test
    mock_model = MockModel()
    wrapper = VAGLMWrapper(mock_model, va_hidden_dim=32, pad_token_id=0)

    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))  # Avoid 0 (pad token)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(1, 1000, (batch_size, seq_len))
    va_values = torch.tensor([[0.8, 0.6], [-0.7, -0.3]])

    # Test forward with VA
    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        va_values=va_values,
    )

    print(f"Original input shape: [{batch_size}, {seq_len}]")
    print(f"After prepend - input_ids shape: {output.input_ids_shape}")
    print(f"After prepend - attention_mask shape: {output.attention_mask_shape}")
    print(f"After prepend - labels shape: {output.labels_shape}")
    print(f"Output logits shape: {output.logits.shape}")

    # Shapes should be seq_len + 1 because we prepended a placeholder
    assert output.input_ids_shape == (batch_size, seq_len + 1), f"input_ids shape mismatch!"
    assert output.attention_mask_shape == (batch_size, seq_len + 1), f"attention_mask shape mismatch!"
    assert output.labels_shape == (batch_size, seq_len + 1), f"labels shape mismatch!"
    assert output.logits.shape == (batch_size, seq_len + 1, 256), f"logits shape mismatch!"

    # Test forward without VA
    output_no_va = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        va_values=None,
    )
    print(f"\nWithout VA - input_ids shape: {output_no_va.input_ids_shape}")
    assert output_no_va.input_ids_shape == (batch_size, seq_len), "Shape should be unchanged without VA!"

    print("\nAll tests passed!")
    print("\nApproach: Prepend placeholder token, then REPLACE its embedding with VA embedding.")
    print("This gives VA its own position without mixing with other token meanings.")
