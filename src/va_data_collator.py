"""
Custom Data Collator for VA-conditioned training.

Handles both text tokenization and VA value extraction from the dataset.
"""

import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin


@dataclass
class VADataCollator(DataCollatorMixin):
    """
    Data collator that handles both text tokenization and VA value extraction.

    Expected dataset format:
    {
        "text": "<|system|>\n...<|user|>\n...<|assistant|>\n...",
        "valence": 0.8,
        "arousal": 0.6
    }
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048
    padding: Union[bool, str] = True
    truncation: bool = True
    return_tensors: str = "pt"
    label_pad_token_id: int = -100  # Ignore index for loss
    text_field: str = "text"
    valence_field: str = "valence"
    arousal_field: str = "arousal"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features.

        Args:
            features: List of dataset examples, each with 'text', 'valence', 'arousal'

        Returns:
            Batch dict with:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - labels: [batch, seq_len]
                - va_values: [batch, 2]
        """
        # Extract texts and VA values
        texts = [f[self.text_field] for f in features]
        valences = [f.get(self.valence_field, 0.0) for f in features]
        arousals = [f.get(self.arousal_field, 0.0) for f in features]

        # Tokenize texts
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        # Create labels (same as input_ids, with padding tokens masked)
        labels = batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        batch["labels"] = labels

        # Create VA values tensor
        va_values = torch.tensor(
            [[v, a] for v, a in zip(valences, arousals)],
            dtype=torch.float32
        )
        batch["va_values"] = va_values

        return batch


@dataclass
class VADataCollatorForCausalLM(DataCollatorMixin):
    """
    Data collator for causal LM training with VA conditioning.

    This version handles the causal LM setup where labels are shifted
    input_ids (the model handles the shift internally).
    """

    tokenizer: PreTrainedTokenizerBase
    max_length: int = 2048
    padding: Union[bool, str] = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    label_pad_token_id: int = -100
    text_field: str = "text"
    valence_field: str = "valence"
    arousal_field: str = "arousal"
    mlm: bool = False  # For compatibility with some trainers

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features for causal LM training.

        Args:
            features: List of dataset examples

        Returns:
            Batch dict with input_ids, attention_mask, labels, va_values
        """
        # Extract texts and VA values
        texts = []
        valences = []
        arousals = []

        for f in features:
            texts.append(f[self.text_field])
            valences.append(float(f.get(self.valence_field, 0.0)))
            arousals.append(float(f.get(self.arousal_field, 0.0)))

        # Tokenize texts
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors=self.return_tensors,
        )

        # For causal LM, labels = input_ids (model handles the shift)
        labels = batch["input_ids"].clone()

        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id

        batch["labels"] = labels

        # Create VA values tensor
        va_values = torch.tensor(
            [[v, a] for v, a in zip(valences, arousals)],
            dtype=torch.float32
        )
        batch["va_values"] = va_values

        return batch


def remove_va_from_text(text: str) -> str:
    """
    Remove the text-based VA conditioning from system prompt.

    Converts:
        "<|system|>\nPlease respond in English. User emotion (valence=0.80, arousal=0.60)\n<|user|>"
    To:
        "<|system|>\nPlease respond in English.\n<|user|>"
    """
    import re

    # Pattern to match "User emotion (valence=X.XX, arousal=X.XX)"
    pattern = r"\s*User emotion \(valence=[^)]+, arousal=[^)]+\)"
    return re.sub(pattern, "", text)


def preprocess_va_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess a single example to remove text-based VA from system prompt.

    Use with dataset.map():
        dataset = dataset.map(preprocess_va_dataset)
    """
    example["text"] = remove_va_from_text(example["text"])
    return example


if __name__ == "__main__":
    print("Testing VADataCollator...")

    # Mock tokenizer for testing
    from transformers import AutoTokenizer

    # Use a simple test without loading the full tokenizer
    class MockTokenizer:
        pad_token_id = 0

        def __call__(self, texts, **kwargs):
            # Simple mock tokenization
            batch_size = len(texts)
            seq_len = 20
            return {
                "input_ids": torch.randint(1, 100, (batch_size, seq_len)),
                "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            }

    tokenizer = MockTokenizer()

    # Test data
    features = [
        {"text": "Hello world", "valence": 0.8, "arousal": 0.6},
        {"text": "Goodbye world", "valence": -0.7, "arousal": -0.3},
        {"text": "Neutral text", "valence": 0.0, "arousal": 0.0},
    ]

    collator = VADataCollator(
        tokenizer=tokenizer,
        max_length=20,
    )

    batch = collator(features)

    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"labels shape: {batch['labels'].shape}")
    print(f"va_values shape: {batch['va_values'].shape}")
    print(f"va_values:\n{batch['va_values']}")

    assert batch["input_ids"].shape[0] == 3, "Batch size mismatch"
    assert batch["va_values"].shape == (3, 2), "VA values shape mismatch"
    assert torch.allclose(batch["va_values"][0], torch.tensor([0.8, 0.6])), "VA values content mismatch"

    # Test text preprocessing
    test_text = "<|system|>\nPlease respond in English. User emotion (valence=0.80, arousal=0.60)\n<|user|>\nHello"
    cleaned = remove_va_from_text(test_text)
    print(f"\nOriginal text: {test_text}")
    print(f"Cleaned text: {cleaned}")
    assert "valence=" not in cleaned, "VA not removed from text"

    print("\nAll tests passed!")
