"""Mock AirLLM for testing API structure without heavy dependencies."""

import torch


class MockTokenizer:
    """Mock tokenizer for API testing."""
    def __init__(self):
        self.vocab_size = 32000
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
    
    def encode(self, text, **kwargs):
        """Mock encoding - returns list of token IDs."""
        return [1, 2, 3, 4, 5]
    
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Mock decoding - returns generated text based on token IDs."""
        # Simple mock: generate text based on token count
        if isinstance(token_ids, torch.Tensor):
            num_tokens = len(token_ids) if token_ids.dim() == 1 else token_ids.shape[-1]
        else:
            num_tokens = len(token_ids)
        
        # Generate mock response proportional to token count
        mock_responses = [
            "The answer is an important mathematical concept.",
            "This is a complex topic that requires careful consideration.",
            "The result demonstrates a fundamental principle of computation.",
            "This example shows how systems process information efficiently.",
            "The output illustrates key concepts in modern computing.",
            "This represents an optimal solution for the given problem.",
            "The calculation follows established mathematical principles.",
            "This demonstrates the power of algorithmic thinking.",
        ]
        
        # Pick response based on token hash for some pseudo-randomness
        response = mock_responses[sum(token_ids) % len(mock_responses)]
        return response
    
    def __call__(self, text, return_tensors=None, **kwargs):
        """Make tokenizer callable to work like HuggingFace tokenizers.
        
        Args:
            text: Input text or list of texts
            return_tensors: Type of tensor to return ('pt' for PyTorch, None for lists)
            **kwargs: Additional tokenization parameters (ignored for mock)
        
        Returns:
            Dict with 'input_ids' and optionally 'attention_mask'
        """
        # Handle single or batch input
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        # Mock tokenization: create token sequences
        # In real scenario, this would tokenize each text
        token_sequences = []
        for t in texts:
            # Simple mock: split by words and create token IDs
            words = t.split()
            tokens = [hash(word) % 30000 + 1 for word in words] if words else [2]
            token_sequences.append(tokens)
        
        # Pad to same length
        max_len = max(len(seq) for seq in token_sequences) if token_sequences else 1
        padded = []
        attention_masks = []
        for seq in token_sequences:
            padded_seq = seq + [self.pad_token_id] * (max_len - len(seq))
            padded.append(padded_seq)
            attention_masks.append([1] * len(seq) + [0] * (max_len - len(seq)))
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }
        elif return_tensors == "np":
            import numpy as np
            return {
                "input_ids": np.array(padded, dtype=np.int64),
                "attention_mask": np.array(attention_masks, dtype=np.int64),
            }
        else:
            # Return as lists
            return {
                "input_ids": padded,
                "attention_mask": attention_masks,
            }


class MockAutoModel:
    """Mock model for API testing."""
    def __init__(self, *args, **kwargs):
        self.config = type('Config', (), {
            'architectures': ['LlamaForCausalLM'],
            'num_parameters': '1B',
            'hidden_size': 768,
            'vocab_size': 32000,
        })()
        self.tokenizer = MockTokenizer()
    
    def generate(self, input_ids, max_new_tokens=100, return_dict_in_generate=False, **kwargs):
        """Mock generation - returns token IDs or GenerateOutput object."""
        batch_size = input_ids.shape[0]
        # Generate mock output tokens
        output_ids = torch.cat([
            input_ids,
            torch.randint(1, 30000, (batch_size, max_new_tokens), dtype=torch.long)
        ], dim=1)
        
        if return_dict_in_generate:
            # Return HuggingFace-style GenerateOutput object
            return type('GenerateOutput', (), {
                'sequences': output_ids,
            })()
        return output_ids
    
    def __call__(self, *args, **kwargs):
        """Mock forward pass."""
        return type('Output', (), {
            'last_hidden_state': torch.randn(1, 100, 768),
            'logits': torch.randn(1, 100, 32000)
        })()


class AutoModel:
    """Mock AutoModel.from_pretrained()."""
    @staticmethod
    def from_pretrained(model_name, **kwargs):
        """Load mock model."""
        return MockAutoModel()


# Export for module loading
__all__ = ['AutoModel']
