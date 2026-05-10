
# optimization/compressor.py
"""
GRADIENT COMPRESSOR - Reduces communication overhead in federated learning.

PROBLEM: Sending full model weights every round is expensive.
  A modern ResNet has 25M parameters × 4 bytes = 100MB per client per round!
  With 100 clients × 10 rounds = 100GB of traffic. Not practical.

SOLUTIONS IMPLEMENTED:

1. Top-K Sparsification (keep top K% of gradients)
   - Only transmit the LARGEST gradient values
   - Zero out small gradients (they contribute little)
   - Reduces payload by (1 - compression_ratio)

2. Quantization (float32 → int8)
   - Convert 32-bit floats to 8-bit integers
   - 4x compression with minimal accuracy loss
   - Dequantize on server before aggregation

Real-world impact:
  Top-30% sparsification + 8-bit quantization
  → ~87% reduction in communication cost
"""
import numpy as np
import torch

class GradientCompressor:
    def __init__(self, config):
        self.ratio = config.compression_ratio   # fraction to KEEP
        self.bits  = config.quantize_bits       # 8 for int8 quantization

    def compress(self, weight_delta):
        """
        Compress weight update before transmission.
        Returns: (compressed_delta, bytes_used)
        """
        compressed = {}
        total_bytes = 0

        for key, tensor in weight_delta.items():
            flat = tensor.flatten()

            # Step 1: Top-K Sparsification
            k = max(1, int(len(flat) * self.ratio))  # keep top K%
            topk_vals, topk_idx = torch.topk(flat.abs(), k)

            # Create sparse update (mask of non-zero positions)
            sparse = torch.zeros_like(flat)
            sparse[topk_idx] = flat[topk_idx]

            # Step 2: Quantize to int8
            max_val = sparse.abs().max().item() + 1e-8
            scale = max_val / 127.0

            # Quantize: float → int8
            quantized = (sparse / scale).clamp(-127, 127).to(torch.int8)

            # Store with dequantization metadata
            compressed[key] = {
                'quantized': quantized,
                'scale': scale,
                'shape': tensor.shape
            }

            # Count bytes: indices (4B each) + values (1B each) + scale (4B)
            total_bytes += k * 4 + k * 1 + 4  # index + int8_val + scale

        return self._decompress(compressed), total_bytes

    def _decompress(self, compressed):
        """Convert compressed dict back to regular tensor dict."""
        result = {}
        for key, data in compressed.items():
            dequantized = data['quantized'].float() * data['scale']
            result[key] = dequantized.reshape(data['shape'])
        return result

    @staticmethod
    def baseline_bytes(weight_dict):
        """How many bytes WITHOUT compression (float32)."""
        return sum(v.numel() * 4 for v in weight_dict.values())
