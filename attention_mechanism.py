import numpy as np
import math

""" see ana_attention.py for instructions on beam search """

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(query, key, value):
    """
    Computes the scaled dot-product attention.

    Args:
    query: Query matrix of shape (..., seq_len_q, depth)
    key: Key matrix of shape (..., seq_len_k, depth)
    value: Value matrix of shape (..., seq_len_v, depth_v)
    
    Returns:
    The output after applying attention to the value matrix and the attention weights.
    """
    # Calculate the dot product between query and key
    matmul_qk = np.matmul(query, key.transpose(0, 2, 1))
    
    # Scale matmul_qk
    depth = query.shape[-1]
    scaled_attention_logits = matmul_qk / math.sqrt(depth)
    
    # Apply softmax to get the weights
    attention_weights = softmax(scaled_attention_logits)
    
    # Apply the attention weights to the values
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights

# Example usage
np.random.seed(0)
temp_query = np.random.rand(1, 60, 512)  # (batch_size, seq_len_q, depth)
temp_key = np.random.rand(1, 70, 512)    # (batch_size, seq_len_k, depth)
temp_value = np.random.rand(1, 70, 512)  # (batch_size, seq_len_v, depth_v)

output, attention_weights = scaled_dot_product_attention(temp_query, temp_key, temp_value)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
