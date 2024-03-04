import numpy as np

def scaled_dot_product_attention(query, key, value):
    matmul_qk = np.matmul(query, key.transpose(-1, -2))
    dk = query.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)
    attention_weights = softmax(scaled_attention_logits, axis=-1)
    output = np.matmul(attention_weights, value)
    return output, attention_weights

def multi_head_attention(query, key, value, num_heads):
    depth = query.shape[-1] // num_heads
    query, key, value = [
        split_heads(x, num_heads, depth) for x in [query, key, value]]
    attention_output, _ = scaled_dot_product_attention(query, key, value)
    attention_output = concatenate_heads(attention_output, num_heads, depth)
    return attention_output

def pointwise_feed_forward_network(d_model, dff):
    return np.array([np.random.rand(d_model, dff), 
                     np.random.rand(dff), 
                     np.random.rand(dff, d_model), 
                     np.random.rand(d_model)])

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return pos_encoding

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def softmax(x, axis=None):
    x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

def split_heads(x, num_heads, depth):
    x = x.reshape(*x.shape[:-1], num_heads, depth)
    return np.transpose(x, [0, 2, 1, 3])

def concatenate_heads(x, num_heads, depth):
    x = np.transpose(x, [0, 2, 1, 3])
    return x.reshape(*x.shape[:-2], num_heads*depth)
