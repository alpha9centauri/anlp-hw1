from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query/key.
    query: (bs, seqlen, n_local_heads, head_dim)
    key:   (bs, seqlen, n_local_kv_heads, head_dim)
    """
    _, seqlen, _, _ = query.shape
    device = query.device

    # RoPE requires pairing dimensions -> head_dim must be even
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Build inverse frequencies for each 2D pair:
    # inv_freq[i] = 1 / (theta^(2i/head_dim)), i=0..head_dim/2-1
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))  # (head_dim//2,)

    # Positions for current sequence length
    t = torch.arange(seqlen, device=device).float()  # (seqlen,)

    # Angles: outer product -> (seqlen, head_dim//2)
    freqs = torch.outer(t, inv_freq)

    # cos/sin for rotation
    cos = torch.cos(freqs)  # (seqlen, head_dim//2)
    sin = torch.sin(freqs)  # (seqlen, head_dim//2)

    # reshape for broadcasting to query/key shapes
    cos_q = reshape_for_broadcast(cos, query.float().reshape(query.shape[:-1] + (-1, 2))[..., 0])
    sin_q = reshape_for_broadcast(sin, query.float().reshape(query.shape[:-1] + (-1, 2))[..., 0])

    cos_k = reshape_for_broadcast(cos, key.float().reshape(key.shape[:-1] + (-1, 2))[..., 0])
    sin_k = reshape_for_broadcast(sin, key.float().reshape(key.shape[:-1] + (-1, 2))[..., 0])

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Complex rotation:
    # (a + ib) * (cos + i sin) = (a cos - b sin) + i(a sin + b cos)
    q_rot_real = query_real * cos_q - query_imag * sin_q
    q_rot_imag = query_real * sin_q + query_imag * cos_q

    k_rot_real = key_real * cos_k - key_imag * sin_k
    k_rot_imag = key_real * sin_k + key_imag * cos_k

    # stack back real/imag pairs and flatten to original last dim
    query_out = torch.stack((q_rot_real, q_rot_imag), dim=-1).flatten(-2)
    key_out = torch.stack((k_rot_real, k_rot_imag), dim=-1).flatten(-2)

    # cast back to original dtypes
    query_out = query_out.type_as(query)
    key_out = key_out.type_as(key)

    return query_out, key_out
