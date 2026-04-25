import torch


def torch_nanminmax(in_tensor, extrema, dim=None, keepdim=False):
    """Equivalent of np.nanmin and np.nanmax for batched tensors."""
    assert extrema in ['min', 'max'], "extrema can only be min, max"
    assert type(dim) == tuple, "dim must be a tuple"

    tmp = ~torch.isnan(in_tensor)
    for d in sorted(dim, reverse=True):
        tmp = tmp.any(dim=d, keepdim=keepdim)
    in_tensor_valid = tmp

    if extrema == 'min':
        max_val_dtype = torch.finfo(in_tensor.dtype).max
        result = torch.nan_to_num(in_tensor, nan=max_val_dtype).amin(dim=dim, keepdim=keepdim)
    else:
        min_val_dtype = torch.finfo(in_tensor.dtype).min
        result = torch.nan_to_num(in_tensor, nan=min_val_dtype).amax(dim=dim, keepdim=keepdim)

    return torch.where(
        in_tensor_valid,
        result,
        torch.tensor(float('nan'), dtype=in_tensor.dtype, device=in_tensor.device),
    )
