import torch


def torch_nanminmax(in_tensor, extrema, dim=None, keepdim=False):
    """Equivalent of np.nanmin and np.nanmax
    
    in_tensor is batched data: (B, C, H, W)
    extrema: 'min' or 'max'
    """
    assert extrema in ['min', 'max'], f"extrema can only be min, max"
    assert type(dim) == tuple, "dim must be a tuple"

    # Check each item in batch, see if tensor is valid (there is no nan)
    # any doesn't have dim, need to check each dim
    tmp = (~torch.isnan(in_tensor))
    for d in sorted(dim, reverse=True):     # descending
        tmp = tmp.any(dim=d, keepdim=keepdim)
    in_tensor_valid = tmp

    if extrema == 'min':
        max_val_dtype = torch.finfo(in_tensor.dtype).max
        result = torch.nan_to_num(in_tensor, nan=max_val_dtype).amin(dim=dim, keepdim=keepdim)
    else:
        min_val_dtype = torch.finfo(in_tensor.dtype).min
        result = torch.nan_to_num(in_tensor, nan=min_val_dtype).amax(dim=dim, keepdim=keepdim)
    
    # fill in valid item index (not all nan) with the min/max and non valid item index nan values
    result = torch.where(in_tensor_valid, result, torch.tensor(float('nan'), dtype=in_tensor.dtype, device=in_tensor.device))
    
    return result


if __name__ == "__main__":
    a = torch.Tensor([[1, 2, 3], [2, 3, float('nan')]]).unsqueeze(0).unsqueeze(0)
    print(a.shape)
    a_min = torch_nanminmax(a, 'min', dim=(1, 2, 3), keepdim=True)
    a_max = torch_nanminmax(a, 'max', dim=(1, 2, 3), keepdim=True)
    print(a_min.shape, a_min)
    print(a_max.shape, a_max)

    b = torch.Tensor([[[1, 2, 3], [2, 3, float('nan')]], [[float('nan'), float('nan'), float('nan')], [float('nan'), float('nan'), float('nan')]]]).unsqueeze(1)
    print(b.shape)
    b_min = torch_nanminmax(b, 'min', dim=(1, 2, 3), keepdim=True)
    b_max = torch_nanminmax(b, 'max', dim=(1, 2, 3), keepdim=True)
    print(b_min.shape, b_min)
    print(b_max.shape, b_max)
    
