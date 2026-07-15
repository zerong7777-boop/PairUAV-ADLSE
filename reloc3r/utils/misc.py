# utilitary functions from DUSt3R
import torch


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def slice_kwargs(kwargs, mask):
        def slice_value(value):
            if torch.is_tensor(value) and value.shape[:1] == mask.shape[:1]:
                return value[mask]
            if isinstance(value, list):
                return [slice_value(item) for item in value]
            if isinstance(value, tuple):
                return tuple(slice_value(item) for item in value)
            return value

        result = {}
        for key, value in kwargs.items():
            result[key] = slice_value(value)
        return result

    def wrapper_no(decout, true_shape, **head_kwargs):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W), **head_kwargs)
        return res

    def wrapper_yes(decout, true_shape, **head_kwargs):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W), **head_kwargs)
        if is_portrait.all():
            return transposed(head(decout, (W, H), **head_kwargs))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W), **slice_kwargs(head_kwargs, is_landscape))
        p_result = transposed(head(selout(is_portrait),  (W, H), **slice_kwargs(head_kwargs, is_portrait)))

        # allocate full result
        result = {}
        for k in set(l_result).union(p_result):
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no

def transposed(dic):
    if 'pose' in dic.keys():
        return dic 
    result = {}
    for k, v in dic.items():
        if getattr(v, "ndim", 0) >= 3:
            result[k] = v.swapaxes(1, 2)
        else:
            result[k] = v
    return result


# def invalid_to_nans(arr, valid_mask, ndim=999):
#     if valid_mask is not None:
#         arr = arr.clone()
#         arr[~valid_mask] = float('nan')
#     if arr.ndim > ndim:
#         arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
#     return arr


# def invalid_to_zeros(arr, valid_mask, ndim=999):
#     if valid_mask is not None:
#         arr = arr.clone()
#         arr[~valid_mask] = 0
#         nnz = valid_mask.view(len(valid_mask), -1).sum(1)
#     else:
#         nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
#     if arr.ndim > ndim:
#         arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
#     return arr, nnz
