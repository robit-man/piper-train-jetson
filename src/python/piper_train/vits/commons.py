import logging
import math
from typing import Optional

import torch
from torch.nn import functional as F

_LOGGER = logging.getLogger("vits.commons")


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += (
        0.5 * (torch.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2.0 * logs_q)
    )
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
    return g


def slice_segments(x, ids_str, segment_size=4):
    b, c, t = x.size()
    ret = torch.zeros(b, c, segment_size, dtype=x.dtype, device=x.device)

    for i in range(b):
        idx_str = max(0, ids_str[i].item())  # Ensure idx_str is a Python int
        idx_end = min(idx_str + segment_size, t)
        actual_segment_size = idx_end - idx_str

        if actual_segment_size > 0:
            ret[i, :, :actual_segment_size] = x[i, :, idx_str:idx_end]
        else:
            _LOGGER.warning(
                f"Empty slice for batch {i}: idx_str={idx_str}, idx_end={idx_end}, "
                f"segment_size={segment_size}, input_tensor_size={t}"
            )

    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    """
    Randomly slices segments from input tensor `x`.
    Args:
        x (torch.Tensor): Input tensor of shape [B, C, T].
        x_lengths (torch.Tensor or None): Actual lengths of each sample in the batch.
        segment_size (int): Size of the segment to slice.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sliced tensor and start indices.
    """
    b, c, t = x.size()

    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
        _LOGGER.info("x_lengths was None. Set to full length.")

    # Clamp and convert `ids_str_max` into valid bounds
    ids_str_max = torch.clamp(x_lengths - segment_size, min=0)
    # Avoid division by zero
    ids_str_max = torch.clamp(ids_str_max, min=1)
    ids_str = torch.floor(torch.rand(b, device=x.device) * ids_str_max).long()

    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length, dtype=torch.float, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (
        num_timescales - 1
    )
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float, device=position.device) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
    signal = F.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.view(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length: int):
    mask = torch.tril(torch.ones(length, length, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).unsqueeze(0).unsqueeze(0)
    return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def sequence_mask(length, max_length: Optional[int] = None):
    if length is None:
        if max_length is None:
            raise ValueError("Either `length` or `max_length` must be provided.")
        _LOGGER.warning("`length` is None. Creating a mask with all positions valid.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask = torch.ones((1, max_length), dtype=torch.bool, device=device)
        return mask
    if max_length is None:
        max_length = length.max().item()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).type_as(mask)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, (0, 0, 1, 0, 0, 0))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    if not parameters:
        _LOGGER.warning("No gradients found for clipping.")
        return 0

    norm_type = float(norm_type)
    clip_value = float(clip_value) if clip_value is not None else None

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)

    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
