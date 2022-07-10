import torch

def up_sample(z_vals, weight, n_importance):
    batch_size, n_samples = z_vals.shape
    weight = weight/(weight.sum(-1)[...,None]+1e-5)
    weight = weight.reshape(batch_size, n_samples)
    prev_weight, next_weight = weight[:, :-1], weight[:, 1:]
    mid_weight = (prev_weight + next_weight) * 0.5
    z_samples = sample_pdf(z_vals, mid_weight, n_importance, det=True).detach()
    return z_samples

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    # 已知一些点的weight，再次采样额外的点
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    #! 先生成u表示随机选择的一些density，再放到coarse的cdf中确定在哪一段？不懂
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, weight, last=False):
    batch_size, n_samples = z_vals.shape
    _, n_importance = new_z_vals.shape
    pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
    z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
    z_vals, index = torch.sort(z_vals, dim=-1)

    if not last:
        new_weight = weight_network.weight(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
        weight = torch.cat([weight, new_weight], dim=-1)
        xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
        index = index.reshape(-1)
        weight = weight[(xx, index)].reshape(batch_size, n_samples + n_importance)

    return z_vals, weight