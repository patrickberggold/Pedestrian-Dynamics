""" Based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit_with_patch_dropout.py """

import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, patch_dropout = 0.5):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img) # BS, 3, 256, 256 --> BS, 8, 8, 1024, means patch_size=32, hidden_dim=1024
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe # BS, 8, 8, 1024 --> BS, 64, 1024

        x = self.patch_dropout(x) # BS, 64, 1024 --> BS, 32, 1024

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

#### Extractor

def exists(val):
    return val is not None

def identity(t):
    return t

def clone_and_detach(t):
    return t.clone().detach()

def apply_tuple_or_single(fn, val):
    if isinstance(val, tuple):
        return tuple(map(fn, val))
    return fn(val)

class Extractor(nn.Module):
    def __init__(
        self,
        vit,
        device = None,
        layer = None,
        layer_name = 'transformer',
        layer_save_input = False,
        return_embeddings_only = False,
        detach = True
    ):
        super().__init__()
        self.vit = vit
        if not isinstance(vit, SimpleViT):
            layer_name = 'backbone'

        self.data = None
        self.latents = None
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

        self.layer = layer
        self.layer_name = layer_name
        self.layer_save_input = layer_save_input # whether to save input or output of layer
        self.return_embeddings_only = return_embeddings_only

        self.detach_fn = clone_and_detach if detach else identity

    def _hook(self, _, inputs, output):
        layer_output = inputs if self.layer_save_input else output
        self.latents = apply_tuple_or_single(self.detach_fn, layer_output)

    def _register_hook(self):
        if not exists(self.layer):
            assert hasattr(self.vit, self.layer_name), 'layer whose output to take as embedding not found in vision transformer'
            layer = getattr(self.vit, self.layer_name)
        else:
            layer = self.layer

        handle = layer.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        del self.latents
        self.latents = None

    def forward(
        self,
        img,
        return_embeddings_only = False
    ):
        assert not self.ejected, 'extractor has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)
        if not isinstance(self.vit, SimpleViT):
            return pred

        target_device = self.device if exists(self.device) else img.device
        latents = apply_tuple_or_single(lambda t: t.to(target_device), self.latents)

        if return_embeddings_only or self.return_embeddings_only:
            return latents

        return pred, latents