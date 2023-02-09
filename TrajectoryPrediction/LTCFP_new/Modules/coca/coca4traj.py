""" Based on https://github.com/lucidrains/CoCa-pytorch/blob/main/coca_pytorch/coca_pytorch.py """

import torch.nn as nn

import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .img_backbone import ImgBackbonePretrained
from .vit import SimpleViT, Extractor

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# to latents


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents):
        super().__init__()
        self.to_latents = nn.Linear(dim, dim_latents, bias=False)

    def forward(self, x):
        latents = self.to_latents(x)
        return F.normalize(latents, dim=-1)

# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4, forTraj = False):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))
        self.forTraj = forTraj

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, attn_mask=None, apply_causal_mask=True):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        # mid = attn_mask[0, -2, -300:] # all True
        # last = attn_mask[0, -1, -300:] # keeps the False values
        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h) # q gets same shape as q in vanilla TF with [4, 8, 512, 64] instead of [4, 8, 7, 64]

        # rotary embeddings

        # positions = self.get_rotary_embedding(n, device)
        # q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask -> for Q*K
        if apply_causal_mask:
            causal_mask = self.get_mask(n, device) # triangular mask
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max) # assign values to tensor where causal_mask == True --> leads to masking out all values with -inf 

        # extra attention mask - for masking out attention from text CLS token to padding

        if exists(attn_mask):
            attn_mask = rearrange(attn_mask, 'b i j -> b 1 i j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)

# cross attention - using multi-query + one-headed key / values as in PaLM w/ optional parallel feedforward

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=8,
        parallel_ff=False,
        ff_mult=4,
        norm_context=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        ) if parallel_ff else None

    def forward(self, x, context):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # pre-layernorm, for queries and context

        x = self.norm(x)
        context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)
        attn = sim.softmax(dim=-1)

        # aggregate

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out

# transformer


class CoCa4Traj(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        unimodal_depth,
        multimodal_depth,
        dim_latents = None,
        num_img_queries=256,
        dim_head=64,
        heads=8,
        ff_mult=4,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        pad_id=0,
        forTraj=False,
        use_contrastive_loss=True,
        normalize_dataset=True,
        img_arch = 'BeIT'
    ):
        # TODO: 
        # - compare with/without caption loss
        # - BOS and other tokens
        # - implement & compare exposure bias mitigation methods
        # - implement & compare other non-autoregressive decoders
        # - try with embedding (unique id per coord + start/end/pad)
        # - cuda OOM issue during inference https://discuss.pytorch.org/t/train-transformer-without-teacher-forcing/132938/2
        # - pretraining TF: https://d2l.ai/chapter_attention-mechanisms-and-transformers/large-pretraining-transformers.html
            # - Encoder via masking (see BERT), Enc-Dec via reconstruction (using noise see BART, or using multitask unification see T5)
            
        super().__init__()
        self.dim = dim

        self.num_obs_steps = 8
        self.ds_mean = 320
        self.ds_std = 75

        self.pad_id = pad_id
        self.normalize_dataset = normalize_dataset 
        self.use_contrastive_loss = use_contrastive_loss
        self.img_arch = img_arch
        
        if self.use_contrastive_loss:
            self.caption_loss_weight = caption_loss_weight
            self.contrastive_loss_weight = contrastive_loss_weight

            num_img_queries += 1

            # contrastive learning temperature

            self.temperature = nn.Parameter(torch.Tensor([1.]))

            # to latents

            dim_latents = default(dim_latents, dim)
            self.img_to_latents = EmbedToLatents(dim, dim_latents)
            self.text_to_latents = EmbedToLatents(dim, dim_latents)

        # token embeddings
        self.forTraj = forTraj

        if self.forTraj:
            num_tokens = 2
            self.token_emb = nn.Linear(2, dim)
            self.label_emb = nn.Linear(2, dim)
        else:
            self.token_emb = nn.Embedding(num_tokens, dim)
        self.text_cls_token = nn.Parameter(torch.randn(dim))        
        
        # img_backbone = SimpleViT(
        #     image_size = 256,
        #     patch_size = 32,
        #     num_classes = 1000,
        #     dim = 1024,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     patch_dropout = 0.5  # https://arxiv.org/abs/2212.00794
        # )
        
        img_backbone = ImgBackbonePretrained(arch=self.img_arch)
        # from transformers import SegformerForSemanticSegmentation
        # img_backbone = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-cityscapes-1024-1024')

        # from transformers import BeitForSemanticSegmentation 
        # img_backbone = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640') # [1, 1601, 768]
        
        # test_model = img_backbone.to('cuda:0')
        # img = torch.randn((1,3,1024,1024), device='cuda:0')
        # out = test_model(img)

        image_dim = img_backbone.hidden_size
        
        self.img_encoder = Extractor(img_backbone, return_embeddings_only=True, detach=False)

        # attention pooling for image tokens

        self.img_queries = nn.Parameter(torch.randn(num_img_queries, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)

        self.img_attn_pool_norm = LayerNorm(dim)
        self.text_cls_norm = LayerNorm(dim)

        # unimodal layers

        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
            )

        self.unimodal_pred_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_pred_layers.append(
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
            )

        # multimodal layers

        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult)),
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
            ]))

        # to logits
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        if not self.forTraj:
            # they used embedding weight tied projection out to logits, not common, but works
            self.to_logits[-1].weight = self.token_emb.weight
            nn.init.normal_(self.token_emb.weight, std=0.02)
        
    
    def embed_text(self, text):
        batch, device = text.shape[0], text.device

        seq = text.shape[1]

        # Pad text for testing...
        if not self.forTraj: text = torch.cat((text[:, :1], torch.zeros((4, 511-1), dtype=int, device='cuda:0')), dim=-1)

        text_tokens = self.token_emb(text) # [4, 511] --> [4, 511, 512]

        if self.use_contrastive_loss:

            # append text cls tokens

            text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
            text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

            # create specific mask for text cls token at the end
            # to prevent it from attending to padding       --> ok, so padding == False so the CLS token is True?

            if self.forTraj:
                # altogether, the cls token might be useless since we dont want to do classification...
                cls_mask = text!=self.pad_id 
                cls_mask = torch.logical_or(cls_mask[:,:,0], cls_mask[:,:,1])
                cls_mask = rearrange(cls_mask, 'b j -> b 1 j') # 15, 1, 8
            else:
                cls_mask = rearrange(text!=self.pad_id, 'b j -> b 1 j')

            attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True) # cls_mask=(4, 1, 511) --> attn_mask=(4, 512, 512)
        else:
            attn_mask = None

        # go through unimodal layers

        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask, apply_causal_mask=False) # 1000, 4, 512

        # get text cls token
        if self.use_contrastive_loss:
            text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
            text_embeds = self.text_cls_norm(text_cls_tokens)
            return text_embeds, text_tokens
        else:
            return None, text_tokens


    def embed_pred(self, labels):
        batch, device = labels.shape[0], labels.device

        label_tokens = self.label_emb(labels)
         # go through unimodal layers

        for attn_ff in self.unimodal_pred_layers:
            label_tokens = attn_ff(label_tokens)

        # get text cls token
        return label_tokens

    def embed_image(self, images=None, image_tokens=None):
        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image tokens

        assert not (exists(images) and exists(image_tokens))

        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            image_tokens = self.img_encoder(images) # BS, 3, 256, 256 --> BS, 8, 8, 1024 (patch_size=32, hidden_size=1024) --> BS, 64, 1024 --> BS, 32, 1024

        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        if self.use_contrastive_loss:
            return img_queries[:, 0], img_queries[:, 1:]

        return None, img_queries # 1, 256, 512

    def forward_original(
        self,
        text,
        images=None,
        image_tokens=None,
        labels=None,
        return_loss=False,
        return_embeddings=False,
        train=True # True implements Teacher Forcing
    ):
        batch, device = text.shape[0], text.device

        if return_loss and not exists(labels):
            text, labels = text[:, :-1], text[:, 1:]

        text_embeds, text_tokens = self.embed_text(text)

        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        # return embeddings if that is what the researcher wants

        if return_embeddings:
            return text_embeds, image_embeds

        # go through multimodal layers

        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)
            text_tokens = cross_attn(text_tokens, image_tokens)

        logits = self.to_logits(text_tokens)

        if not return_loss:
            return logits

        # shorthand

        ce = F.cross_entropy

        # calculate caption loss (cross entropy loss)

        logits = rearrange(logits, 'b n c -> b c n')
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # embedding to latents

        text_latents = self.text_to_latents(text_embeds)
        image_latents = self.img_to_latents(image_embeds)

        # calculate contrastive loss

        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        return caption_loss + contrastive_loss

    
    def forward(
        self,
        batch,
        decoder_mode=0, # # decoder_mode: 0=teacher forcing, 1=running free, 2=scheduled sampling
    ):
        images = batch[0]
        abs_coordinates = batch[1].squeeze(0)

        obs = abs_coordinates[:, :self.num_obs_steps]
        # train = True
        if decoder_mode==0:
            # teacher forcing
            labels = abs_coordinates[:, self.num_obs_steps:]
        else:
            labels = obs[:, -1].unsqueeze(1)
            # obs = abs_coordinates[:, :7] and label = abs_coordinates[:, 8] ?

        batch, device = obs.shape[0], obs.device
        # text de-/encoder
        obs_embeds, obs_tokens = self.embed_text(obs)
        # image encoder
        image_embeds, image_tokens = self.embed_image(images=images)

        # go through multimodal layers
        
        # teacher forcing
        if decoder_mode==0:
            label_tokens = self.embed_pred(labels)
            for attn_ff, cross_attn, cross_pred_attn in self.multimodal_layers:
                obs_tokens = attn_ff(obs_tokens, apply_causal_mask=False)
                obs_tokens = cross_attn(obs_tokens, image_tokens)
                label_tokens = cross_pred_attn(label_tokens, obs_tokens)
            logits = self.to_logits(label_tokens)

        # running free
        else:
            dec_input = labels
            for i in range(abs_coordinates[:, self.num_obs_steps:].size(1)):

                label_tokens = self.embed_pred(dec_input)

                for attn_ff, cross_attn, cross_pred_attn in self.multimodal_layers:
                    obs_tokens = attn_ff(obs_tokens, apply_causal_mask=False)
                    obs_tokens = cross_attn(obs_tokens, image_tokens)
                    label_tokens = cross_pred_attn(label_tokens, obs_tokens)
                
                dec_output = self.to_logits(label_tokens)
                dec_input = torch.cat((dec_input, dec_output[:,-1].unsqueeze(1)), dim=1)

            logits = dec_input[:, 1:]

        if self.use_contrastive_loss:
            return logits, obs_embeds, image_embeds

        return logits, None, None


    def compute_loss(self, prediction, batch, stage):

        labels = batch[1].squeeze(0)[:, self.num_obs_steps:]

        if self.use_contrastive_loss:
            logits, obs_embeds, image_embeds = prediction

            # embedding to latents

            text_latents = self.text_to_latents(obs_embeds)
            image_latents = self.img_to_latents(image_embeds)

            # calculate contrastive loss

            sim = einsum('i d, j d -> i j', text_latents, image_latents)
            sim = sim * self.temperature.exp()
            # https://github.com/lucidrains/DALLE-pytorch/issues/32
            contrastive_labels = torch.arange(batch[1].size(1), device=sim.device)
            a = F.cross_entropy(sim, contrastive_labels)
            b = sim.t()
            contrastive_loss = (F.cross_entropy(sim, contrastive_labels) + F.cross_entropy(sim.t(), contrastive_labels)) * 0.5
            contrastive_loss = contrastive_loss * self.contrastive_loss_weight

            caption_loss = F.mse_loss(logits.squeeze(), labels)
            caption_loss = caption_loss * self.caption_loss_weight

            total_loss = caption_loss + contrastive_loss

            return total_loss, {'MSE_total': total_loss.item(), 'MSE_caption': caption_loss.item()}, None

        else:
            prediction = prediction[0].squeeze()
            caption_loss = F.mse_loss(prediction, labels)
            # TODO implement ADE instead of MSE as metric
            if self.normalize_dataset:
                prediction_true = prediction.clone().detach()
                labels_true = labels.clone().detach()

                prediction_true = prediction_true * self.ds_std + self.ds_mean
                labels_true = labels_true * self.ds_std + self.ds_mean

                metrics = {'MSE_true_caption:': F.mse_loss(prediction_true, labels_true).item()}
            else:
                metrics = None


            return caption_loss, {'MSE_caption': caption_loss.item()}, metrics