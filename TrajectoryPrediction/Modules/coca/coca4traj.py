""" Based on https://github.com/lucidrains/CoCa-pytorch/blob/main/coca_pytorch/coca_pytorch.py """

import torch.nn as nn

import torch
from torch import einsum, nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .img_backbone import ImgBackbonePretrained
from .vit import SimpleViT, Extractor

import time

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
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

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
        # assert False, 'Why was this commented'
        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

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
        config,
        dim,
        sequence_enc_blocks,
        multimodal_blocks,
        dim_latents = None,
        num_img_queries=256,
        dim_head=64,
        heads=8,
        ff_mult=4,
        caption_loss_weight=1.,
        contrastive_loss_weight=1.,
        use_contrastive_loss=True,
    ):
        # TODO: 
        # - compare with/without caption loss
        # - implement & compare exposure bias mitigation methods
        # - cuda OOM issue during inference https://discuss.pytorch.org/t/train-transformer-without-teacher-forcing/132938/2
            
        super().__init__()
        self.dim = dim

        self.normalize_dataset = config['normalize_dataset'] 
        self.img_arch = config['img_arch']
        self.mode = config['mode']
        self.num_obs_steps = config['num_obs_steps']
        self.seq_length = config['pred_length']+self.num_obs_steps

        self.ds_mean = 320
        self.ds_std = 75

        coord_dims = 2

        # token embeddings
        self.pad_id = 0
        self.check_for_fill = torch.isinf

        # TODO try different output heads --> implement them...
        if config['data_format'] in ['partial_tokenized_random', 'partial_tokenized_by_frame']:
            self.tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[TRAJ]', '[UNK]']
            dict_size = len(self.tokens) if self.tokens is not None else 0
            self.to_logits = ToLogitsNet(dim, coord_dims, dict_size)
        elif config['data_format'] == 'full_tokenized_by_frame':
            self.tokens = ['[PAD]', '[BOS]', '[EOS]', '[MASK]', '[TRAJ]', '[UNK]'] + [i for i in range(50)] + [i for i in range(50)]
            dict_size = len(self.tokens) if self.tokens is not None else 0
            self.to_logits = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dict_size, bias=False)
            )
        else:
            self.tokens = None
            self.to_logits = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, coord_dims, bias=False)
            )

        if self.tokens is None:
            assert self.mode != 'MTM', 'for mode MTM tokens must not be None!'

        self.use_contrastive_loss = use_contrastive_loss

        self.ce_factor = 1.
        
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

            self.text_cls_token = nn.Parameter(torch.randn(dim))
            self.text_cls_norm = LayerNorm(dim)

        self.token_emb = TrajectoryEmbedding(coord_dims, self.tokens, dim, data_format = config['data_format'])

            # self.to_logits = nn.Sequential(
            #     LayerNorm(dim),
            #     nn.ModuleList([
            #         nn.Linear(dim, dict_size, bias=False),
            #         nn.Linear(dim, coord_dims, bias=False)
            #     ])
            # )

            # self.to_logits[-1][0].weight = self.token_emb.embed_token.weight

        # they used embedding weight tied projection out to logits, not common, but works
        # self.to_logits[-1].weight = self.token_emb.weight
        # nn.init.normal_(self.token_emb.weight, std=0.02)
        

        img_backbone = ImgBackbonePretrained(arch=self.img_arch)
        image_dim = img_backbone.hidden_size
        
        self.img_encoder = Extractor(img_backbone, return_embeddings_only=True, detach=False)

        # ATTENTION POOLING FOR IMAGE TOKENS
        self.img_queries = nn.Parameter(torch.randn(num_img_queries, dim)) # num image queries for multimodal, but 1 extra CLS for contrastive learning
        self.img_attn_pool = CrossAttention(dim=dim, context_dim=image_dim, dim_head=dim_head, heads=heads, norm_context=True)
        self.img_attn_pool_norm = LayerNorm(dim)


        # choose different configuration, depending on the task
        # unimodal layers
        if self.mode in ['TRAJ_PRED', 'MTM']:
            self.sequence_encoder = nn.ModuleList([])
            for ind in range(sequence_enc_blocks):
                self.sequence_encoder.append(
                    Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                )

        if self.mode in ['TRAJ_PRED']:
            self.target_sequence_encoder = nn.ModuleList([])
            for ind in range(sequence_enc_blocks):
                self.target_sequence_encoder.append(
                    Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                )

        # multimodal layers
        if self.mode in ['TRAJ_PRED', 'MTM']:
            self.multimodal_layers = nn.ModuleList([])
            for ind in range(multimodal_blocks):
                if self.mode == 'TRAJ_PRED':
                    module_list = nn.ModuleList([
                        Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                        Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult)),
                        Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))
                    ])
                elif self.mode == 'MTM':
                    module_list = nn.ModuleList([
                        Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),
                        Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult)),
                    ])
                else:
                    raise NotImplementedError
                self.multimodal_layers.append(module_list)

        # initialize weights
        # TODO init all layers over multiple options
        self.apply(self._initialize_weights)
    

    def _initialize_weights(self, m):
        if hasattr(m, 'weight'):
            try:
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.02)
            except ValueError:
                # Prevent ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
                m.weight.data.uniform_(-0.02, 0.02)
                # print("Bypassing ValueError...")
        elif hasattr(m, 'bias'):
            if m.bias is not None:
                m.bias.data.zero_()

    
    def tokenize(self, token):
        return self.tokens.index(token)

    
    def embed_text(self, obs, tokens):
        # batch, device = obs.shape[0], obs.device

        # seq = obs.shape[1]

        text_tokens = self.token_emb(obs, tokens) # [4, 511] --> [4, 511, 512]

        if self.use_contrastive_loss:

            # append text cls tokens
            batch = obs.shape[0]
            text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
            text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

            # create specific mask for text cls token at the end
            # to prevent it from attending to padding       --> ok, so padding == False so the CLS token is True?

            # altogether, the cls token might be useless since we dont want to do classification...
            raise NotImplementedError('whats this')
            cls_mask = obs!=self.pad_id 
            cls_mask = torch.logical_or(cls_mask[:,:,0], cls_mask[:,:,1])
            cls_mask = rearrange(cls_mask, 'b j -> b 1 j') # 15, 1, 8

            attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True) # cls_mask=(4, 1, 511) --> attn_mask=(4, 512, 512)
        else:
            attn_mask = None

        # go through unimodal layers

        for attn_ff in self.sequence_encoder:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask, apply_causal_mask=False) # 1000, 4, 512

        # get text cls token
        if self.use_contrastive_loss:
            text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
            text_embeds = self.text_cls_norm(text_cls_tokens)
            return text_embeds, text_tokens
        else:
            return None, text_tokens


    def embed_pred(self, labels, tokens):
        # batch, device = labels.shape[0], labels.device

        label_tokens = self.token_emb(labels, tokens)
         # go through unimodal layers

        for attn_ff in self.target_sequence_encoder:
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


    def set_mapping(self, abs_coordinates, tokens, decoder_mode, *args):
        if self.mode == 'TRAJ_PRED': return self.set_mapping_traj_pred(abs_coordinates, tokens, decoder_mode)
        elif self.mode == 'MTM': return self.set_mapping_mtm(abs_coordinates, tokens, decoder_mode, *args)
        else: raise NotImplementedError


    def set_mapping_traj_pred(self, abs_coordinates, tokens, decoder_mode):
        obs_coords = abs_coordinates[:, :self.num_obs_steps] if abs_coordinates is not None else None
        obs_tokens = tokens[:, :self.num_obs_steps] if tokens is not None else None
        # train = True
        if decoder_mode==0:
            # teacher forcing
            label_coords = abs_coordinates[:, self.num_obs_steps:] if abs_coordinates is not None else None
            label_tokens = tokens[:, self.num_obs_steps:] if tokens is not None else None
        else:
            label_coords = obs_coords[:, -1].unsqueeze(1) if abs_coordinates is not None else None
            label_tokens = obs_tokens[:, -1].unsqueeze(1) if tokens is not None else None
            # obs = abs_coordinates[:, :7] and label = abs_coordinates[:, 8] ?
        return obs_coords, obs_tokens, label_coords, label_tokens


    def set_mapping_mtm(self, abs_coordinates, tokens, decoder_mode, *args):
        mtm_mask = args[0]
        masked_tokens = tokens.masked_fill(mtm_mask, self.tokenize('[MASK]'))
        masked_x = abs_coordinates[:,:,0].masked_fill(mtm_mask, torch.inf).unsqueeze(-1)
        masked_y = abs_coordinates[:,:,1].masked_fill(mtm_mask, torch.inf).unsqueeze(-1)
        masked_coordinates = torch.cat((masked_x, masked_y), dim=-1)
        assert torch.equal(torch.argwhere(mtm_mask), torch.argwhere(masked_tokens==self.tokenize('[MASK]')))
        return masked_coordinates, masked_tokens, abs_coordinates, tokens # TODO only token reconstruction or also traj reconstruction

    
    def cross_attention(self, obs_tokens, image_tokens, label_coords, label_tokens, decoder_mode):
        if self.mode == 'TRAJ_PRED': return self.cross_attention_traj_pred(obs_tokens, image_tokens, label_coords, label_tokens, decoder_mode)
        elif self.mode == 'MTM': return self.cross_attention_mtm(obs_tokens, image_tokens, label_coords, label_tokens, decoder_mode)
        else: raise NotImplementedError


    def cross_attention_traj_pred(self, obs_tokens, image_tokens, label_coords, label_tokens, decoder_mode):
        # teacher forcing
        if decoder_mode==0:

            label_tokens = self.embed_pred(label_coords, label_tokens)

            for attn_ff, cross_attn, cross_pred_attn in self.multimodal_layers:
                obs_tokens = attn_ff(obs_tokens, apply_causal_mask=False)
                obs_tokens = cross_attn(obs_tokens, image_tokens)
                label_tokens = cross_pred_attn(label_tokens, obs_tokens)

            logits = self.to_logits(label_tokens)

        # running free
        else:
            dec_input = label_coords
            dec_tokens = label_tokens
            token_logits = [] if dec_tokens is not None and label_coords is not None else None
            # TODO what is the first token now for running free mode?
            # TODO check if running free mode implemented correctly for tokenized data_loader
            for i in range(self.seq_length - self.num_obs_steps):

                label_tokens = self.embed_pred(dec_input, dec_tokens)

                for attn_ff, cross_attn, cross_pred_attn in self.multimodal_layers:
                    obs_tokens = attn_ff(obs_tokens, apply_causal_mask=False)
                    obs_tokens = cross_attn(obs_tokens, image_tokens)
                    label_tokens = cross_pred_attn(label_tokens, obs_tokens)
                
                dec_output = self.to_logits(label_tokens)

                # if tokenized, then apply running free mode for tokens as well
                if label_coords is not None:
                    if dec_tokens is not None: 
                        if len(token_logits)==0: 
                            token_logits = dec_output[:,-1,2:].unsqueeze(1)
                        else:
                            token_logits = torch.cat((token_logits, dec_output[:,-1,2:].unsqueeze(1)), dim=1)
                        dec_output_tokens = torch.argmax(dec_output[:,:,2:], dim=-1) # argmax is predicted class by cross-entropy
                        dec_tokens = torch.cat((dec_tokens, dec_output_tokens[:,-1].unsqueeze(1)), dim=1)
                        dec_output = dec_output[:,:,:2]
                    dec_input = torch.cat((dec_input, dec_output[:,-1].unsqueeze(1)), dim=1)
                else:
                    if dec_tokens is not None:
                        dec_output_tokens = torch.argmax(dec_output, dim=-1) # argmax is predicted class by cross-entropy
                        dec_tokens = torch.cat((dec_tokens, dec_output_tokens[:,-1].unsqueeze(1)), dim=1)
            logits = dec_input[:, 1:] if dec_input is not None else dec_tokens[:, 1:]
            if token_logits is not None: logits = torch.cat((logits, token_logits), dim=-1)
        
        return logits


    def cross_attention_mtm(self, obs_tokens, image_tokens, label_coords, label_tokens, decoder_mode):

        for attn_ff, cross_attn in self.multimodal_layers:
            obs_tokens = attn_ff(obs_tokens, apply_causal_mask=False)
            obs_tokens = cross_attn(obs_tokens, image_tokens)
        logits = self.to_logits(obs_tokens)

        return logits


    def forward(
        self,
        batch,
        decoder_mode=0,
    ):
        images = batch['images']
        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        tokens = batch['tokens'] if 'tokens' in batch else None
        mtm_mask = batch['mtm_mask'].squeeze(0) if 'mtm_mask' in batch else None

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.squeeze(0)
        elif isinstance(tokens, tuple):
            tokens = tokens[0].squeeze(0)

        obs_coords, obs_tokens, label_coords, label_tokens = self.set_mapping(abs_coordinates, tokens, decoder_mode, mtm_mask)

        # batch, device = obs_coords.shape[0], obs_coords.device
        # text de-/encoder
        obs_embeds, obs_tokens = self.embed_text(obs_coords, obs_tokens)
        # image encoder
        image_embeds, image_tokens = self.embed_image(images=images)

        # go through multimodal layers
        logits = self.cross_attention(obs_tokens, image_tokens, label_coords, label_tokens, decoder_mode)

        if self.use_contrastive_loss:
            return logits, obs_embeds, image_embeds

        return logits, None, None


    def compute_loss_with_tokens(self, label_coords, label_tokens, prediction_coords, prediction_tokens):
        # TODO turn off assertions after one batch
        is_traj_mask = label_tokens.eq(self.tokenize('[TRAJ]')).unsqueeze(-1).repeat(1, 1, 2)
        assert torch.equal(self.check_for_fill(label_coords)[:,:,0], self.check_for_fill(label_coords)[:,:,1]), 'obs masks on x- and y-direction need to be equal!'
        assert torch.equal(~self.check_for_fill(label_coords)[:,:,0], label_tokens.eq(self.tokenize('[TRAJ]'))), '4-mask in tokens needs to be equal to non-inf mask in traj!'

        label_coords = label_coords.masked_fill(~is_traj_mask, 0)
        caption_loss_traj = torch.sum(((prediction_coords-label_coords)*is_traj_mask)**2.0)  / torch.sum(is_traj_mask)
        caption_loss_tokens = F.cross_entropy(prediction_tokens.permute(0, 2, 1), label_tokens.long())

        return caption_loss_traj + self.ce_factor + caption_loss_tokens, is_traj_mask
    

    # def compute_loss_without_coords(self, label_tokens, prediction_tokens):
    #     is_traj_mask = torch.ones_like(prediction_tokens, device=prediction_tokens.device, dtype=bool)
    #     caption_loss_traj = F.cross_entropy(prediction_tokens.permute(0, 2, 1), label_tokens.long())
    #     return caption_loss_traj, is_traj_mask


    def compute_loss_without_tokens(self, label_coords, prediction_coords):
        is_traj_mask = torch.ones_like(prediction_coords, device=prediction_coords.device, dtype=bool)
        caption_loss_traj = F.mse_loss(prediction_coords, label_coords)
        return caption_loss_traj, is_traj_mask


    def compute_loss(self, prediction, batch, stage):

        abs_coordinates = batch['coords'].squeeze(0) if 'coords' in batch else None
        tokens = batch['tokens'].squeeze(0) if 'tokens' in batch else None
        mtm_mask = batch['mtm_mask'].squeeze(0) if 'mtm_mask' in batch else None
    
        _, _, label_coords, label_tokens = self.set_mapping(abs_coordinates, tokens, 0, mtm_mask)

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

            caption_loss = F.mse_loss(logits.squeeze(), label_coords)
            caption_loss = caption_loss * self.caption_loss_weight

            total_loss = caption_loss + contrastive_loss

            return total_loss, {'MSE_total': total_loss.item(), 'MSE_caption': caption_loss.item()}, None

        else:
            if label_tokens is not None and label_coords is not None:
                prediction_coords = prediction[0][:,:,:2]
                prediction_tokens = prediction[0][:,:,2:]
                caption_loss, is_traj_mask = self.compute_loss_with_tokens(label_coords, label_tokens, prediction_coords, prediction_tokens)  
            # elif label_tokens is not None and label_coords is None:
            #     caption_loss, is_traj_mask = self.compute_loss_without_coords(label_tokens, prediction_tokens)
            else:
                prediction_coords = prediction[0].squeeze()
                caption_loss, is_traj_mask = self.compute_loss_without_tokens(label_coords, prediction_coords)

            prediction_m = prediction_coords.clone().detach()
            labels_m = label_coords.clone().detach()            
            if self.normalize_dataset:
                prediction_m = prediction_m * self.ds_std + self.ds_mean
                labels_m = labels_m * self.ds_std + self.ds_mean

            with torch.no_grad():
                ade_loss = torch.sum((torch.abs(prediction_m - labels_m).masked_fill(~is_traj_mask, 0)))  / torch.sum(is_traj_mask)
                metrics = {'ADE_true_caption:': ade_loss.item()}
                
                if label_tokens is not None:
                    prediction_tokens = torch.argmax(prediction_tokens, dim=-1)
                    token_accuracy = (label_tokens == prediction_tokens).sum() / torch.numel(label_tokens)
                    metrics['Token_accuracy'] = token_accuracy.item()

            return caption_loss, {'MSE_caption': caption_loss.item()}, metrics
    

class ToLogitsNet(nn.Module):
    def __init__(self, dim, coord_dims, dict_size) -> None:
        super().__init__()
        self.dim = dim
        self.coord_dims = coord_dims
        self.dict_size = dict_size

        self.to_token_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, dict_size, bias=False)
        )
        self.to_coord_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, coord_dims, bias=False)
        )
    
    def forward(self, x):
        token_logits = self.to_token_logits(x)
        coord_logits = self.to_coord_logits(x)
        return torch.cat((coord_logits, token_logits), dim=-1)


class TrajectoryEmbedding(nn.Module):
    def __init__(self, coord_dims, tokens, dim, data_format='') -> None:
        super().__init__()
        self.coord_dims = coord_dims
        self.tokens = tokens
        self.dim = dim
        self.data_format = data_format
        self.check_for_fill = torch.isinf

        self.embed_coord = nn.Linear(coord_dims, dim)
        if self.tokens is not None:           
            self.embed_token = nn.Embedding(len(tokens), dim)
        
        if self.data_format == 'full_tokenized_by_frame':
            self.coord_network = nn.Sequential(nn.Linear(10, 100), nn.Linear(100, 1))
            # every x- and y-digit with the same weight/influence as [BOS], [EOS], ... does not make sense --> full/partial hybrid?: same but no regression problem, but actually classification

    def tokenize(self, token):
        return self.tokens.index(token)


    def forward(self, obs, all_tokens):

        if self.data_format == 'full_tokenized_by_frame':
            # get all traj tokens
            is_traj_mask = torch.argwhere(all_tokens[:,:,1] != -1)
            traj_tokens = all_tokens[is_traj_mask[:,0], is_traj_mask[:,1], :]
            assert torch.all(traj_tokens != -1)
            # get all non traj tokens
            is_not_traj_mask = torch.argwhere(all_tokens[:,:,1] == -1)
            non_traj_tokens = all_tokens[is_not_traj_mask[:,0], is_not_traj_mask[:,1], 0]
            assert torch.all(all_tokens[is_not_traj_mask[:,0], is_not_traj_mask[:,1], 1:] == -1)

            traj_tokens = self.embed_token(traj_tokens)
            non_traj_tokens = self.embed_token(non_traj_tokens)

            traj_tokens = self.coord_network(traj_tokens.permute(0,2,1)).squeeze(2)

            # is_traj_mask_np = is_traj_mask.detach().cpu().numpy()
            # is_not_traj_mask_np = is_not_traj_mask.detach().cpu().numpy()
            tokens_tensor = torch.full((all_tokens.shape[0], all_tokens.shape[1], self.dim), fill_value=float('-inf'), dtype=torch.float32, device=all_tokens.device)

            # Fill the tensor using indices from indices1 and indices2
            tokens_tensor[is_traj_mask[:, 0], is_traj_mask[:, 1], :] = traj_tokens
            tokens_tensor[is_not_traj_mask[:, 0], is_not_traj_mask[:, 1], :] = non_traj_tokens

            assert torch.all(torch.isfinite(tokens_tensor))
            return tokens_tensor

        is_traj_mask = self.check_for_fill(obs)[:,:,0]
        
        # TODO turn off assertions after one batch
        if all_tokens is not None:
            assert torch.equal(self.check_for_fill(obs)[:,:,0], self.check_for_fill(obs)[:,:,1]), 'obs masks on x- and y-direction need to be equal!'
            # assert torch.equal(is_traj_mask.unsqueeze(-1).repeat(1, 1, self.dim)[:,:,0], all_tokens.ne(4)), 'non-4-mask in tokens needs to be equal to inf mask in traj!'
        else:
            assert torch.count_nonzero(self.check_for_fill(obs)) == 0, 'If no tokens supplied, obs cannot contain any inf values!'
        
        obs = obs.masked_fill(is_traj_mask.unsqueeze(-1).repeat(1, 1, obs.size(-1)), 0)
        coord_emb = self.embed_coord(obs)
        # check inf coords before and after pass
        assert torch.all(torch.isinf(coord_emb)[:,:,0] == torch.isinf(obs)[:,:,0]) and torch.all(torch.isinf(coord_emb)[:,:,443] == torch.isinf(obs)[:,:,1])

        if self.tokens is not None:
            token_emb = self.embed_token(all_tokens)

            # only mask out non-trajectories, but not TRAJ token (since it needs to be learned)
            coord_emb = coord_emb.masked_fill(is_traj_mask.unsqueeze(-1).repeat(1, 1, self.dim), 0)
            # token_emb = token_emb.masked_fill(~is_traj_mask, 0)

            output = coord_emb + token_emb

            # diff = time.process_time() - start
            # print(f'Forward pass time with inf: {diff} s with {torch.count_nonzero(torch.isinf(obs)[:,:,0])} Falses')
        else:
            output = coord_emb
        
        return output
