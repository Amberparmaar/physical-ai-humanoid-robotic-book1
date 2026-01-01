---
sidebar_position: 3
title: Vision Processing and Scene Understanding
---

# Vision Processing and Scene Understanding

In this chapter, we'll explore how Vision-Language-Action (VLA) systems process visual information and understand scenes in order to ground language in visual context and enable appropriate actions. Vision processing is the foundation of spatial reasoning and object interaction in embodied AI systems.

## Understanding Visual Perception in VLA Systems

Vision processing in VLA systems goes beyond simple object recognition to encompass:

1. **Spatial Reasoning**: Understanding spatial relationships between objects
2. **Scene Understanding**: Interpreting the overall context and layout
3. **Object Affordances**: Recognizing what actions are possible with objects
4. **Temporal Coherence**: Understanding motion and change over time
5. **Part-Whole Relationships**: Understanding object composition and structure

### The Role of Vision in VLA Systems

In VLA systems, vision serves multiple critical functions:

- **Perception**: Identifying objects, people, and environmental features
- **Localization**: Determining the robot's position and orientation
- **Mapping**: Building representations of the environment
- **Interaction**: Understanding object affordances and manipulation possibilities
- **Safety**: Detecting obstacles and hazardous conditions

## Advanced Vision Processing Techniques

### Vision Transformers for Scene Understanding

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

class VisionTransformer(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 d_model: int = 768, num_layers: int = 12, num_heads: int = 12,
                 d_ff: int = 3072, dropout: float = 0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize positional embeddings
        self._init_pos_embed()

    def _init_pos_embed(self):
        """Initialize positional embeddings with sine-cosine pattern"""
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches ** 0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, d_model)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='ij')  # shape: 2 x h x w
    grid = torch.stack(grid, dim=0)  # shape: 2 x h x w

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], dim=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # Use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
```

### Segment Anything for Scene Understanding

```python
class SegmentAnythingModel(nn.Module):
    def __init__(self, vision_model: nn.Module, d_model: int = 256):
        super().__init__()
        self.vision_model = vision_model
        self.d_model = d_model

        # Image encoder (ViT)
        self.image_encoder = vision_model

        # Prompt encoder
        self.prompt_encoder = PromptEncoder(
            embed_dim=d_model,
            image_embedding_size=(64, 64),  # ViT patch size
            input_image_size=(1024, 1024),
            mask_in_chans=16
        )

        # Mask decoder
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=d_model,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=d_model,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def forward(self, images, point_prompts=None, box_prompts=None, mask_prompts=None):
        # Encode image
        image_embeddings = self.image_encoder(images)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompts,
            boxes=box_prompts,
            masks=mask_prompts,
        )

        # Decode masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )

        # Upscale masks to original image size
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(224, 224),
            original_size=(images.shape[2], images.shape[3])
        )

        return masks, iou_predictions

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim: int, image_embedding_size: tuple,
                 input_image_size: tuple, mask_in_chans: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # Point encoding
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim),  # For null points
            nn.Embedding(1, embed_dim),  # For foreground points
        ])

        # Box encoding
        self.box_tiwh_embed = nn.Embedding(4, embed_dim)
        self.bbox_corner_embed = nn.Embedding(4, embed_dim)

        # Mask encoding
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            nn.InstanceNorm2d(mask_in_chans // 4),
            nn.ReLU(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            nn.InstanceNorm2d(mask_in_chans),
            nn.ReLU(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def get_dense_pe(self) -> torch.Tensor:
        """Returns the positional encoding used to encode point prompts."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self, points, boxes, masks):
        """Encode prompts"""
        bs = self._get_batch_size(points, boxes, masks)

        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.get_device())

        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

    def _embed_points(self, point_coords, point_labels):
        """Embed point prompts"""
        point_coords = point_coords + 0.5  # Shift to center of pixel
        point_coords = point_coords / torch.tensor(self.input_image_size, device=point_coords.device).unsqueeze(0)
        point_embedding = self.pe_layer.forward_with_coords(point_coords, self.input_image_size)

        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.not_a_point_embed.weight * (point_labels == -1)

        return point_embedding

    def _embed_boxes(self, boxes):
        """Embed box prompts"""
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2) / torch.tensor(self.input_image_size, device=boxes.device).unsqueeze(0)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        return corner_embedding

    def _embed_masks(self, masks):
        """Embed mask prompts"""
        return self.mask_downscaling(masks)

class MaskDecoder(nn.Module):
    def __init__(self, num_multimask_outputs, transformer, transformer_dim, iou_head_depth, iou_head_hidden_dim):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for i in range(self.num_mask_tokens)
        ])

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output):
        """Predict masks given image and prompt embeddings."""
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings
        )

        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings):
        """Predicts masks."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)

        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class TwoWayTransformer(nn.Module):
    def __init__(self, depth: int, embedding_dim: int, num_heads: int, mlp_dim: int, activation: nn.Module = nn.ReLU):
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.blocks = nn.ModuleList()

        for i in range(depth):
            self.blocks.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=1
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
        """Run forward pass."""
        # Precompute image embedding
        if image_embedding.shape[0] != point_embedding.shape[0]:
            bs = point_embedding.shape[0]
            image_embedding = image_embedding.repeat(bs, 1, 1, 1)
            image_pe = image_pe.repeat(bs, 1, 1, 1)

        # BxCxHxW -> BxHWxC
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare tokens
        tokens = point_embedding
        for blk in self.blocks:
            tokens, image_embedding = blk(
                tokens, image_embedding, image_pe
            )

        # Apply final attenion layer
        image_embedding = self.final_attn_token_to_image(
            q=tokens, k=image_embedding, v=image_embedding
        )
        tokens = tokens + image_embedding
        tokens = self.norm_final_attn(tokens)

        return tokens, image_embedding

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: nn.Module = nn.ReLU,
        skip_first_layer_pe: bool = False,
    ):
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, 2, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, tokens, image_embedding, image_pe):
        # Self attention block
        if self.skip_first_layer_pe:
            tokens = self.self_attn(q=tokens, k=tokens, v=tokens)
        else:
            q = k = v = self.norm1(tokens)
            tokens = tokens + self.self_attn(q=q, k=k, v=v)

        # Cross attention block, tokens attending to image embedding
        q = self.norm2(tokens)
        k = v = image_embedding
        tokens = tokens + self.cross_attn_token_to_image(q=q, k=k, v=v)

        # MLP block
        tokens = tokens + self.mlp(self.norm3(tokens))

        # Cross attention block, image embedding attending to tokens
        q = self.norm4(image_embedding)
        k = v = tokens
        image_embedding = image_embedding + self.cross_attn_image_to_token(q=q, k=k, v=v)

        return tokens, image_embedding

class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )
        self.activation = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
```

## Object Detection and Recognition

### YOLO-based Detection for Real-time Applications

```python
class YOLOv8Backbone(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # CSPDarknet backbone
        self.stem = Conv(3, d_model // 8, 3, 2)
        self.stage1 = C2f(d_model // 8, d_model // 4, 1, shortcut=False)
        self.down1 = Conv(d_model // 4, d_model // 2, 3, 2)
        self.stage2 = C2f(d_model // 2, d_model // 2, 2, shortcut=True)
        self.down2 = Conv(d_model // 2, d_model, 3, 2)
        self.stage3 = C2f(d_model, d_model, 2, shortcut=True)
        self.down3 = Conv(d_model, d_model * 2, 3, 2)
        self.stage4 = C2f(d_model * 2, d_model * 2, 2, shortcut=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        stage1_out = x

        x = self.down1(x)
        x = self.stage2(x)
        stage2_out = x

        x = self.down2(x)
        x = self.stage3(x)
        stage3_out = x

        x = self.down3(x)
        x = self.stage4(x)
        stage4_out = x

        return stage1_out, stage2_out, stage3_out, stage4_out

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(Conv(self.c, self.c, 3, 1, g=g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class YOLOv8Neck(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Top-down pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.top_down_stage3 = C2f(d_model * 2, d_model, 1)
        self.top_down_stage2 = C2f(d_model, d_model // 2, 1)

        # Bottom-up pathway
        self.bottom_up_stage3 = C2f(d_model, d_model, 1)
        self.bottom_up_stage4 = C2f(d_model * 2, d_model * 2, 1)

    def forward(self, features):
        stage2, stage3, stage4 = features

        # Top-down pathway
        x = self.upsample(stage4)
        x = torch.cat([x, stage3], dim=1)
        x = self.top_down_stage3(x)

        x_up = self.upsample(x)
        x_up = torch.cat([x_up, stage2], dim=1)
        x_up = self.top_down_stage2(x_up)

        # Bottom-up pathway
        x_down = F.max_pool2d(x_up, 2)
        x_down = torch.cat([x_down, x], dim=1)
        x_down = self.bottom_up_stage3(x_down)

        x_final = F.max_pool2d(x_down, 2)
        x_final = torch.cat([x_final, stage4], dim=1)
        x_final = self.bottom_up_stage4(x_final)

        return x_up, x_down, x_final

class YOLOv8Head(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 256, strides: list = [8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # Classification and regression heads
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                Conv(d_model, d_model, 3, 1),
                Conv(d_model, d_model, 3, 1)
            ) for _ in strides
        ])

        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                Conv(d_model, d_model, 3, 1),
                Conv(d_model, d_model, 3, 1)
            ) for _ in strides
        ])

        # Final prediction layers
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(d_model, num_classes, 1) for _ in strides
        ])

        self.reg_preds = nn.ModuleList([
            nn.Conv2d(d_model, 4, 1) for _ in strides
        ])

        self.obj_preds = nn.ModuleList([
            nn.Conv2d(d_model, 1, 1) for _ in strides
        ])

    def forward(self, features):
        outputs = []

        for i, (feat, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred) in enumerate(
            zip(features, self.cls_convs, self.reg_convs, self.cls_preds, self.reg_preds, self.obj_preds)
        ):
            cls_feat = cls_conv(feat)
            reg_feat = reg_conv(feat)

            cls_output = cls_pred(cls_feat)
            reg_output = reg_pred(reg_feat)
            obj_output = obj_pred(reg_feat)

            # Concatenate outputs
            output = torch.cat([reg_output, obj_output, cls_output], dim=1)
            outputs.append(output)

        return outputs

class YOLOv8Detector(nn.Module):
    def __init__(self, num_classes: int = 80, d_model: int = 256):
        super().__init__()
        self.backbone = YOLOv8Backbone(d_model)
        self.neck = YOLOv8Neck(d_model)
        self.head = YOLOv8Head(num_classes, d_model)

    def forward(self, x):
        # Backbone
        feats = self.backbone(x)

        # Neck
        neck_outs = self.neck(feats[1:])  # Skip stage1 for YOLO neck

        # Head
        outputs = self.head(neck_outs)

        return outputs
```

## 3D Scene Understanding

### Depth Estimation and 3D Reconstruction

```python
class DepthEstimator(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
        )

        # Decoder for depth estimation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 3, 1, 1),
            nn.Sigmoid()  # Normalize depth to [0, 1]
        )

    def forward(self, x):
        features = self.backbone(x)
        depth = self.decoder(features)
        return depth

class MonoDepthEstimator(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Encoder-decoder structure
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
        )

        # Skip connections
        self.skip_conv = nn.Conv2d(3, 64, 3, 2, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        features = self.encoder(x)

        # Decode with skip connection
        depth = self.decoder(features)

        # Upsample to match input size
        depth = F.interpolate(depth, size=x.shape[2:], mode='bilinear', align_corners=False)

        return depth

class PointCloudReconstructor(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Process RGB and depth separately
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
        )

        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
        )

        # Fusion and processing
        self.fusion = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
        )

        # Point cloud generation
        self.point_generator = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1, 1, 0),  # 3D coordinates
        )

    def forward(self, rgb, depth):
        # Encode RGB and depth
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)

        # Concatenate features
        fused_feat = torch.cat([rgb_feat, depth_feat], dim=1)

        # Process fused features
        processed_feat = self.fusion(fused_feat)

        # Generate point cloud coordinates
        point_cloud = self.point_generator(processed_feat)

        return point_cloud

class SceneGraphBuilder(nn.Module):
    def __init__(self, num_classes: int = 80, d_model: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model

        # Object detection and feature extraction
        self.object_detector = YOLOv8Detector(num_classes, d_model)
        self.feature_extractor = VisionTransformer(d_model=d_model)

        # Relationship prediction
        self.rel_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 50)  # 50 common relationship types
        )

        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, d_model // 4),  # bbox coordinates
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
        )

    def forward(self, image):
        # Detect objects
        detections = self.object_detector(image)

        # Extract features for detected regions
        features = self.feature_extractor(image)

        # Build scene graph
        objects = self.parse_detections(detections)
        scene_graph = self.build_graph(objects, features)

        return scene_graph

    def parse_detections(self, detections):
        """Parse detection outputs to extract objects"""
        # This would involve decoding YOLO outputs to get bounding boxes, classes, confidences
        objects = []
        for detection in detections:
            # Decode bounding boxes, classes, and confidences
            # For simplicity, we'll return mock objects
            pass
        return objects

    def build_graph(self, objects, features):
        """Build scene graph from objects and features"""
        nodes = []
        edges = []

        # Create nodes for each object
        for obj in objects:
            node_features = self.extract_node_features(obj, features)
            nodes.append({
                'class': obj['class'],
                'bbox': obj['bbox'],
                'features': node_features
            })

        # Create edges for object relationships
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    rel_features = self.compute_relationship(node1, node2)
                    edges.append({
                        'source': i,
                        'target': j,
                        'relationship': rel_features
                    })

        return {'nodes': nodes, 'edges': edges}

    def extract_node_features(self, obj, features):
        """Extract features for a specific object region"""
        # Extract features from the region of interest
        bbox = obj['bbox']
        # This would involve ROI pooling or similar techniques
        return features.mean(dim=[2, 3])  # Simplified

    def compute_relationship(self, node1, node2):
        """Compute relationship between two objects"""
        # Combine node features and spatial relationship
        node1_features = node1['features']
        node2_features = node2['features']

        # Compute spatial relationship
        bbox1 = node1['bbox']
        bbox2 = node2['bbox']
        spatial_rel = self.compute_spatial_relationship(bbox1, bbox2)

        # Combine features
        combined_features = torch.cat([
            node1_features, node2_features, spatial_rel
        ], dim=-1)

        # Predict relationship type
        relationship = self.rel_predictor(combined_features)

        return relationship

    def compute_spatial_relationship(self, bbox1, bbox2):
        """Compute spatial relationship between two bounding boxes"""
        # Extract spatial features
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        # Relative positions and sizes
        rel_x = (x1 + x2) / 2 - (x3 + x4) / 2
        rel_y = (y1 + y2) / 2 - (y3 + y4) / 2
        rel_w = (x2 - x1) / (x4 - x3 + 1e-6)
        rel_h = (y2 - y1) / (y4 - y3 + 1e-6)

        spatial_features = torch.tensor([rel_x, rel_y, rel_w, rel_h])
        return self.spatial_encoder(spatial_features)
```

## Attention Mechanisms for Scene Understanding

### Spatial Attention for Object Localization

```python
class SpatialAttention(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(d_model, d_model // 8, 1)
        self.conv2 = nn.Conv2d(d_model // 8, 1, 1)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)

        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, d_model: int = 256, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(d_model, d_model // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(d_model // reduction, d_model, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, d_model: int = 256, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(d_model, reduction)
        self.spatial_att = SpatialAttention(d_model)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class VisionTransformerWithAttention(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, d_model: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d_model, patch_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, d_model))

        # Transformer blocks with attention
        self.blocks = nn.ModuleList([
            BlockWithCBAM(d_model, 12) for _ in range(12)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        return self.norm(x)

class BlockWithCBAM(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_model * 4)
        self.norm2 = nn.LayerNorm(d_model)

        # CBAM attention for spatial feature enhancement
        self.cbam = CBAM(d_model)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        # Reshape for CBAM (B, C, H, W)
        x_2d = x.transpose(1, 2).view(B, C, H, W)
        x_2d = self.cbam(x_2d)
        x = x_2d.view(B, C, N).transpose(1, 2)

        # Self-attention
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.mlp(x))

        return x
```

## Scene Understanding Integration for VLA

### Unified Scene Understanding Module

```python
class UnifiedSceneUnderstanding(nn.Module):
    def __init__(self, num_classes: int = 80, d_model: int = 768):
        super().__init__()
        self.d_model = d_model

        # Multi-task backbone
        self.backbone = VisionTransformer(d_model=d_model)

        # Task-specific heads
        self.object_detection_head = YOLOv8Head(num_classes, d_model // 4)
        self.depth_estimation_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(d_model // 2, num_classes, 1, 1, 0)
        )

        # Scene graph builder
        self.scene_graph_builder = SceneGraphBuilder(num_classes, d_model)

        # Cross-task attention
        self.cross_task_attention = MultiHeadSelfAttention(d_model, 8)

        # Output fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, image):
        # Extract features
        features = self.backbone(image)

        # Reshape features for different tasks
        B, N, C = features.shape
        H = W = int(N ** 0.5)
        features_2d = features.transpose(1, 2).view(B, C, H, W)

        # Multi-task predictions
        detection_output = self.object_detection_head([features_2d])
        depth_output = self.depth_estimation_head(features_2d)
        segmentation_output = self.segmentation_head(features_2d)

        # Build scene graph
        scene_graph = self.scene_graph_builder(image)

        # Cross-task attention
        detection_feat = F.adaptive_avg_pool2d(
            detection_output[0].sigmoid(), (H, W)
        ).view(B, -1, H * W).transpose(1, 2)
        depth_feat = F.adaptive_avg_pool2d(
            depth_output, (H, W)
        ).view(B, -1, H * W).transpose(1, 2)
        seg_feat = F.adaptive_avg_pool2d(
            segmentation_output, (H, W)
        ).view(B, -1, H * W).transpose(1, 2)

        # Fuse features from different tasks
        fused_features = torch.cat([
            features,
            detection_feat.mean(dim=1, keepdim=True).expand(-1, N, -1),
            depth_feat.mean(dim=1, keepdim=True).expand(-1, N, -1),
            seg_feat.mean(dim=1, keepdim=True).expand(-1, N, -1)
        ], dim=-1)

        fused_output = self.fusion_layer(fused_features)

        return {
            'detection': detection_output,
            'depth': depth_output,
            'segmentation': segmentation_output,
            'scene_graph': scene_graph,
            'fused_features': fused_output
        }

class VisionLanguageAlignment(nn.Module):
    def __init__(self, vision_dim: int, lang_dim: int, d_model: int = 768):
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, d_model)
        self.lang_proj = nn.Linear(lang_dim, d_model)

        # Cross-modal attention
        self.vision_to_lang = MultiHeadSelfAttention(d_model, 8)
        self.lang_to_vision = MultiHeadSelfAttention(d_model, 8)

        # Alignment prediction head
        self.alignment_head = nn.Linear(d_model * 2, 1)

    def forward(self, vision_features, lang_features):
        # Project features
        vision_proj = self.vision_proj(vision_features)
        lang_proj = self.lang_proj(lang_features)

        # Cross-attention
        vision_aligned = self.vision_to_lang(
            vision_proj, lang_proj, lang_proj
        )
        lang_aligned = self.lang_to_vision(
            lang_proj, vision_proj, vision_proj
        )

        # Predict alignment scores
        combined = torch.cat([vision_aligned, lang_aligned], dim=-1)
        alignment_scores = self.alignment_head(combined)

        return alignment_scores, vision_aligned, lang_aligned

class SceneGroundingModule(nn.Module):
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.scene_understanding = UnifiedSceneUnderstanding(d_model=d_model)
        self.vision_language_aligner = VisionLanguageAlignment(
            vision_dim=d_model, lang_dim=d_model, d_model=d_model
        )

        # Grounding prediction head
        self.grounding_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),  # bounding box coordinates
            nn.Sigmoid()
        )

    def forward(self, image, text_features):
        # Understand the scene
        scene_output = self.scene_understanding(image)
        vision_features = scene_output['fused_features']

        # Align vision and language
        alignment_scores, aligned_vision, aligned_lang = self.vision_language_aligner(
            vision_features, text_features
        )

        # Predict grounded bounding box
        bbox = self.grounding_head(aligned_vision)

        return {
            'alignment_scores': alignment_scores,
            'grounded_bbox': bbox,
            'scene_analysis': scene_output
        }
```

## Quality Assessment for Vision Processing

### Vision Processing Quality Metrics

```python
class VisionQualityAssessor:
    def __init__(self):
        self.metrics = {}

    def evaluate_object_detection(self, predictions, ground_truth, iou_threshold: float = 0.5):
        """Evaluate object detection quality"""
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        for pred in predictions:
            matched = False
            for gt in ground_truth:
                iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold and pred['class'] == gt['class']:
                    tp += 1
                    matched = True
                    break

            if not matched:
                fp += 1

        fn = len(ground_truth) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mAP': self.calculate_map(predictions, ground_truth)
        }

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area

    def calculate_map(self, predictions, ground_truth, iou_thresholds=None):
        """Calculate mean Average Precision"""
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        aps = []
        for iou_thresh in iou_thresholds:
            # Calculate AP for this IoU threshold
            # This is a simplified version - in practice, you'd calculate precision-recall curve
            correct_detections = 0
            total_detections = len(predictions)

            for pred in predictions:
                for gt in ground_truth:
                    if self.calculate_iou(pred['bbox'], gt['bbox']) >= iou_thresh:
                        correct_detections += 1
                        break

            ap = correct_detections / total_detections if total_detections > 0 else 0
            aps.append(ap)

        return sum(aps) / len(aps) if aps else 0

    def evaluate_depth_estimation(self, predicted_depth, ground_truth_depth):
        """Evaluate depth estimation quality"""
        # Calculate various depth metrics
        abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth)
        sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth)
        rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2))
        rmse_log = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2))

        # Threshold accuracy (how many pixels are within x% of ground truth)
        thresh_1 = torch.mean((torch.max(predicted_depth / ground_truth_depth,
                                       ground_truth_depth / predicted_depth) < 1.25).float())
        thresh_2 = torch.mean((torch.max(predicted_depth / ground_truth_depth,
                                       ground_truth_depth / predicted_depth) < 1.25 ** 2).float())
        thresh_3 = torch.mean((torch.max(predicted_depth / ground_truth_depth,
                                       ground_truth_depth / predicted_depth) < 1.25 ** 3).float())

        return {
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'rmse': rmse.item(),
            'rmse_log': rmse_log.item(),
            'delta_1.25': thresh_1.item(),
            'delta_1.25^2': thresh_2.item(),
            'delta_1.25^3': thresh_3.item()
        }

    def evaluate_segmentation(self, predicted_masks, ground_truth_masks):
        """Evaluate segmentation quality"""
        # Calculate IoU for each class
        ious = []
        for pred_mask, gt_mask in zip(predicted_masks, ground_truth_masks):
            intersection = torch.logical_and(pred_mask, gt_mask).sum()
            union = torch.logical_or(pred_mask, gt_mask).sum()

            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection.float() / union.float()

            ious.append(iou)

        mean_iou = sum(ious) / len(ious) if ious else 0.0

        # Pixel accuracy
        total_pixels = ground_truth_masks.numel()
        correct_pixels = (predicted_masks == ground_truth_masks).sum()
        pixel_accuracy = correct_pixels.float() / total_pixels

        return {
            'mean_iou': mean_iou.item(),
            'pixel_accuracy': pixel_accuracy.item(),
            'ious': [iou.item() for iou in ious]
        }

    def assess_scene_understanding(self, scene_graph, ground_truth_graph):
        """Assess scene understanding quality"""
        # Compare scene graphs
        node_precision = self.calculate_graph_node_precision(scene_graph, ground_truth_graph)
        node_recall = self.calculate_graph_node_recall(scene_graph, ground_truth_graph)
        edge_precision = self.calculate_graph_edge_precision(scene_graph, ground_truth_graph)
        edge_recall = self.calculate_graph_edge_recall(scene_graph, ground_truth_graph)

        return {
            'node_precision': node_precision,
            'node_recall': node_recall,
            'edge_precision': edge_precision,
            'edge_recall': edge_recall,
            'graph_similarity': self.calculate_graph_similarity(scene_graph, ground_truth_graph)
        }

    def calculate_graph_node_precision(self, pred_graph, gt_graph):
        """Calculate node precision for scene graph"""
        pred_nodes = set(node['class'] for node in pred_graph['nodes'])
        gt_nodes = set(node['class'] for node in gt_graph['nodes'])

        if not pred_nodes:
            return 1.0 if not gt_nodes else 0.0

        correct = len(pred_nodes.intersection(gt_nodes))
        return correct / len(pred_nodes)

    def calculate_graph_node_recall(self, pred_graph, gt_graph):
        """Calculate node recall for scene graph"""
        pred_nodes = set(node['class'] for node in pred_graph['nodes'])
        gt_nodes = set(node['class'] for node in gt_graph['nodes'])

        if not gt_nodes:
            return 1.0

        correct = len(pred_nodes.intersection(gt_nodes))
        return correct / len(gt_nodes)

    def calculate_graph_similarity(self, graph1, graph2):
        """Calculate overall similarity between scene graphs"""
        # This would involve more sophisticated graph comparison algorithms
        # For now, using a simple overlap measure
        nodes1 = set(node['class'] for node in graph1['nodes'])
        nodes2 = set(node['class'] for node in graph2['nodes'])

        intersection = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))

        return intersection / union if union > 0 else 0.0
```

## Key Takeaways

- Vision processing in VLA systems integrates multiple computer vision tasks
- Vision Transformers provide powerful representations for scene understanding
- Object detection enables grounding of language in visual objects
- Depth estimation and 3D reconstruction provide spatial understanding
- Scene graphs capture relationships between objects
- Attention mechanisms help focus on relevant visual information
- Quality assessment ensures reliable vision processing

## Next Steps

In the next chapter, we'll explore language understanding and grounding, learning how VLA systems process natural language instructions and connect them to visual and action spaces.