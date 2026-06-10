# Code Changes — RF-DETR Architectural Enhancements

This document shows the exact old code (before modification) and new code (after modification)
for each of the three enhancements, with file locations.

---

## Enhancement 1 — P2 Feature Level Extension

### File: `rf-detr/rfdetr/models/backbone/backbone.py` (line ~95)

**OLD:**
```python
        level2scalefactor = dict(P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]
```

**NEW:**
```python
        # Added P2 (4.0x scale) for higher resolution features to detect fine details like thin cracks
        level2scalefactor = dict(P2=4.0, P3=2.0, P4=1.0, P5=0.5, P6=0.25)
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]
```

---

### File: `rf-detr/rfdetr/main.py` (line ~971)

**OLD:**
```python
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5'))
```

**NEW:**
```python
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P2', 'P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--enc_layers', default=0, type=int,
                        help="Number of encoder layers with MSDeformAttn (0 to disable)")
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help="Number of feature levels for multi-scale processing")
```

---

## Enhancement 2 — Multi-Scale Intermediate Backbone Features

### File: `rf-detr/rfdetr/models/backbone/projector.py` (line ~175)

**OLD:**  
The `MultiScaleProjector` only handled `scale == 2.0`, `1.0`, `0.5`, and `0.25`.
There was no branch for `scale == 4.0`:

```python
        for scale in scale_factors:
            if scale == 0.25:
                self.use_extra_pool = True
                continue

            stages_sampling.append([])
            for in_dim in in_channels:
                layers = []

                if scale == 2.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                    ])
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend([
                        ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                    ])
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
```

**NEW:**  
Added the `scale == 4.0` branch with two stacked transposed convolutions (total 4× upsample):

```python
        for scale in scale_factors:
            if scale == 0.25:
                self.use_extra_pool = True
                continue

            stages_sampling.append([])
            for in_dim in in_channels:
                layers = []

                if scale == 4.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        get_norm('LN', in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ])
                elif scale == 2.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                    ])
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend([
                        ConvX(in_dim, in_dim, 3, 2, layer_norm=layer_norm),
                    ])
                else:
                    raise NotImplementedError("Unsupported scale_factor:{}".format(scale))
                layers = nn.Sequential(*layers)
                stages_sampling[-1].append(layers)
```

---

### File: `rf-detr/rfdetr/main.py` (line ~945)

**OLD:**  
`out_feature_indexes` defaulted to only the final DINOv2 stage:

```python
    parser.add_argument('--out_feature_indexes', default=[12], type=int, nargs='+', help='only for vit now')
```

**NEW:**  
Four intermediate stages are requested so all depth levels are fused:

```python
    parser.add_argument('--out_feature_indexes', default=[3, 6, 9, 12], type=int, nargs='+', help='only for vit now')
```

---

## Enhancement 3 — Multi-Scale Deformable Attention in Transformer Encoder

### File: `rf-detr/rfdetr/models/transformer.py`

**OLD:**  
The file had no `TransformerEncoderLayer` or `TransformerEncoder` classes.
The `Transformer` class had no encoder attribute and its `__init__` signature had no
`num_encoder_layers` parameter:

```python
class Transformer(nn.Module):

    def __init__(self, d_model=512, sa_nhead=8, ca_nhead=8, num_queries=300,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, group_detr=1,
                 two_stage=False,
                 num_feature_levels=4, dec_n_points=4,
                 lite_refpoint_refine=False,
                 decoder_norm_type='LN',
                 bbox_reparam=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(...)
        ...
        self.decoder = TransformerDecoder(...)
        # No encoder here
```

The `forward` method went directly from flattening features to the decoder:

```python
    def forward(self, srcs, masks, pos_embeds, refpoint_embed, query_feat):
        ...
        memory = torch.cat(src_flatten, 1)
        # directly passed to decoder — no encoder step
        ...
```

`build_transformer` had no `enc_layers` argument:

```python
def build_transformer(args):
    try:
        two_stage = args.two_stage
    except:
        two_stage = False

    return Transformer(
        d_model=args.hidden_dim,
        ...
        # no num_encoder_layers argument
    )
```

---

**NEW:**  
Two new classes added before `Transformer` (lines 129–228):

```python
class TransformerEncoderLayer(nn.Module): 
    """Encoder layer with Multi-Scale Deformable Attention for better feature aggregation"""
    def __init__(self, d_model=256, nhead=8, dim_feedforward=1024,
                 dropout=0.1, activation="relu", normalize_before=False,
                 num_feature_levels=4, enc_n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, num_feature_levels, nhead, enc_n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
# take featuer -> apply mulit sclae -> refine -> stablized 
    def forward_post(self, src, pos, reference_points, spatial_shapes,
                     level_start_index, padding_mask=None):
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points, src, spatial_shapes,
            level_start_index, padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes,
                level_start_index, padding_mask=None):
        return self.forward_post(src, pos, reference_points, spatial_shapes,
                                 level_start_index, padding_mask)


class TransformerEncoder(nn.Module): # run multiple encoder layers in sequence to refine features deeply 
    """Encoder with multiple layers using MSDeformAttn"""
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): # tell attenction where to look in each featuer map 
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
# run multi-layer encoder with multi-scale attention 
    def forward(self, src, spatial_shapes, level_start_index, valid_ratios,
                pos=None, padding_mask=None):
        output = src
        if valid_ratios is None:
            valid_ratios = torch.ones(src.shape[0], spatial_shapes.shape[0], 2,
                                     device=src.device, dtype=src.dtype)
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes,
                           level_start_index, padding_mask)
        return output
```

`Transformer.__init__` updated to conditionally build the encoder (lines ~234–244):

```python
    def __init__(self, ..., num_encoder_layers=2):
        super().__init__()

        # Add encoder with MSDeformAttn for better multi-scale feature aggregation
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(
                d_model, ca_nhead, dim_feedforward,
                dropout, activation, normalize_before,
                num_feature_levels=num_feature_levels,
                enc_n_points=4)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        else:
            self.encoder = None
```

`Transformer.forward` applies the encoder before the decoder (lines ~330–344):

```python
        memory = torch.cat(src_flatten, 1)
        ...
        # Apply encoder with MSDeformAttn if available
        if self.encoder is not None:
            memory = self.encoder(
                memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios.to(memory.dtype) if valid_ratios is not None else None,
                pos=lvl_pos_embed_flatten,
                padding_mask=mask_flatten
            )
        # decoder follows
```

`build_transformer` updated to read `enc_layers` from args (lines ~678–683):

```python
def build_transformer(args):
    try:
        two_stage = args.two_stage
    except:
        two_stage = False

    try:
        num_encoder_layers = args.enc_layers
    except:
        num_encoder_layers = 2  # Default to 2 encoder layers for improved feature aggregation

    return Transformer(
        ...
        num_encoder_layers=num_encoder_layers,
    )
```
