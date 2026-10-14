import hashlib
import os

import numpy as np
import torch
import torch.nn as nn

from functools import partial
from pathlib import Path

from ssl4polyp._compat import ensure_torch_container_abcs

ensure_torch_container_abcs()

from timm.models.vision_transformer import VisionTransformer

from .mae import models_mae

from .DPT_decoder import DPT_decoder


class VisionTransformer_from_Any(VisionTransformer):
    
    def _pos_embed(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def __init__(
        self,
        head,
        num_classes,
        frozen,
        dense,
        embed_dim,
        depth,
        num_heads,
        out_token,
        ImageNet_weights=False,
    ):
        # qkv_bias and the LayerNorm epsilon are explicit because their defaults changed
        # between timm releases (0.3.2: False / 1e-5; 0.4.12, used for the v2 runs: True / 1e-6).
        super().__init__(
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if ImageNet_weights:
            weights_path, expected_sha256 = (
                ImageNet_weights
                if isinstance(ImageNet_weights, tuple)
                else (SUP_IMNET_DEFAULT_PATH, SUP_IMNET_SHA256)
            )
            self.load_pretrained(resolve_sup_imnet_weights(weights_path, expected_sha256))
        self.head = nn.Identity()

        self.head_bool = head
        if head:
            self.lin_head = nn.Linear(embed_dim, num_classes)
        self.frozen = frozen
        self.dense = dense

        if dense:
            self.decoder = DPT_decoder(num_classes=num_classes, dense=dense)
        self.out_token = out_token

    def load_pretrained(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)

        if checkpoint_path.suffix.lower() == ".npz":
            with torch.no_grad():
                _load_npz_weights(self, str(checkpoint_path))
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = None

        if isinstance(checkpoint, dict):
            for key in (
                "state_dict",
                "model",
                "model_state",
                "weights",
                "params",
            ):
                candidate = checkpoint.get(key)
                if isinstance(candidate, dict):
                    state_dict = candidate
                    break
            if state_dict is None and all(
                isinstance(k, str) for k in checkpoint.keys()
            ):
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()

        if not isinstance(state_dict, dict):
            print(
                f"Warning: unsupported checkpoint format for {checkpoint_path!s}: {type(checkpoint)}"
            )
            return

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            print(f"Unexpected keys when loading pretrained weights: {unexpected}")

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self._pos_embed(x)

        z = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.dense and i in [2, 5, 8, 11]:
                z.append(x)

        return z if self.dense else self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        if self.dense:
            x = self.decoder(x)
        else:
            if self.out_token == "cls":
                x = x[:, 0]
            elif self.out_token == "spatial":
                x = x[:, 1:].mean(1)
            if self.head_bool:
                x = self.lin_head(x)
        return x


class ViT_from_MAE(models_mae.MaskedAutoencoderViT):
    def __init__(
        self,
        weight_path,
        head,
        num_classes,
        frozen,
        dense,
        embed_dim,
        depth,
        num_heads,
        out_token,
    ):
        super(ViT_from_MAE, self).__init__(
            patch_size=16,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if weight_path is not None:
            weights = torch.load(weight_path, map_location="cpu", weights_only=False)["model"]
            self.load_my_state_dict(weights)
        del self.mask_token
        del self.decoder_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred

        self.head = head
        if head:
            self.lin_head = nn.Linear(embed_dim, num_classes)
        self.frozen = frozen
        self.dense = dense
        if dense:
            self.decoder = DPT_decoder(num_classes=num_classes, dense=dense)
        self.out_token = out_token

    def load_my_state_dict(self, state_dict):
        i = 0
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
            i += 1
        print(f"Successfully loaded params for {i} items")

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        z = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if self.dense and i in [2, 5, 8, 11]:
                z.append(x)

        return z if self.dense else self.norm(x)

    def forward(self, imgs):
        x = self.forward_encoder(imgs)
        if self.dense:
            x = self.decoder(x)
        else:
            if self.out_token == "cls":
                x = x[:, 0]
            elif self.out_token == "spatial":
                x = x[:, 1:].mean(1)
            if self.head:
                x = self.lin_head(x)
        return x


# SUP-imnet: AugReg ViT-B/16 (ImageNet-21k, fine-tuned on ImageNet-1k). Downloaded once from
# SUP_IMNET_URL into the local cache below; runs never download it again.
SUP_IMNET_URL = (
    "https://storage.googleapis.com/vit_models/augreg/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
)
SUP_IMNET_DEFAULT_PATH = (
    "checkpoints/pretrained/vit_b/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
)
SUP_IMNET_SHA256 = "ac104c0df8158c46754510e50495536fa6de9647e572e38b5659dc54f260b124"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_sup_imnet_weights(path, expected_sha256=SUP_IMNET_SHA256) -> str:
    """Return the cached SUP-imnet weights after checking their SHA-256; never downloads."""

    weights = Path(os.path.expanduser(str(path)))
    if not weights.exists():
        raise FileNotFoundError(
            f"SUP-imnet weights not found at '{weights}'. Download them once from {SUP_IMNET_URL} "
            "into that path (see env/README.md)."
        )
    if expected_sha256:
        digest = _sha256(weights)
        if digest != expected_sha256:
            raise RuntimeError(
                f"SUP-imnet weights at '{weights}' have SHA-256 {digest}, expected {expected_sha256}."
            )
    return str(weights)


def _load_npz_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = "") -> None:
    """Load Google Brain Flax ``.npz`` weights into a non-hybrid ViT.

    Ported from ``timm.models.vision_transformer._load_weights`` (timm 0.4.12, Apache-2.0),
    which timm 0.3.2 does not provide. Only the non-hybrid path is kept; the classifier and
    pre-logits weights are skipped because the head is replaced afterwards.
    """

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    if embed_conv_w.shape != model.patch_embed.proj.weight.shape:
        raise ValueError(
            f"Patch embedding shape mismatch: {tuple(embed_conv_w.shape)} vs "
            f"{tuple(model.patch_embed.proj.weight.shape)}"
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(w[f"{prefix}Transformer/posembed_input/pos_embedding"], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        raise ValueError(
            f"Position embedding shape mismatch: {tuple(pos_embed_w.shape)} vs {tuple(model.pos_embed.shape)}"
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat([_n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T for n in ("query", "key", "value")])
        )
        block.attn.qkv.bias.copy_(
            torch.cat([_n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1) for n in ("query", "key", "value")])
        )
        block.attn.proj.weight.copy_(_n2p(w[f"{mha_prefix}out/kernel"]).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(_n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"]))
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(_n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"]))
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))
