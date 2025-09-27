import torch
import torch.nn as nn

from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from torch.hub import download_url_to_file, get_dir as get_torch_hub_dir

from ssl4polyp._compat import ensure_torch_container_abcs

ensure_torch_container_abcs()

from timm.models.vision_transformer import VisionTransformer

try:  # timm<0.9 exposes the helper while newer versions removed it
    from timm.models.vision_transformer import _load_weights  # type: ignore
except ImportError:  # pragma: no cover - best effort compatibility shim
    _load_weights = None

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
        super().__init__(
            patch_size=16, embed_dim=embed_dim, depth=depth, num_heads=num_heads
        )

        if ImageNet_weights:
            loc = _download_cached_file(
                "https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
            )
            self.load_pretrained(loc)
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
            if _load_weights is not None:
                _load_weights(self, str(checkpoint_path))
                return

            # Fallback to the upstream implementation which supports
            # loading numpy checkpoints on newer timm releases.
            return super().load_pretrained(str(checkpoint_path))

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
            weights = torch.load(weight_path, map_location="cpu")["model"]
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


def _download_cached_file(url: str) -> str:
    cache_root = Path(get_torch_hub_dir()) / "ssl4polyp"
    cache_root.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name or "download"
    destination = cache_root / filename
    if not destination.exists():
        download_url_to_file(url, destination, progress=True)
    return str(destination)

