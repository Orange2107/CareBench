from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from ..base.base_encoder import (
    DenseNet121CheXEncoder,
    DenseNet121ImageNetEncoder,
    HFCheXpertViTEncoder,
    LSTMEncoder,
    ResNet50Encoder,
    TransformerEncoder,
)


def _to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _normalize_seq_lengths(seq_lengths, device: torch.device, max_len: int) -> torch.Tensor:
    if isinstance(seq_lengths, list):
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=device)
    elif not isinstance(seq_lengths, torch.Tensor):
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long, device=device)
    else:
        seq_lengths = seq_lengths.to(device)
    return torch.clamp(seq_lengths, min=1, max=max_len)


def _sequence_mask(seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    steps = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0)
    return steps < seq_lengths.unsqueeze(1)


class HealNetInputAdapter(nn.Module):
    def get_channel_dim(self) -> int:
        raise NotImplementedError

    def get_num_spatial_axes(self) -> int:
        raise NotImplementedError


class RawEHRAdapter(HealNetInputAdapter):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x

    def get_channel_dim(self) -> int:
        return self.input_dim

    def get_num_spatial_axes(self) -> int:
        return 1


class HealNetTransformerEHRAdapter(TransformerEncoder):
    def forward(self, x, seq_lengths, output_prob: bool = False):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        seq_lengths = _normalize_seq_lengths(seq_lengths, x.device, x.size(1))
        key_padding_mask = ~_sequence_mask(seq_lengths, x.size(1))
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        x = x * (~key_padding_mask).unsqueeze(-1).to(x.dtype)
        return x

    def get_channel_dim(self) -> int:
        return self.get_output_dim()

    def get_num_spatial_axes(self) -> int:
        return 1


class HealNetLSTMEHRAdapter(LSTMEncoder):
    def forward(self, x, seq_lengths, output_prob: bool = False):
        x = self.input_projection(x)
        seq_lengths = _normalize_seq_lengths(seq_lengths, x.device, x.size(1))
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_x)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.size(1)
        )
        out = self.feature_projection(out)
        out = self.dropout(out)
        out = out * _sequence_mask(seq_lengths, x.size(1)).unsqueeze(-1).to(out.dtype)
        return out

    def get_channel_dim(self) -> int:
        return self.get_output_dim()

    def get_num_spatial_axes(self) -> int:
        return 1


class RawCXRAdapter(HealNetInputAdapter):
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return img.permute(0, 2, 3, 1)

    def get_channel_dim(self) -> int:
        return 3

    def get_num_spatial_axes(self) -> int:
        return 2


class HealNetResNetCXRAdapter(ResNet50Encoder):
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.backbone.fc(x)

    def get_channel_dim(self) -> int:
        return self.get_output_dim()

    def get_num_spatial_axes(self) -> int:
        return 1


class HealNetHFCheXpertViTCXRAdapter(HFCheXpertViTEncoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.vit.embeddings(pixel_values=x)
        encoder_output = self.model.vit.encoder(embeddings)
        hidden_state = encoder_output.last_hidden_state if hasattr(encoder_output, "last_hidden_state") else encoder_output[0]
        hidden_state = self.model.vit.layernorm(hidden_state)
        patch_tokens = hidden_state[:, 1:, :]
        return self.proj(patch_tokens)

    def get_channel_dim(self) -> int:
        return self.get_output_dim()

    def get_num_spatial_axes(self) -> int:
        return 1


class HealNetDenseNet121CheXCXRAdapter(DenseNet121CheXEncoder):
    def forward(self, x):
        if x.shape[1] == 3:
            x = (
                0.299 * x[:, 0:1, :, :]
                + 0.587 * x[:, 1:2, :, :]
                + 0.114 * x[:, 2:3, :, :]
            )

        mean_gray = 0.45
        std_gray = 0.225
        x = (x * std_gray + mean_gray)
        x = (x - 0.5) * 2048

        features = self.backbone.features(x)
        x = rearrange(features, "b c h w -> b (h w) c")
        return self.backbone.classifier(x)

    def get_channel_dim(self) -> int:
        return self.get_output_dim()

    def get_num_spatial_axes(self) -> int:
        return 1


class HealNetDenseNet121ImageNetCXRAdapter(DenseNet121ImageNetEncoder):
    def forward(self, x):
        features = self.backbone.features(x)
        features = torch.relu(features)
        x = rearrange(features, "b c h w -> b (h w) c")
        return self.backbone.classifier(x)

    def get_channel_dim(self) -> int:
        return self.get_output_dim()

    def get_num_spatial_axes(self) -> int:
        return 1


def build_ehr_adapter(hparams) -> HealNetInputAdapter:
    input_mode = getattr(hparams, "ehr_input_mode", "raw").lower()
    if input_mode == "raw":
        return RawEHRAdapter(input_dim=hparams.input_dim)
    if input_mode != "encoded":
        raise ValueError(f"Unsupported HealNet EHR input mode: {input_mode}. Supported modes: 'raw', 'encoded'.")

    hidden_size = getattr(
        hparams,
        "hidden_size",
        getattr(hparams, "ehr_context_dim", getattr(hparams, "latent_dim", 256)),
    )
    encoder_type = getattr(hparams, "ehr_encoder_type", "transformer").lower()
    dropout = getattr(hparams, "ehr_dropout", getattr(hparams, "dropout", 0.2))
    num_classes = getattr(hparams, "num_classes", 1)

    if encoder_type == "transformer":
        return HealNetTransformerEHRAdapter(
            input_size=hparams.input_dim,
            num_classes=num_classes,
            d_model=hidden_size,
            n_head=getattr(hparams, "ehr_n_head", 4),
            n_layers=getattr(hparams, "ehr_n_layers", getattr(hparams, "ehr_n_layers_distinct", 1)),
            dropout=dropout,
        )
    if encoder_type == "lstm":
        return HealNetLSTMEHRAdapter(
            input_size=hparams.input_dim,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=getattr(hparams, "ehr_n_layers", 1),
            dropout=dropout,
            bidirectional=_to_bool(getattr(hparams, "ehr_bidirectional", True)),
        )

    raise ValueError(
        f"Unsupported HealNet EHR encoder type: {encoder_type}. "
        "Supported types: 'transformer', 'lstm'."
    )


def build_cxr_adapter(hparams) -> HealNetInputAdapter:
    input_mode = getattr(hparams, "cxr_input_mode", "raw").lower()
    if input_mode == "raw":
        return RawCXRAdapter()
    if input_mode != "encoded":
        raise ValueError(f"Unsupported HealNet CXR input mode: {input_mode}. Supported modes: 'raw', 'encoded'.")

    hidden_size = getattr(
        hparams,
        "hidden_size",
        getattr(hparams, "cxr_context_dim", getattr(hparams, "latent_dim", 256)),
    )
    encoder_type = getattr(hparams, "cxr_encoder", "resnet50").lower()
    pretrained = _to_bool(getattr(hparams, "pretrained", True))

    if encoder_type == "resnet50":
        return HealNetResNetCXRAdapter(hidden_size=hidden_size, pretrained=pretrained)
    if encoder_type == "hf_chexpert_vit":
        return HealNetHFCheXpertViTCXRAdapter(
            hidden_size=hidden_size,
            pretrained=pretrained,
            hf_model_id=getattr(hparams, "hf_model_id", "codewithdark/vit-chest-xray"),
            freeze_vit=getattr(hparams, "freeze_vit", True),
            bias_tune=getattr(hparams, "bias_tune", False),
            partial_layers=getattr(hparams, "partial_layers", 0),
        )
    if encoder_type == "densenet121-res224-chex":
        return HealNetDenseNet121CheXCXRAdapter(hidden_size=hidden_size, pretrained=pretrained)
    if encoder_type == "densenet121-imagenet":
        return HealNetDenseNet121ImageNetCXRAdapter(hidden_size=hidden_size, pretrained=pretrained)

    raise ValueError(
        f"Unsupported HealNet CXR encoder type: {encoder_type}. "
        "Supported types: 'resnet50', 'hf_chexpert_vit', 'densenet121-res224-chex', 'densenet121-imagenet'."
    )
