import math
import os

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)

import cached_conv as cc
import gin
import torch.nn as nn
from absl import app, flags
from effortless_config import Config

import rave

flags.DEFINE_string(
    "run", default=None, required=True, help="Run or torchscript (.ts file) to export"
)
flags.DEFINE_boolean(
    "split",
    default=False,
    help="Split model export into separate encoder and decoder files.",
)
flags.DEFINE_float(
    "fidelity",
    default=.95,
    lower_bound=.1,
    upper_bound=.999,
    help='Fidelity to use during inference (Variational mode only)',
)
flags.DEFINE_integer(
    "latent_dims",
    default=None,
    help="Explicit latent size (power of 2). Overrides --fidelity. "
         "Must not exceed the full latent size.",
)
FLAGS = flags.FLAGS


class MockEncoder(nn.Module):
    def __init__(self, pretrained: rave.RAVE, latent_size: int):
        super().__init__()
        self.pretrained = pretrained
        self.latent_size = latent_size

    def forward(self, x):
        x_enc = x
        if self.pretrained.input_mode == "pqmf":
            x_enc = rave.model._pqmf_encode(self.pretrained.pqmf, x_enc)
        elif self.pretrained.input_mode == "mel":
            x_enc = self.pretrained._mel_encode(x)
        z = self.pretrained.encoder.encoder(x_enc)
        z = self.pretrained.encoder.reparametrize(z)[0]
        z = z - self.pretrained.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.pretrained.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z


class MockDecoder(nn.Module):
    def __init__(self, pretrained: rave.RAVE, latent_size: int):
        super().__init__()
        self.pretrained = pretrained
        self.latent_size = latent_size
        self.full_latent_size = pretrained.latent_size

    def forward(self, x):
        noise = torch.randn(
            x.shape[0],
            self.full_latent_size - x.shape[1],
            x.shape[-1],
        ).type_as(x)
        z = torch.cat([x, noise], 1)
        z = F.conv1d(z, self.pretrained.latent_pca.T.unsqueeze(-1))
        z = z + self.pretrained.latent_mean.unsqueeze(-1)
        return self.pretrained.decode(z)


class MockTSModule(nn.Module):
    def __init__(self, pretrained: torch.jit._script.RecursiveScriptModule):
        super().__init__()
        self.pretrained = pretrained

    def forward(self, x):
        return self.pretrained.forward(x)


class MockTSEncoder(nn.Module):
    def __init__(self, pretrained: torch.jit._script.RecursiveScriptModule):
        super().__init__()
        self.pretrained = pretrained

    def forward(self, x):
        return self.pretrained.encode(x)


class MockTSDecoder(nn.Module):
    def __init__(self, pretrained: torch.jit._script.RecursiveScriptModule):
        super().__init__()
        self.pretrained = pretrained

    def forward(self, x):
        return self.pretrained.decode(x)


def main(argv):
    source = FLAGS.run
    if os.path.isfile(source) and os.path.splitext(source)[1] == ".ts":
        export_from_torchscript()
    else:
        export_from_run()


def export_from_torchscript():
    pretrained = torch.jit.load(FLAGS.run)
    x = torch.randn(1, pretrained.n_channels, 2**15)
    name = os.path.basename(os.path.normpath(FLAGS.run))
    export_path = FLAGS.run[:-3]

    if FLAGS.split:
        encoder, decoder = MockTSEncoder(pretrained), MockTSDecoder(pretrained)
        z = encoder(x)
        torch.onnx.export(
            encoder,
            x,
            f"{export_path}_encoder_ts.onnx",
            export_params=True,
            opset_version=12,
            input_names=["audio_in"],
            output_names=["latent_out"],
            dynamic_axes={
                "audio_in": {2: "audio_length"},
                "latent_out": {0: "batch", 2: "latent_time"},
            },
            do_constant_folding=False,
            dynamo=False,
        )
        decoder(z)
        torch.onnx.export(
            decoder,
            z,
            f"{export_path}_decoder_ts.onnx",
            export_params=True,
            opset_version=12,
            input_names=["latent_in"],
            output_names=["audio_out"],
            dynamic_axes={
                "latent_in": {0: "batch", 2: "latent_time"},
                "audio_out": {0: "batch", 2: "audio_time"},
            },
            do_constant_folding=False,
            dynamo=False,
        )
    else:
        pretrained(x)

        torch.onnx.export(
            pretrained,
            x,
            f"{export_path}_ts.onnx",
            export_params=True,
            opset_version=12,
            input_names=["audio_in"],
            output_names=["audio_out"],
            dynamic_axes={
                "audio_in": {2: "audio_length"},
                "audio_out": {0: "batch", 2: "audio_time"},
            },
            do_constant_folding=False,
            dynamo=False,
        )


def export_from_run():
    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    print(f"using {checkpoint}")

    pretrained = rave.RAVE()
    pretrained.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    pretrained.eval()

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)
        if hasattr(m, "warmed_up"):
            m.warmed_up = torch.tensor(1)

    def recursive_replace(model: nn.Module):
        for name, child in model.named_children():
            if isinstance(child, cc.convs.Conv1d):
                conv = nn.Conv1d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child._pad[0],
                    child.dilation,
                    child.groups,
                    child.bias,
                )
                conv.weight.data.copy_(child.weight.data)
                if conv.bias is not None:
                    conv.bias.data.copy_(child.bias.data)
                setattr(model, name, conv)
            elif isinstance(child, cc.convs.ConvTranspose1d):
                conv = nn.ConvTranspose1d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.output_padding,
                    child.groups,
                    child.bias,
                    child.dilation,
                    child.padding_mode,
                )
                conv.weight.data.copy_(child.weight.data)
                if conv.bias is not None:
                    conv.bias.data.copy_(child.bias.data)
                setattr(model, name, conv)
            else:
                recursive_replace(child)

    recursive_replace(pretrained)

    if FLAGS.latent_dims is not None:
        latent_size = FLAGS.latent_dims
        if latent_size > pretrained.latent_size:
            raise ValueError(
                f"--latent_dims {latent_size} exceeds the full latent size "
                f"{pretrained.latent_size}")
        if latent_size < 1 or (latent_size & (latent_size - 1)) != 0:
            raise ValueError(
                f"--latent_dims must be a positive power of 2, got {latent_size}")
    elif isinstance(pretrained.encoder, rave.blocks.VariationalEncoder):
        latent_size = max(
            np.argmax(pretrained.fidelity.numpy() > FLAGS.fidelity), 1)
        latent_size = 2**math.ceil(math.log2(latent_size))
    else:
        latent_size = pretrained.latent_size

    x = torch.randn(1, pretrained.n_channels, 2**15)
    name = os.path.basename(os.path.normpath(FLAGS.run))
    export_path = os.path.join(FLAGS.run, name)

    if FLAGS.split:
        encoder, decoder = MockEncoder(pretrained, latent_size), MockDecoder(pretrained, latent_size)
        z = encoder(x)
        torch.onnx.export(
            encoder,
            x,
            f"{export_path}_encoder.onnx",
            export_params=True,
            opset_version=12,
            input_names=["audio_in"],
            output_names=["latent_out"],
            dynamic_axes={
                "audio_in": {2: "audio_length"},
                "latent_out": {0: "batch", 2: "latent_time"},
            },
            do_constant_folding=False,
            dynamo=False,
        )
        decoder(z)
        torch.onnx.export(
            decoder,
            z,
            f"{export_path}_decoder.onnx",
            export_params=True,
            opset_version=12,
            input_names=["latent_in"],
            output_names=["audio_out"],
            dynamic_axes={
                "latent_in": {0: "batch", 2: "latent_time"},
                "audio_out": {0: "batch", 2: "audio_time"},
            },
            do_constant_folding=False,
            dynamo=False,
        )
    else:
        pretrained(x)

        torch.onnx.export(
            pretrained,
            x,
            f"{export_path}.onnx",
            export_params=True,
            opset_version=12,
            input_names=["audio_in"],
            output_names=["audio_out"],
            dynamic_axes={
                "audio_in": {2: "audio_length"},
                "audio_out": {0: "batch", 2: "audio_time"},
            },
            do_constant_folding=False,
            dynamo=False,
        )


if __name__ == "__main__":
    app.run(main)
