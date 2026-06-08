import torch

torch.set_grad_enabled(False)
import os

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
FLAGS = flags.FLAGS


class MockEncoder(nn.Module):

    def __init__(self, pretrained: rave.RAVE):
        super().__init__()
        self.pretrained = pretrained

    def forward(self, x):
        z = self.pretrained.encode(x)
        return self.pretrained.encoder.reparametrize(z)[0]


class MockDecoder(nn.Module):
    def __init__(self, pretrained: rave.RAVE):
        super().__init__()
        self.pretrained = pretrained

    def forward(self, x):
        return self.pretrained.decode(x)


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
                "latent_out": [0],
            },
            do_constant_folding=False,
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
                "latent_in": [0],
                "audio_out": [0],
            },
            do_constant_folding=False,
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
                "audio_out": [0],
            },
            do_constant_folding=False,
        )


def export_from_run():
    breakpoint()
    gin.parse_config_file(os.path.join(FLAGS.run, "config.gin"))
    checkpoint = rave.core.search_for_run(FLAGS.run)

    print(f"using {checkpoint}")

    pretrained = rave.RAVE()
    pretrained.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    pretrained.eval()

    for m in pretrained.modules():
        if hasattr(m, "weight_g"):
            nn.utils.remove_weight_norm(m)

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

    x = torch.randn(1, pretrained.n_channels, 2**15)
    name = os.path.basename(os.path.normpath(FLAGS.run))
    export_path = os.path.join(FLAGS.run, name)

    if FLAGS.split:
        encoder, decoder = MockEncoder(pretrained), MockDecoder(pretrained)
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
                "latent_out": [0],
            },
            do_constant_folding=False,
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
                "latent_in": [0],
                "audio_out": [0],
            },
            do_constant_folding=False,
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
                "audio_out": [0],
            },
            do_constant_folding=False,
        )


if __name__ == "__main__":
    app.run(main)
