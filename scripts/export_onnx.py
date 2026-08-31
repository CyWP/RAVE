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
flags.DEFINE_boolean(
    "streaming",
    default=False,
    help="Export a streamable model with explicit cached_conv state as inputs/outputs.",
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


class StreamingWrapper(nn.Module):
    """Turns cached_conv's internal streaming state into explicit tensor inputs/outputs.

    Forward: output, *new_states = wrapper(x, *states)
    """

    def __init__(self, model, state_filter=None):
        super().__init__()
        self.model = model
        self.state_modules = []
        for name, module in model.named_modules():
            if isinstance(module, cc.CachedPadding1d):
                if state_filter is None or state_filter(name):
                    self.state_modules.append((name, module, "pad"))
            elif isinstance(module, cc.CachedConvTranspose1d):
                if state_filter is None or state_filter(name):
                    self.state_modules.append((name, module, "cache"))

    def forward(self, x, *states):
        if len(states) != len(self.state_modules):
            raise ValueError(
                f"Expected {len(self.state_modules)} states, got {len(states)}"
            )
        for state, (_, module, attr) in zip(states, self.state_modules):
            setattr(module, attr, state)
        y = self.model(x)
        new_states = tuple(getattr(module, attr) for _, module, attr in self.state_modules)
        return (y, *new_states)


def make_streaming_model(model, example_input, state_filter=None):
    """Prepare a cached-conv model for explicit-state streaming.

    Returns: wrapper, initial_states, state_names
    """
    model.eval()
    wrapper = StreamingWrapper(model, state_filter=state_filter)
    cc.use_cached_conv(True)

    # Run once to let CachedPadding1d.init_cache / CachedConvTranspose1d.init_cache
    # create the buffer tensors with the correct shapes.
    with torch.no_grad():
        wrapper.model(example_input)

    # Clone the initial state tensors.
    all_states = []
    for name, module, attr in wrapper.state_modules:
        try:
            all_states.append(getattr(module, attr).clone())
        except AttributeError:
            all_states.append(None)

    # Clear internal cached_conv state so the wrapper receives it explicitly.
    # Filter out zero-size and uninitialized states.
    filtered = []
    filtered_states = []
    for i, (name, module, attr) in enumerate(wrapper.state_modules):
        if all_states[i] is None:
            continue
        mod = module
        if hasattr(mod, '_buffers') and attr in mod._buffers:
            del mod._buffers[attr]
        mod.initialized = 1
        if all_states[i].numel() > 0:
            filtered.append((name, module, attr))
            filtered_states.append(all_states[i])

    wrapper.state_modules = filtered
    state_names = tuple(name for name, _, _ in wrapper.state_modules)
    return wrapper, tuple(filtered_states), state_names


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
    if FLAGS.streaming:
        cc.use_cached_conv(True)

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

    if not FLAGS.streaming:
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

    if FLAGS.streaming and FLAGS.split:
        # Split streaming: export separate streaming encoder and decoder
        encoder = MockEncoder(pretrained, latent_size)
        decoder = MockDecoder(pretrained, latent_size)

        def _encoder_filter(name):
            if "pretrained.pqmf.inverse_conv" in name:
                return False
            return "pretrained.pqmf" in name or "pretrained.encoder" in name

        def _decoder_filter(name):
            return "pretrained.decoder" in name

        # Build encoder wrapper
        enc_wrapper = StreamingWrapper(encoder, state_filter=_encoder_filter)
        with torch.no_grad():
            enc_wrapper.model(x)  # initialize encoder caches
        enc_initial = []
        enc_filtered = []
        for name, module, attr in enc_wrapper.state_modules:
            try:
                state = getattr(module, attr).clone()
            except AttributeError:
                continue
            if state.numel() > 0:
                enc_initial.append(state)
                enc_filtered.append((name, module, attr))
            if hasattr(module, '_buffers') and attr in module._buffers:
                del module._buffers[attr]
            module.initialized = 1
        enc_wrapper.state_modules = enc_filtered

        # Build decoder wrapper
        dec_wrapper = StreamingWrapper(decoder, state_filter=_decoder_filter)
        z_dummy = torch.randn(1, latent_size, 2**12)
        with torch.no_grad():
            dec_wrapper.model(z_dummy)  # initialize decoder caches
        dec_initial = []
        dec_filtered = []
        for name, module, attr in dec_wrapper.state_modules:
            try:
                state = getattr(module, attr).clone()
            except AttributeError:
                continue
            if state.numel() > 0:
                dec_initial.append(state)
                dec_filtered.append((name, module, attr))
            if hasattr(module, '_buffers') and attr in module._buffers:
                del module._buffers[attr]
            module.initialized = 1
        dec_wrapper.state_modules = dec_filtered

        print(f"Encoder: {len(enc_filtered)} state modules")
        for i, (n, s) in enumerate(zip(
                [n for n, _, _ in enc_wrapper.state_modules], enc_initial)):
            print(f"  {i}: {n} {tuple(s.shape)}")
        print(f"Decoder: {len(dec_filtered)} state modules")
        for i, (n, s) in enumerate(zip(
                [n for n, _, _ in dec_wrapper.state_modules], dec_initial)):
            print(f"  {i}: {n} {tuple(s.shape)}")

        # Export streaming encoder
        enc_inputs = (x, *enc_initial)
        enc_in_names = ["audio_in"] + [f"state_{i}" for i in range(len(enc_initial))]
        enc_out_names = ["latent_out"] + [f"state_{i}_out" for i in range(len(enc_initial))]
        torch.onnx.export(
            enc_wrapper, enc_inputs, f"{export_path}_encoder_streaming.onnx",
            export_params=True, opset_version=12,
            input_names=enc_in_names, output_names=enc_out_names,
            dynamic_axes={
                "audio_in": {2: "audio_length"},
                "latent_out": {0: "batch", 2: "latent_time"},
            },
            do_constant_folding=True, dynamo=False,
        )

        # Export streaming decoder
        z = torch.randn(1, latent_size, 2**12)
        dec_inputs = (z, *dec_initial)
        dec_in_names = ["latent_in"] + [f"state_{i}" for i in range(len(dec_initial))]
        dec_out_names = ["audio_out"] + [f"state_{i}_out" for i in range(len(dec_initial))]
        torch.onnx.export(
            dec_wrapper, dec_inputs, f"{export_path}_decoder_streaming.onnx",
            export_params=True, opset_version=12,
            input_names=dec_in_names, output_names=dec_out_names,
            dynamic_axes={
                "latent_in": {0: "batch", 2: "latent_time"},
                "audio_out": {0: "batch", 2: "audio_time"},
            },
            do_constant_folding=True, dynamo=False,
        )
    elif FLAGS.streaming:
        wrapper, initial_states, state_names = make_streaming_model(pretrained, x)

        print(f"Discovered {len(state_names)} cached_conv state modules:")
        for i, (sname, state) in enumerate(zip(state_names, initial_states)):
            print(f"  {i}: {sname} {tuple(state.shape)}")

        inputs = (x, *initial_states)
        input_names = ["audio_in"] + [f"state_{i}" for i in range(len(initial_states))]
        output_names = ["audio_out"] + [f"state_{i}_out" for i in range(len(initial_states))]

        torch.onnx.export(
            wrapper,
            inputs,
            f"{export_path}_streaming.onnx",
            export_params=True,
            opset_version=12,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "audio_in": {2: "audio_length"},
                "audio_out": {0: "batch", 2: "audio_time"},
            },
            do_constant_folding=True,
            dynamo=False,
        )
    elif FLAGS.split:
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
