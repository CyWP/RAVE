# RAVE ONNX Export

## Environment

```powershell
$env:CONDA_ENV = "C:\conda_envs\rave"
& "$env:CONDA_ENV\python.exe" ...
```

## Test Command

```powershell
& "C:\conda_envs\rave\python.exe" -m scripts.export_onnx --split --run "D:\Research\RAVE\runs\runs\beta_test_hi_d138a4ff80\"
```

Omit `--split` for a combined (non-split) export.

## Streaming Export

```powershell
& "C:\conda_envs\rave\python.exe" -m scripts.export_onnx --streaming --run "D:\Research\RAVE\runs\runs\beta_test_hi_d138a4ff80\"
```

Exports a streamable model with cached_conv state as explicit inputs/outputs. Produces `<run_name>_streaming.onnx`.

Combine with `--split` for separate streaming encoder/decoder:

```powershell
& "C:\conda_envs\rave\python.exe" -m scripts.export_onnx --streaming --split --run "D:\Research\RAVE\runs\runs\beta_test_hi_d138a4ff80\"
```

Produces `<run_name>_encoder_streaming.onnx` and `<run_name>_decoder_streaming.onnx`.

```powershell
& "C:\conda_envs\rave\python.exe" scripts/test_streaming_export.py
```

## Options

- `--fidelity <float>` (default `0.95`, range `0.1`-`0.999`): For a variational encoder, derives the reduced latent size from the model's PCA fidelity curve. A higher fidelity keeps more latent channels (more controllable dims, lower reconstruction error). Only applies to split exports.
- `--latent_dims <int>` (optional): Explicit latent size (must be a power of 2 and not exceed the full latent size, otherwise it raises). Overrides `--fidelity` and forces the use of exactly that many PCA components.
- `--streaming` (optional): Export a streamable model with cached_conv state as explicit tensor inputs/outputs. The ONNX model takes `(audio_in, state_0, ..., state_N)` and returns `(audio_out, state_0_out, ..., state_N_out)`.

Combine with `--split` for separate streaming encoder/decoder:

```powershell
& "C:\conda_envs\rave\python.exe" -m scripts.export_onnx --streaming --split --run "D:\Research\RAVE\runs\runs\beta_test_hi_d138a4ff80\"
```

Produces `<run_name>_encoder_streaming.onnx` and `<run_name>_decoder_streaming.onnx`.

## Expected output

Three `.onnx` files in the run directory when using `--split`:
- `<run_name>_encoder.onnx`
- `<run_name>_decoder.onnx`
- `<run_name>.onnx` (combined, for non-split)
- `<run_name>_streaming.onnx` (when using `--streaming`)
- `<run_name>_encoder_streaming.onnx` (when using `--streaming --split`)
- `<run_name>_decoder_streaming.onnx` (when using `--streaming --split`)
