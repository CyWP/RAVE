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

## Expected output

Three `.onnx` files in the run directory when using `--split`:
- `<run_name>_encoder.onnx`
- `<run_name>_decoder.onnx`
- `<run_name>.onnx` (combined, for non-split)
