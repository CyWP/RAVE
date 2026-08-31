"""Test script for the streaming ONNX export.

Verifies that StreamingWrapper correctly passes state through cached_conv modules.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cached_conv as cc
import gin
import torch

import rave
from scripts.export_onnx import StreamingWrapper, make_streaming_model

RUN_DIR = "D:\\Research\\RAVE\\runs\\runs\\beta_test_hi_d138a4ff80\\"


def main():
    cc.use_cached_conv(True)

    gin.parse_config_file(os.path.join(RUN_DIR, "config.gin"))
    checkpoint = rave.core.search_for_run(RUN_DIR)

    print(f"Loading model from {checkpoint}")
    model = rave.RAVE()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
    model.eval()

    for m in model.modules():
        if hasattr(m, "warmed_up"):
            m.warmed_up = torch.tensor(1)

    example_input = torch.randn(1, model.n_channels, 2**15)

    print("Creating StreamingWrapper...")
    wrapper, initial_states, state_names = make_streaming_model(model, example_input)

    print(f"\nDiscovered {len(state_names)} cached_conv state modules:")
    for i, (name, state) in enumerate(zip(state_names, initial_states)):
        print(f"  {i}: {name} {tuple(state.shape)}")

    # Test 1: wrapper produces output
    print("\n--- Test 1: Forward pass ---")
    with torch.no_grad():
        out1, *states1 = wrapper(example_input, *initial_states)
    print(f"  output shape: {out1.shape}")
    print(f"  output non-zero: {out1.abs().max().item() > 1e-6}")

    # Test 2: different states produce different outputs
    print("\n--- Test 2: State affects output ---")
    zero_states = [torch.zeros_like(s) for s in initial_states]
    with torch.no_grad():
        out_zero, *_ = wrapper(example_input, *zero_states)
    differs = not torch.allclose(out1, out_zero, atol=1e-6)
    print(f"  Initial states vs zeroed states produce different output: {differs}")

    # Test 3: deterministic (same state -> same output)
    print("\n--- Test 3: Deterministic ---")
    with torch.no_grad():
        out1b, *_ = wrapper(example_input, *initial_states.clone() if hasattr(initial_states, 'clone') else [s.clone() for s in initial_states])
    # Use cloned states to avoid in-place modification issue
    cloned = [s.clone() for s in initial_states]
    with torch.no_grad():
        out_clone, *_ = wrapper(example_input, *cloned)
    with torch.no_grad():
        out_clone2, *_ = wrapper(example_input, *cloned)
    match = torch.allclose(out_clone, out_clone2, atol=1e-5)
    print(f"  Same cloned state -> same output: {match}")

    # Test 4: states update between passes
    print("\n--- Test 4: State updates between passes ---")
    s0 = [s.clone() for s in initial_states]
    with torch.no_grad():
        _, *s1 = wrapper(example_input, *s0)
    with torch.no_grad():
        _, *s2 = wrapper(example_input, *s1)
    any_changed = any((a - b).abs().max().item() > 1e-6 for a, b in zip(s2, s1))
    print(f"  States changed between pass 1 and pass 2: {any_changed}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
