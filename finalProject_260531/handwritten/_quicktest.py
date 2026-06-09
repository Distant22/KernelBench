"""Quick standalone correctness+timing test for a hand-written ModelNew.

Usage: python _quicktest.py <kernel_file.py> <problem_id>
Compares ModelNew against the KernelBench level1 reference Model on V100.
"""
import importlib.util
import math
import sys
import time
import torch

torch.manual_seed(0)

KB = "/work/distant22/KernelBench/KernelBench/level1"
REFS = {
    97: "97_ScaledDotProductAttention.py",
    76: "76_conv_standard_1D_dilated_strided__.py",
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    kfile = sys.argv[1]
    pid = int(sys.argv[2])
    dev = torch.device("cuda:0")

    ref = load_module(f"{KB}/{REFS[pid]}", f"ref{pid}")
    kern = load_module(kfile, f"kern{pid}")

    init = ref.get_init_inputs()
    inputs = ref.get_inputs()
    inputs = [x.to(dev) if isinstance(x, torch.Tensor) else x for x in inputs]

    model = ref.Model(*init).to(dev)
    try:
        model_new = kern.ModelNew(*init).to(dev)
    except TypeError:
        model_new = kern.ModelNew().to(dev)

    # copy weights if any
    try:
        model_new.load_state_dict(model.state_dict(), strict=False)
    except Exception as e:
        print("  (load_state_dict skipped:", e, ")")

    with torch.no_grad():
        out_ref = model(*inputs)
        torch.cuda.synchronize()
        print("compiled+ran reference. out shape", tuple(out_ref.shape))
        out_new = model_new(*inputs)
        torch.cuda.synchronize()
        print("compiled+ran ModelNew. out shape", tuple(out_new.shape))

        tol = 1e-2
        # chunked max-abs-diff to avoid materializing several full-size temporaries
        a = out_ref.reshape(-1)
        b = out_new.reshape(-1)
        n = a.numel()
        chunk = 1 << 24
        max_diff = 0.0
        for i in range(0, n, chunk):
            d = torch.max(torch.abs(a[i:i + chunk] - b[i:i + chunk])).item()
            if d > max_diff:
                max_diff = d
        ok = max_diff <= tol
        print(f"max_abs_diff={max_diff:.3e}  correct(<= {tol})={ok}")

        def bench(fn, n=50):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.time() - t0) / n * 1e3

        t_new = bench(lambda: model_new(*inputs))
        t_ref = bench(lambda: model(*inputs))
        print(f"kernel={t_new:.3f} ms  eager={t_ref:.3f} ms  speedup={t_ref/t_new:.3f}x")


if __name__ == "__main__":
    main()
