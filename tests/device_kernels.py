"""The kernels' simulator tests on a NeuronCore (TRNRT_TEST_DEVICE=1, conftest.py): nki.simulate(kernel)(*args) becomes
"compile the kernel through wrap_nki in a torch.compile graph, as the engine does, run it on the core, hand the outputs
back as the simulator would". Every needs_nki test then checks the device's output against its emulation or its
expected values, on the host's Neuron SDK, with no test of its own.

The core is NEURON_RT_VISIBLE_CORES (default 0); a NEFF holding collectives needs NEURON_RT_ROOT_COMM_ID, set here to a
port from TRNRT_TEST_DEVICE_PORT (default 62190) unless the caller set it. Graphs compile into NEURON_LIBTORCH_CACHE_ROOT
(default a directory under the system temp dir), so a second run loads them.
"""

from __future__ import annotations

import inspect
import os
import tempfile

import numpy as np
import torch

_STATE: dict = {}  # "device", "backend": set once, at the first kernel run


def _setup():
    if _STATE:
        return
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{os.environ.get('TRNRT_TEST_DEVICE_PORT', '62190')}")
    os.environ.setdefault("NEURON_LIBTORCH_CACHE_ROOT", os.path.join(tempfile.gettempdir(), "trnrt_device_kernels"))
    os.environ["NEURON_LIBTORCH_ASSERT_CACHE_HIT"] = "0"
    from trnrt import platform

    platform.configure_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401  registers the "neuron" device

    from trnrt.engine.model_runner import canonical_neuron_backend

    _STATE.update(device=torch.device("neuron:0"), backend=canonical_neuron_backend())


def _np_to_torch(a: np.ndarray) -> torch.Tensor:
    name = a.dtype.name
    if name == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(a).view(np.uint16)).view(torch.bfloat16)
    if name.startswith("float8_e4m3"):
        return torch.from_numpy(np.ascontiguousarray(a).view(np.uint8)).view(torch.float8_e4m3fn)
    if name.startswith("float8_e5m2"):
        return torch.from_numpy(np.ascontiguousarray(a).view(np.uint8)).view(torch.float8_e5m2)
    return torch.from_numpy(np.ascontiguousarray(a))


def _torch_to_np(t: torch.Tensor, like: np.dtype | None):
    t = t.cpu()
    if t.dtype == torch.bfloat16:
        import ml_dtypes

        return t.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        import ml_dtypes

        dt = ml_dtypes.float8_e4m3fn if t.dtype == torch.float8_e4m3fn else ml_dtypes.float8_e5m2
        return t.view(torch.uint8).numpy().view(dt)
    return t.numpy()


def simulate(kernel):
    """nki.simulate's signature, run on the device."""
    import nki
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    if not isinstance(kernel, nki.jit(lambda: None).__class__):  # a plain function: nki.simulate wraps it
        kernel = nki.jit(kernel)
    grid = getattr(kernel, "lnc", None) or 1
    names = list(inspect.signature(kernel.func).parameters)

    def run(*args, **kwargs):
        _setup()
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1"))  # the runtime's (tests pin platform.nki_grid)
        if grid != lnc:  # neuronx-cc refuses it (NCC_EVRF066): a simulator-only case
            import pytest

            pytest.skip(f"a grid-{grid} kernel on an LNC {lnc} device")
        kw = dict(zip(names, args), **kwargs)
        numpy_in = any(isinstance(v, np.ndarray) for v in kw.values())
        tensors, consts = {}, {}
        for k, v in kw.items():
            if isinstance(v, np.ndarray):
                tensors[k] = _np_to_torch(v).to(_STATE["device"])
            elif isinstance(v, torch.Tensor):
                tensors[k] = v.to(_STATE["device"])
            else:
                consts[k] = v

        def graph(**t):
            return wrap_nki(kernel)[grid](**t, **consts)

        # One graph per run: dynamo's guards on this closure would hand a later run with other constants (or another
        # kernel) an earlier run's compiled graph. The NEFF itself stays in the compile cache.
        torch._dynamo.reset()
        out = torch.compile(graph, backend=_STATE["backend"], fullgraph=True, dynamic=False)(**tensors)
        conv = (lambda t: _torch_to_np(t, None)) if numpy_in else (lambda t: t.cpu())
        if isinstance(out, (tuple, list)):
            return type(out)(conv(o) for o in out)
        return conv(out)

    return run


def simulator_only(reason: str = "counts the instructions nki.simulate traces"):
    """Skip mark for a test that inspects the simulator itself (its trace, its instruction counts), which a device
    run does not produce."""
    import pytest

    return pytest.mark.skipif(os.environ.get("TRNRT_TEST_DEVICE") == "1", reason=f"simulator only: {reason}")


def install() -> None:
    """Route nki.simulate to the device (conftest.py, under TRNRT_TEST_DEVICE=1)."""
    import nki

    nki.simulate = simulate
