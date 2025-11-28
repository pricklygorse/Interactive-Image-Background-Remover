# --- onnx_worker.py ---
import sys
import json
import base64

import numpy as np
import onnxruntime as ort


def main():
    raw = sys.stdin.read()
    if not raw:
        raise RuntimeError("onnx_worker: No input on stdin")

    data = json.loads(raw)

    model_path = data["model_path"]
    prefer_gpu = data.get("prefer_gpu", True)

    # Reconstruct input tensor
    input_b64 = data["input"]
    input_shape = data["input_shape"]
    arr_bytes = base64.b64decode(input_b64)
    inp = np.frombuffer(arr_bytes, dtype=np.float32).reshape(input_shape)

    # Build session with CPU arena disabled (matches your old behavior)
    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if prefer_gpu and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=providers,
    )

    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: inp})[0].astype(np.float32)

    out_bytes = result.tobytes()
    out_b64 = base64.b64encode(out_bytes).decode("utf8")

    # Print just the base64 string
    sys.stdout.write(out_b64)


if __name__ == "__main__":
    main()
