# Hosting the public demo

There are two ways to host the demo. The browser demo is the permanent public
link and costs nothing to run. The GPU PC demo serves the full ensemble but
lasts only while the PC is on.

## Browser demo on a static Hugging Face Space

The demo runs at
[huggingface.co/spaces/vaibhavshiroorkar/deepfake-generalization](https://huggingface.co/spaces/vaibhavshiroorkar/deepfake-generalization).
It is a static Space, so there is no server. The visitor's browser downloads
about 110 MB of ONNX models once and runs everything locally with
onnxruntime-web, on WebGPU where available and WebAssembly otherwise. Uploaded
video never leaves the device.

It serves the best single model (`visual-best`) with the familiarity gate. The
six-model ensemble is too large to download into a browser.
`hosting/browser/README.md` describes the port and how closely it matches the
Python results.

The models live in `browser/` of the model repo
[vaibhavshiroorkar/deepfake-generalization-demo](https://huggingface.co/vaibhavshiroorkar/deepfake-generalization-demo),
next to the runtime the GPU PC demo below uses. To rebuild them, point
`DDF_PROJECT_ROOT` at a demo runtime and run:

```powershell
uv run --with onnx --with onnxruntime --with onnxscript python hosting/browser/export.py out/
```

Upload `out/` to `browser/` in the model repo. To ship a page change, upload
the files in `hosting/browser/` except `export.py` to the Space repository.

## GPU PC with a temporary tunnel

The Windows demo uses this PC's GPU and a temporary Cloudflare Tunnel.
The PC must stay awake. Stopping the tunnel removes public access.
The tunnel hostname can change after a restart.

From the repository folder, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/public_demo.ps1 -Action start
```

The command prints the public HTTPS URL after model loading finishes.
It downloads a pinned Cloudflared release and checks its SHA-256 before use.
It serves only the visual ensemble and single visual model, with a 90 MiB upload limit.
The older multimodal head remains disabled until its corrected training is complete.

The runtime lives under `runs/hosting/`. It contains copies of the six visual
checkpoints, the familiarity model, selected visual evaluation files and settings.
It contains no dataset directory, so local dataset samples are unavailable.
Visitors can upload their own clips. Upload files are removed after each request.
Admin access is disabled for this public demo.

Check status or stop both background processes:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/public_demo.ps1 -Action status
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/public_demo.ps1 -Action stop
```

The stop command checks the recorded process commands before stopping them.
The current state is in `runs/hosting/current-demo.json`. Server and tunnel logs
are in the directory named by that file. Model-loading errors appear in
`server.stderr.log`; tunnel connection errors appear in `tunnel.stderr.log`.

A fresh checkout needs Python 3.11 through 3.13, `uv`, FFmpeg, a compatible
NVIDIA driver and the ignored model artifacts. Install the environment with:

```powershell
uv sync --locked --extra cu130 --extra ml --extra media --extra web
```

The model files do not come from Git. Copy the visual checkpoints from
`runs/program-20260906/checkpoints/` and the familiarity model from
`runs/probe-20260914/authenticity-density.npz` before starting.

This setup is a temporary research demo. Anyone with its URL can access inference.
For continuous hosting, move the runtime to a GPU server with persistent storage,
authentication and automatic service restarts. The app listens on port 8000;
cloud deployments should bind to `0.0.0.0`.

See [Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/)
for temporary URL behavior, and [Runpod HTTP ports](https://docs.runpod.io/pods/configuration/expose-ports)
for GPU server routing.
