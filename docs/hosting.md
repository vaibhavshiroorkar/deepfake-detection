# Hosting the public demo

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
