#!/bin/bash
# Resume the Xception download until the expected size is reached, then verify and log it.
F='D:\cache\weights\xception-43020ad28.pth'
URL=https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-cadene/xception-43020ad28.pth
for i in $(seq 1 300); do
  sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
  [ "$sz" = "91675053" ] && break
  curl -sSL -C - -o "$F" "$URL" --connect-timeout 20 --max-time 600 || sleep 5
done
cd "d:/Final Year Project" && PYTHONPATH=src HF_HUB_CACHE='D:\cache\hf\hub' TORCH_HOME='D:\cache\torch' .venv/Scripts/python.exe scripts/sweep/fetch_encoders.py --only IMG-04
