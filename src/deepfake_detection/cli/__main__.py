from deepfake_detection.cli import main

# Guarded: on Windows, DataLoader workers re-import the main module, and an
# unguarded call would start a second CLI run inside every worker.
if __name__ == "__main__":
    raise SystemExit(main())
