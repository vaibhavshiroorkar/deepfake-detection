"""Reusable Streamlit pieces for the stream pages: the Train and Run tabs.

The dashboard's contract is *never trains in-process* (app.py, PROJECT_OVERVIEW
.md §7), so the Train tab only gathers hyperparameters and emits the exact
background-trainer command to launch on a GPU box — it never runs a training
loop here. The Run tab performs real-clip inference via dashboard.lib.inference.

train_command is a pure function (no Streamlit) so it stays unit-testable.
"""

# Defaults mirror the training fields on StreamConfig so the command matches
# what the background trainer would otherwise read from the config.
TRAIN_DEFAULTS = {
    "epochs": 8, "batch_size": 2, "grad_accum_steps": 8,
    "lr_head": 1e-3, "lr_backbone": 5e-6, "weight_decay": 1e-4,
    "grad_clip_norm": 1.0, "seed": 42,
}

# Which data to train on is a training decision, so it belongs in the emitted
# command rather than being implied by whatever the dashboard happened to have
# selected. Splits default to the pipeline's names.
DATA_DEFAULTS = {"dataset": "", "train_split": "train", "val_split": "val"}


def train_command(stream_name: str, backbone_name: str, settings: dict) -> str:
    """Build the background-trainer CLI string for the chosen hyperparameters.

    Data flags are emitted only when a dataset was chosen, so a command built
    without the picker stays exactly as short as it used to be.
    """
    parts = [
        "uv run python -m training.train_visual",
        f"--stream {stream_name}",
        f"--backbone {backbone_name}",
    ]
    if settings.get("dataset"):
        parts += [
            f"--dataset {settings['dataset']}",
            f"--train-split {settings.get('train_split', DATA_DEFAULTS['train_split'])}",
            f"--val-split {settings.get('val_split', DATA_DEFAULTS['val_split'])}",
        ]
    parts += [
        f"--epochs {int(settings['epochs'])}",
        f"--batch-size {int(settings['batch_size'])}",
        f"--grad-accum {int(settings['grad_accum_steps'])}",
        f"--lr-head {settings['lr_head']:g}",
        f"--lr-backbone {settings['lr_backbone']:g}",
        f"--weight-decay {settings['weight_decay']:g}",
        f"--grad-clip {settings['grad_clip_norm']:g}",
        f"--seed {int(settings['seed'])}",
    ]
    return " ".join(parts)


def render_train_data_picker(st, key: str, enabled: bool) -> dict:
    """Dataset + train/val split selects for the Train tab.

    Independent of the Run tab's clip: training reads whole splits, and the clip
    chosen for a single forward pass has nothing to do with what to train on.
    """
    from dashboard.lib import selectors

    found = selectors.registry()
    if not found:
        st.info("No datasets discovered under `data/` — the trainer needs one. "
                "Drop a dataset in and rescan on the Preprocessing page.")
        return dict(DATA_DEFAULTS)

    names = list(found)
    c1, c2, c3 = st.columns(3)
    dataset = c1.selectbox("Dataset", names, key=f"{key}_ds", disabled=not enabled)
    splits = found[dataset].splits
    train_split = c2.selectbox(
        "Train split", splits, key=f"{key}_train_split", disabled=not enabled,
        index=splits.index("train") if "train" in splits else 0)
    val_split = c3.selectbox(
        "Val split", splits, key=f"{key}_val_split", disabled=not enabled,
        index=splits.index("val") if "val" in splits else 0)
    return {"dataset": dataset, "train_split": train_split, "val_split": val_split}


def render_train_tab(st, key: str, stream_name: str, backbone_name: str, enabled: bool):
    """Training hyperparameters + a Launch button that emits the trainer command."""
    st.caption("Configures the background trainer. The dashboard never trains in-process (§7) — "
               "Launch emits the command to run on your training box.")
    st.markdown("**Data**")
    data = render_train_data_picker(st, key, enabled)
    st.markdown("**Hyperparameters**")
    c1, c2, c3 = st.columns(3)
    d = TRAIN_DEFAULTS
    epochs = c1.number_input("Epochs", 1, 100, d["epochs"], key=f"{key}_epochs", disabled=not enabled)
    batch = c1.number_input("Batch size", 1, 64, d["batch_size"], key=f"{key}_bs", disabled=not enabled)
    accum = c1.number_input("Grad accum steps", 1, 64, d["grad_accum_steps"], key=f"{key}_accum",
                            disabled=not enabled)
    lr_head = c2.number_input("LR (head)", 0.0, 1.0, d["lr_head"], step=1e-4, format="%.6f",
                              key=f"{key}_lrh", disabled=not enabled)
    lr_backbone = c2.number_input("LR (backbone)", 0.0, 1.0, d["lr_backbone"], step=1e-6, format="%.6f",
                                  key=f"{key}_lrb", disabled=not enabled)
    weight_decay = c2.number_input("Weight decay", 0.0, 1.0, d["weight_decay"], step=1e-4, format="%.6f",
                                   key=f"{key}_wd", disabled=not enabled)
    grad_clip = c3.number_input("Grad clip norm", 0.0, 10.0, d["grad_clip_norm"], step=0.1,
                                key=f"{key}_clip", disabled=not enabled)
    seed = c3.number_input("Seed", 0, 999999, d["seed"], key=f"{key}_seed", disabled=not enabled)

    if st.button("Launch training", key=f"{key}_train", disabled=not enabled, type="primary"):
        settings = {"epochs": epochs, "batch_size": batch, "grad_accum_steps": accum,
                    "lr_head": lr_head, "lr_backbone": lr_backbone, "weight_decay": weight_decay,
                    "grad_clip_norm": grad_clip, "seed": seed, **data}
        st.success("Queued — run this on your training box:")
        st.code(train_command(stream_name, backbone_name, settings), language="bash")
        st.caption("The `training.train_visual` entrypoint lands in a later stage; the dashboard "
                   "never trains itself.")


def render_run_tab(st, key: str, make_config, video_path, enabled: bool):
    """Build + inspect the model (folded-in 'Build'), then infer on the real clip.

    make_config() -> StreamConfig is called lazily so the model is only built on
    click. video_path is the selected clip's path (or None if none is selected).
    """
    st.caption("Builds the config-driven model, reports its architecture, then forward-passes the "
               "selected clip's face-crop sequence. Weights are untrained — no checkpoint exists "
               "yet — so the probability is a plumbing check, not a real detection.")
    c1, c2 = st.columns(2)
    c1.selectbox("Checkpoint", ["(untrained — random weights)"], key=f"{key}_ckpt",
                 disabled=True, help="No trained checkpoints exist yet; trained weights load here later.")
    fixed_seed = c2.checkbox(
        "Fixed seed", value=True, key=f"{key}_run_fixed", disabled=not enabled,
        help="Seed the RNG before building so the same config always gives the same untrained "
             "output. Uncheck for fresh random weights (a different number) each run.")
    seed = c2.number_input("Seed", 0, 999999, 42, key=f"{key}_run_seed",
                           disabled=not enabled or not fixed_seed)

    if not st.button("Build & run on clip", key=f"{key}_run", disabled=not enabled, type="primary"):
        return

    import torch
    from models.streams.common.visual_stream import build_visual_stream

    config = make_config()
    with st.spinner("Building model…"):
        # Seed right before construction so weight init is reproducible (the model
        # is untrained, so its output is otherwise a different random draw each run).
        if fixed_seed:
            torch.manual_seed(int(seed))
        model = build_visual_stream(config).eval()
        counts = model.param_counts()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Backbone features", counts["feature_dim"])
    m2.metric("Embedding dim", counts["embedding_dim"])
    m3.metric("Total params", f"{counts['total'] / 1e6:.1f}M")
    m4.metric("Trainable params", f"{counts['trainable'] / 1e6:.1f}M")

    if video_path is None:
        st.info("Architecture shown above. Select a clip on the **Preprocessing** page to "
                "forward-pass a real sequence.")
        return
    from pathlib import Path
    if not Path(video_path).exists():
        st.warning(f"Clip file not found on disk:\n`{video_path}`")
        return

    from dashboard.lib import inference, media
    detector, _ = media.get_detector()
    with st.spinner("Detecting faces + running inference…"):
        result = inference.run_visual_stream(config, str(video_path), detector)

    st.metric("Fake probability (untrained)", f"{result['prob']:.3f}")
    st.code(
        f"input : [1, {config.num_frames}, 3, 224, 224]  ({result['num_frames']} face crops)\n"
        f"logit : {result['logit']:.4f}  (dev head, discarded at fusion)\n"
        f"embed : {result['embed_shape']}  -> feature store -> fusion",
        language="text",
    )


def render_disabled_train_run(st, key: str, note: str):
    """Disabled Train/Run tabs for streams/backbones that aren't wired yet."""
    t_tab, r_tab = st.tabs(["Train", "Run"])
    with t_tab:
        c1, c2, c3 = st.columns(3)
        c1.number_input("Epochs", 1, 100, TRAIN_DEFAULTS["epochs"], key=f"{key}_epochs", disabled=True)
        c1.number_input("Batch size", 1, 64, TRAIN_DEFAULTS["batch_size"], key=f"{key}_bs", disabled=True)
        c2.number_input("LR (head)", 0.0, 1.0, TRAIN_DEFAULTS["lr_head"], step=1e-4, format="%.6f",
                        key=f"{key}_lrh", disabled=True)
        c3.number_input("Seed", 0, 999999, TRAIN_DEFAULTS["seed"], key=f"{key}_seed", disabled=True)
        st.button("Launch training", key=f"{key}_train", disabled=True, help=note)
    with r_tab:
        st.selectbox("Checkpoint", ["(not available yet)"], key=f"{key}_ckpt", disabled=True)
        st.button("Build & run on clip", key=f"{key}_run", disabled=True, help=note)
    st.caption(note)
