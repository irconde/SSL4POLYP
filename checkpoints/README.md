# Checkpoints directory

Use this folder as the default location for trained model weights and
snapshots produced by experiments. Suggested structure:

```
checkpoints/
├─ pretrained/              # immutable inputs (drop in once)
│  └─ vit_b/
│     ├─ mae_imagenet.pth
│     └─ mae_hyperkvasir.pth
└─ classification/          # outputs from fine-tuning jobs
    ├─ exp01_seed13/
   │  ├─ best.pth
   │  ├─ last.pth
   │  ├─ tau.json
   │  └─ tb/
    ├─ exp01_seed29/
    ├─ exp01_seed47/
    └─ exp05b_seed13/
```

Place any non-downloadable ViT-B checkpoints under `pretrained/vit_b/` so the
YAML configs can reference them directly. Each experiment/training-seed
combination gets its own folder beneath `classification/`, named with the seed
value passed to the run (e.g. `exp01_seed13`). This is where the training script
will drop model snapshots, thresholds, and TensorBoard logs by default. Use the
`--output-dir` flag if you need to override this layout.

### SUN baseline naming

Baseline SUN checkpoints produced by experiments 1–4 now follow the same
canonical naming convention as the training script (`<ModelTag>__SUNFull_s{seed}`)
where `<ModelTag>` is the camel-cased form emitted by `_canonicalize_tag` (for
example `SUPImNet`, `SSLImNet`, `SSLColon`). If you have older runs using
lower-case stems such as `sup_imnet__SUNFull_s13.pth`, rename them to match the
new convention so downstream jobs (e.g. `scripts/run_exp5a.sh`) can locate the
parents without manual overrides.
