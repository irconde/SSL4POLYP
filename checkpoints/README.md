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
   ├─ exp01_seed42/
   │  ├─ best.pth
   │  ├─ last.pth
   │  ├─ tau.json
   │  └─ tb/
   ├─ exp01_seed47/
   └─ exp05b_seed29/
```

Place any non-downloadable ViT-B checkpoints under `pretrained/vit_b/` so the
YAML configs can reference them directly. Each experiment/seed combination gets
its own folder beneath `classification/`, which is where the training script
will drop model snapshots, thresholds, and TensorBoard logs by default. Use the
`--output-dir` flag if you need to override this layout.
