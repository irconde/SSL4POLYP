# Checkpoints directory

Use this folder as the default location for trained model weights and
snapshots produced by experiments. Suggested structure:

```
checkpoints/
├─ classification/
│  ├─ exp01/
│  │  ├─ best.pth
│  │  └─ tb/
│  └─ ...
└─ mae/
   ├─ pretrain/
   └─ finetune/
```

Training scripts default to saving outputs here but accept command-line flags
to override the destination if you prefer a custom layout.
