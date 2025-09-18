# Outputs directory

Store experiment metadata, TensorBoard runs, predictions and other
intermediate artefacts that you want to keep separate from reusable
checkpoints. For example:

```
outputs/
├─ exp01/
│  ├─ tb/
│  ├─ predictions.csv
│  └─ notes.md
└─ mae/
   └─ pretrain_logs/
```

Feel free to reorganise or override destinations via command-line flags when
running the provided scripts.
