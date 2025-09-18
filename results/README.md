# Results directory

Collect evaluation exports, tables and figures in this folder so they remain
separate from raw experiment logs and checkpoints. A simple structure could
look like:

```
results/
├─ classification/
│  ├─ exp01_metrics.csv
│  └─ confusion_matrix.png
└─ mae/
   └─ linprobe/
      └─ summary.json
```

Commands that emit reports default to writing here but can be overridden on the
command line when required.
