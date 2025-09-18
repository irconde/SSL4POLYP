# Data directory

Store raw datasets, extracted frames, and manifest resources in this folder.
The repository keeps the directory structure under version control while
ignoring the actual data.

A recommended layout is:

```
data/
├─ roots.json              # maps manifest root identifiers to absolute paths
├─ roots.example.json      # sample template provided by the repo
├─ hyperkvasir/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ additional_dataset/
   └─ ...
```

Copy `roots.example.json` to `roots.json` and edit the paths to point to your
local storage before launching experiments.
