# timm 0.3.2 vs 0.4.12: MAE forward/backward equivalence

Date: 2026-11-05 (run on CPU at commit `c3bae2c`). It repeats the check recorded in `results_v3/RUNLOG.md`
(2026-10-16, "timm version guard relaxed"), whose script and output had not been kept.

`fwdbwd.py` builds `mae_vit_base_patch16` (`norm_pix_loss=False`) with seed 0, runs one forward and backward pass on a
fixed input (2 × 3 × 224 × 224, generator seed 123; mask seed 7, ratio 0.75) and saves the parameter names, the initial
weights, the loss and the gradients. `compare.py` compares the two dumps.

From the repository root (the dumps are about 0.9 GB each; write them outside the repository):

```bash
conda run -n ssl4polyp-pretrain python docs/verification/timm_equivalence/fwdbwd.py <tmp>/fb_032.pt    # timm 0.3.2
conda run -n ssl4polyp-ft       python docs/verification/timm_equivalence/fwdbwd.py <tmp>/fb_0412.pt   # timm 0.4.12
conda run -n ssl4polyp-ft       python docs/verification/timm_equivalence/compare.py <tmp>/fb_032.pt <tmp>/fb_0412.pt
```

Result (`output.txt`), both environments with torch 2.4.1+cu121: 111,907,840 parameters with the same names and order;
initial weights max |diff| 0; loss 1.85469853878021 in both (bitwise equal); gradients max |diff| 0 (bitwise equal).
The loss differs from the 1.864511489868164 of the RUNLOG check, whose script (and so its input and mask) was not kept.
