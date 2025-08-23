# SSL4GIE
Official code repository for *Evaluating Domain-Specific Self-Supervised Pre-training for Polyp Detection: A Morphology-Aware and Sample-Efficient Perspective*,
based on "A Study on Self-Supervised Pretraining for Vision Problems in Gastrointestinal Endoscopy"

Original authors: [Edward Sanderson](https://scholar.google.com/citations?user=ea4c7r0AAAAJ&hl=en&oi=ao) and [Bogdan J. Matuszewski](https://scholar.google.co.uk/citations?user=QlUO_oAAAAAJ&hl=en)

Repository author: Ivan Rodriguez-Conde, assistant professor at the Department of Computer Science at University of Vigo ([ivarodriguez@uvigo.gal](mailto:ivarodriguez@uvigo.gal))

Links to the original paper:
+ [IEEE Access (open access)](https://ieeexplore.ieee.org/document/10478725)
+ [arXiv](https://arxiv.org/abs/2401.06278)

## New functionality

We include helper scripts that automate common workflows introduced in the
paper:

- `Models/mae/run_hyperkvasir_pretraining.py` – wraps MAE pretraining on the
  Hyperkvasir-unlabelled dataset with the settings used in the paper. The
  script exposes `--batch-size`, `--epochs` and `--extra-args` so you can tweak
  the training run or forward additional options to `main_pretrain.py`.
- `Classification/run_all_pretrainings.py` – sequentially fine-tunes ViT-B/16
  under all three pretraining schemes (SUP-imnet, SSL-imnet and SSL-colon).
  Customise the architecture and batch size with `--arch` and `--batch-size`,
  and pass further arguments to `train_classification.py` via `--extra-args`.

The following sections describe how to invoke these utilities.

## Usage

### Pretraining

Follow the guidance in this section for obtaining the weights for pretrained models.

+ For encoders pretrained in a supervised manner with ImageNet-1k, the weights provided with the [timm](https://timm.fast.ai/) library (ViT-B) are automatically used by our code and no manual steps are required.
+ For encoders pretrained in a self-supervised manner with ImageNet-1k, using [MAE](https://github.com/facebookresearch/mae), the weights provided with the codebase should be used.
+ For encoders pretrained in a self-supervised manner with
  [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/), using
  [MAE](https://github.com/facebookresearch/mae), use the helper script
  `Models/mae/run_hyperkvasir_pretraining.py` to generate the checkpoint. You can
  override the default `--batch-size` and `--epochs` values and pass additional
  options to `main_pretrain.py` via `--extra-args`. Example:

```bash
python Models/mae/run_hyperkvasir_pretraining.py \
    --data-root /path/to/hyperkvasir-unlabelled \
    --output-dir /path/to/save \
    --batch-size 64 --epochs 400
```

### Finetuning

The finetuning scripts currently support frame-level classification. Pretrained weights produced by our experiments are available [here](https://drive.google.com/drive/folders/151BWqsjTV4PuGFxS20L0TpmUQ4DhhpU4?usp=sharing) (released under a CC BY-NC-SA 4.0 license). Place the desired checkpoint in `SSL4GIE/Classification/Trained models` if you wish to evaluate or make predictions with an already finetuned model.

Prior to running finetuning, download the required data and change directory:

+ For classification, download [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/) and change directory to `SSL4GIE/Classification`.

For finetuning models pretrained in a self-supervised manner, run the following:
```
python train_classification.py \
    --architecture [architecture] \
    --pretraining [pretraining] \
    --ss-framework [ss-framework] \
    --checkpoint [checkpoint] \
    --dataset [dataset] \
    --data-root [data-root] \
    --learning-rate-scheduler \
    --batch-size [batch-size]
```
For finetuning models pretrained in a supervised manner, or not pretrained at all, omit the `--ss-framework` and `--checkpoint` arguments:
```
python train_classification.py \
    --architecture [architecture] \
    --pretraining [pretraining] \
    --dataset [dataset] \
    --data-root [data-root] \
    --learning-rate-scheduler \
    --batch-size [batch-size]
```

* Replace `[architecture]` with name of encoder architecture (`vit_b`).
* Replace `[pretraining]` with general pretraining methodology (`Hyperkvasir`, `ImageNet_self`, `ImageNet_class`, or `random`).
* For models pretrained in a self-supervised manner, replace `[ss-framework]` with pretraining algorithm `mae` and `[checkpoint]` with path to pretrained weights (classification only).
* Replace `[dataset]` with name of dataset (e.g., `Hyperkvasir_anatomical` or `Hyperkvasir_pathological`).
* Replace `[data-root]` with path to the chosen dataset.
* Replace `[batch-size]` with desired batch size.

To run all three pretraining configurations sequentially (SUP-imnet, SSL-imnet and SSL-colon) you may use the helper script:
```
python run_all_pretrainings.py \
    --dataset [dataset] \
    --data-root [data-root] \
    --imagenet-mae /path/to/mae_pretrain_vit_base.pth \
    --hyperkvasir-mae /path/to/hyperkvasir_mae.pth
```
Optionally use `--arch` to choose a backbone (default `vit_b`) and `--batch-size`
to set the batch size for each run. Any additional options supported by
`train_classification.py` can be appended via `--extra-args`.

For all finetuning runs, the following optional arguments are also available:
+ `--frozen` - to freeze the pretrained encoder and only train the decoder. We did not use this in our experiments.
+ `--epochs [epochs]` - to set the number of epochs as desired (replace `[epochs]`). We did not use this in our experiments.
+ `--learning-rate-scheduler` - to use a learning rate scheduler that halves the learning rate when the performance on the validation set does not improve for 10 epochs. We did use this in our experiments.

Please also note that, when using MAE, code from the [MAE](https://github.com/facebookresearch/mae) repository is used, which is released under a CC BY-NC 4.0 license. Any results from such runs are therefore covered by a CC BY-NC 4.0 license.

### Evaluation

Ensure that the weights for the desired model are located in `SSL4GIE/Classification/Trained models`. This will have been done automatically if finetuning was run. Additionally, download the required data and change directory accordingly.

For evaluating classification models pretrained in a self-supervised manner, run the following:
```
python eval_classification.py \
    --architecture [architecture] \
    --pretraining [pretraining] \
    --ss-framework [ss-framework] \
    --dataset [dataset] \
    --data-root [data-root]
```
For evaluating classification models pretrained in a supervised manner, or not pretrained at all, omit the `--ss-framework` argument:
```
python eval_classification.py \
    --architecture [architecture] \
    --pretraining [pretraining] \
    --dataset [dataset] \
    --data-root [data-root]
```

* Replace `[architecture]` with name of encoder architecture (`vit_b`).
* Replace `[pretraining]` with general pretraining methodology (`Hyperkvasir`, `ImageNet_self`, `ImageNet_class`, or `random`).
* For models pretrained in a self-supervised manner, replace `[ss-framework]` with pretraining algorithm `mae`.
* Replace `[dataset]` with name of dataset (e.g., `Hyperkvasir_anatomical` or `Hyperkvasir_pathological`).
* Replace `[data-root]` with path to the chosen dataset.

In addition to printing the results of the evaluation in the output space, the results will also be saved to `SSL4GIE/eval_results.txt`.

### Prediction

Prediction utilities are not implemented for classification models.

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ESandML/SSL4GIE/blob/main/LICENSE) file. Please however note that when using MAE, code from the [MAE](https://github.com/facebookresearch/mae) repository is used, which is released under a CC BY-NC 4.0 license, and any results from such runs are therefore covered by a CC BY-NC 4.0 license.

## Citation

If you use this work, please consider citing us:
```bibtex
@article{sanderson2024study,
  title={A Study on Self-Supervised Pretraining for Vision Problems in Gastrointestinal Endoscopy},
  author={Sanderson, Edward and Matuszewski, Bogdan J.},
  journal={IEEE Access},
  year={2024},
  volume={12},
  number={},
  pages={46181-46201},
  doi={10.1109/ACCESS.2024.3381517}
}
```

## Commercial use

We allow commercial use of this work, as permitted by the [LICENSE](https://github.com/ESandML/SSL4GIE/blob/main/LICENSE). However, where possible, please inform us of this use for the facilitation of our impact case studies.

## Acknowledgements

This work was supported by the Science and Technology Facilities Council [grant number ST/S005404/1].

This work makes use of:
+ The [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/) dataset.
+ The [MAE](https://github.com/facebookresearch/mae) codebase. In addition to using this codebase for pretraining, as well as the weights provided, we also include the [MAE](https://github.com/facebookresearch/mae) repository in `SSL4GIE/Models`, with modifications made for version compatibility.

## Additional information

Links: [AIdDeCo Project](https://www.uclan.ac.uk/research/activity/machine-learning-cancer-detection), [CVML Group](https://www.uclan.ac.uk/research/activity/cvml)

Contact: esanderson4@uclan.ac.uk
