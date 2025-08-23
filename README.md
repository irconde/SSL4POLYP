# SSL4GIE
Official code repository for: A Study on Self-Supervised Pretraining for Vision Problems in Gastrointestinal Endoscopy

Authors: [Edward Sanderson](https://scholar.google.com/citations?user=ea4c7r0AAAAJ&hl=en&oi=ao) and [Bogdan J. Matuszewski](https://scholar.google.co.uk/citations?user=QlUO_oAAAAAJ&hl=en)

Links to the paper:
+ [IEEE Access (open access)](https://ieeexplore.ieee.org/document/10478725)
+ [arXiv](https://arxiv.org/abs/2401.06278)

## 1. Abstract

Solutions to vision tasks in gastrointestinal endoscopy (GIE) conventionally use image encoders pretrained in a supervised manner with ImageNet-1k as backbones. However, the use of modern self-supervised pretraining algorithms and a recent dataset of 100k unlabelled GIE images (Hyperkvasir-unlabelled) may allow for improvements. In this work, we study the fine-tuned performance of models with ResNet50 and ViT-B backbones pretrained in self-supervised and supervised manners with ImageNet-1k and Hyperkvasir-unlabelled (self-supervised only) in a range of GIE vision tasks. In addition to identifying the most suitable pretraining pipeline and backbone architecture for each task, out of those considered, our results suggest three general principles. Firstly, that self-supervised pretraining generally produces more suitable backbones for GIE vision tasks than supervised pretraining. Secondly, that self-supervised pretraining with ImageNet-1k is typically more suitable than pretraining with Hyperkvasir-unlabelled. Thirdly, that ViT-Bs are more suitable in polyp segmentation and monocular depth estimation in colonoscopy, ResNet50s are more suitable in polyp detection, and both architectures perform similarly in anatomical landmark recognition and pathological finding characterisation. We hope this work draws attention to the complexity of pretraining for GIE vision tasks, informs this development of more suitable approaches than the convention, and inspires further research on this topic to help advance this development.

## 2. Usage

### 2.1 Pretraining

Follow the guidance in this section for obtaining the weights for pretrained models.

+ For encoders pretrained in a supervised manner with ImageNet-1k, the weights provided with the [torchvision](https://pytorch.org/vision/stable/index.html) (ResNet50) and [timm](https://timm.fast.ai/) (ViT-B) libraries are automatically used by our code and no manual steps are required.
+ For encoders pretrained in a self-supervised manner with ImageNet-1k, using [MoCo v3](https://github.com/facebookresearch/moco-v3)/[Barlow Twins](https://github.com/facebookresearch/barlowtwins)/[MAE](https://github.com/facebookresearch/mae), the weights provided with the respective codebase should be used. Please note that the weights provided with the MoCo v3 codebase are for the full MoCo v3 model, including the momentum encoder, in a `torch.nn.distributed.DataDistributedParallel` wrapper - for use with this codebase, we recommend first building the full MoCo v3 model, wrapping it with `torch.nn.distributed.DataDistributedParallel`, loading in the provided weights, and saving the weights for `self.module.base_encoder` separately. Please also note that, in our experiments, we use the provided weights for the ResNet50 pretrained with MoCo v3 after 100 epochs, for consistency with the corresponding model pretrained with Hyperkvasir-unlabelled.
+ For encoders pretrained in a self-supervised manner with [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/), using [MoCo v3](https://github.com/facebookresearch/moco-v3)/[Barlow Twins](https://github.com/facebookresearch/barlowtwins)/[MAE](https://github.com/facebookresearch/mae), the respective codebase should be used for pretraining. The code should first be modified to allow pretraining with Hyperkvasir-unabelled (remove `'/train'` from data path) and the guidance in the respective codebase for running the pretraining should then be followed with any arguments adjusted for your given hardware as needed.

### 2.2 Finetuning

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

* Replace `[architecture]` with name of encoder architecture (`resnet50` or `vit_b`).
* Replace `[pretraining]` with general pretraining methodology (`Hyperkvasir`, `ImageNet_self`, `ImageNet_class`, or `random`).
* For models pretrained in a self-supervised manner, replace `[ss-framework]` with pretraining algorithm (`mocov3`, `barlowtwins`, or `mae`) and `[checkpoint]` with path to pretrained weights (classification only).
* Replace `[dataset]` with name of dataset (e.g., `Hyperkvasir_anatomical` or `Hyperkvasir_pathological`).
* Replace `[data-root]` with path to the chosen dataset.
* Replace `[batch-size]` with desired batch size.

For all finetuning runs, the following optional arguments are also available:
+ `--frozen` - to freeze the pretrained encoder and only train the decoder. We did not use this in our experiments.
+ `--epochs [epochs]` - to set the number of epochs as desired (replace `[epochs]`). We did not use this in our experiments.
+ `--learning-rate-scheduler` - to use a learning rate scheduler that halves the learning rate when the performance on the validation set does not improve for 10 epochs. We did use this in our experiments.

Please also note that, when using MoCo v3 or MAE, code from the [MoCo v3](https://github.com/facebookresearch/moco-v3) or [MAE](https://github.com/facebookresearch/mae) repositories is used, which are both released under CC BY-NC 4.0 licenses. Any results from such runs are therefore covered by a CC BY-NC 4.0 license.

### 2.3 Evaluation

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

* Replace `[architecture]` with name of encoder architecture (`resnet50` or `vit_b`).
* Replace `[pretraining]` with general pretraining methodology (`Hyperkvasir`, `ImageNet_self`, `ImageNet_class`, or `random`).
* For models pretrained in a self-supervised manner, replace `[ss-framework]` with pretraining algorithm (`mocov3`, `barlowtwins`, or `mae`).
* Replace `[dataset]` with name of dataset (e.g., `Hyperkvasir_anatomical` or `Hyperkvasir_pathological`).
* Replace `[data-root]` with path to the chosen dataset.

In addition to printing the results of the evaluation in the output space, the results will also be saved to `SSL4GIE/eval_results.txt`.

### 2.4 Prediction

Prediction utilities are not implemented for classification models.

## 3. License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ESandML/SSL4GIE/blob/main/LICENSE) file. Please however note that when using MoCo v3 or MAE, code from the [MoCo v3](https://github.com/facebookresearch/moco-v3) or [MAE](https://github.com/facebookresearch/mae) repositories is used, which are both released under CC BY-NC 4.0 licenses, and any results from such runs are therefore covered by a CC BY-NC 4.0 license.

## 4. Citation

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

## 5. Commercial use

We allow commercial use of this work, as permitted by the [LICENSE](https://github.com/ESandML/SSL4GIE/blob/main/LICENSE). However, where possible, please inform us of this use for the facilitation of our impact case studies.

## 6. Acknowledgements

This work was supported by the Science and Technology Facilities Council [grant number ST/S005404/1].

This work makes use of:
+ The [Hyperkvasir-unlabelled](https://datasets.simula.no/hyper-kvasir/) dataset.
+ The [MoCo v3](https://github.com/facebookresearch/moco-v3), [Barlow Twins](https://github.com/facebookresearch/barlowtwins), and [MAE](https://github.com/facebookresearch/mae) codebases. In addition to using these codebases for pretraining, as well as the weights provided, we also include the [MoCo v3](https://github.com/facebookresearch/moco-v3) and [MAE](https://github.com/facebookresearch/mae) repositories in `SSL4GIE/Models`, with modifications made for version compatibility.

## 7. Additional information

Links: [AIdDeCo Project](https://www.uclan.ac.uk/research/activity/machine-learning-cancer-detection), [CVML Group](https://www.uclan.ac.uk/research/activity/cvml)

Contact: esanderson4@uclan.ac.uk
