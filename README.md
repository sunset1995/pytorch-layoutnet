# pytorch-layoutnet
This is an unofficial implementation of [**LayoutNet**: Reconstructing the 3D Room Layout from a Single RGB Image](https://github.com/zouchuhang/LayoutNet). [Official](https://github.com/zouchuhang/LayoutNet) layout dataset are all converted to `.png` and pretrained models are converted to pytorch `state-dict`.  
Currently only joint bounday branch and corner branch are implemented, 3d layout regressor is ignored as it shows little improvement and is not relate to inference.

## Requirements
- Python 3
- pytorch>=0.4.1
- numpy
- scipy
- Pillow
- torchfile (for converting official data and pretrained weight)

## Preparation
- Download offical [data](https://github.com/zouchuhang/LayoutNet#data) and [pretrained model](https://github.com/zouchuhang/LayoutNet#pretrained-model) as below
```
/pytorch-layoutnet 
  /data
  | /origin
  |   /data  (download and extract from official)
  |   /gt    (download and extract from official)
  /ckpt
    /panofull_*_pretrained.t7  (download and extract from official)
```
- Execute `python torch2pytorch_data.py` to convert `data/origin/**/*` to `data/train`, `data/valid` and `data/test` for pytorch data loader. Under these folder `img/` contains all raw rgb `.png` while `line/`, `edge/`, `cor/` contain preprocessed edge detection result, ground truth boundary and ground truth corner respectively.
- Use `torch2pytorch_pretrained_weight.py` to convert official pretrained pano model to `encoder`, `edg_decoder`, `cor_decoder` pytorch `state_dict` (see `python torch2pytorch_pretrained_weight.py -h` for more detailed). example:
  - `python torch2pytorch_pretrained_weight.py --torch_pretrained ckpt/panofull_joint_box_pretrained.t7 --encoder ckpt/pre_full_encoder.pth --edg_decoder ckpt/pre_full_edg_decoder.pth --cor_decoder ckpt/pre_full_cor_decoder.pth` to convert pretrained pano full models into 3 `state_dict`s with 3d layout regressor branch ignored.

## Visualization

## Training and Evaluation
