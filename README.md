# Robust Multi-Modality Multi-Object Tracking

This is the project page for our ICCV2019 paper: **Robust Multi-Modality Multi-Object Tracking**.

**Authors**: [Wenwei Zhang](http://zhangwenwei.cn), [Hui Zhou](https://scholar.google.com/citations?user=i35tdbMAAAAJ&hl=zh-CN), [Shuyang Sun](https://kevin-ssy.github.io/), [Zhe Wang](https://wang-zhe.me/), [Jianping Shi](http://shijianping.me/), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/)

[[ArXiv]](https://arxiv.org/abs/1909.03850)&nbsp;  [[Project Page]](#)&nbsp;  [[Poster]](http://zhangwenwei.cn/files/mmMOT_poster_final.pdf)

## Introduction

In this work, we design a generic sensor-agnostic multi-modality MOT framework (mmMOT), where each modality (i.e., sensors) is capable of performing its role independently to preserve reliability, and further improving its accuracy through a novel multi-modality fusion module. Our mmMOT can be trained in an end-to-end manner, enables joint optimization for the base feature extractor of each modality and an adjacency estimator for cross modality. Our mmMOT also makes the first attempt to encode deep representation of point cloud in data association process in MOT. 

For more details, please refer our [paper](https://arxiv.org/abs/1909.03850).

## Install

This project is based on pytorch>=1.0, you can install it following the [official guide](https://pytorch.org/get-started/locally/).

We recommand you to build a new conda environment to run the projects as follows:
```bash
conda create -n mmmot python=3.7 cython
conda activate mmmot
conda install pytorch torchvision -c pytorch
conda install numba
```

Then install packages from pip:
```bash
pip install -r requirements.txt
```

You can also follow the guide to install [SECOND](https://github.com/traveller59/second.pytorch), we use the same environment as that for SECOND.


## Usage

We provide several configs and scripts in the `experiments` directory. 

To evaluate the pretrained models or the reimplemented models you can run command
```bash
python -u eval_seq.py --config ${work_path}/config.yaml \
--load-path=${work_path}/${model} \
--result-path=${work_path}/results \
--result_sha=eval
```
The `--result_sha` option is used to distinguish different evaluation attempts.
You can also simply run command like
```
sh ./experiments/pp_pv_40e_mul_A/eval.sh ${partition}
```

To train the model on your own, you can run command
```
python -u main.py --config ${work_path}/config.yaml \
--result-path=${work_path}/results 
```
You can also simply run command like
```
sh ./experiments/pp_pv_40e_mul_A/train.sh ${partition}
```

**Note:** Both the train and eval scripts use srun as default, you can just comment them if you do not use srun.


## Pretrain Model

We provide four models in the [google drive](https://drive.google.com/open?id=1IJ6rWSJw-BExQP-N25RNmQzUeTYSmwj6). 
The corresponding configs can be found in the `experiments` directory.

Following the usage you can directly inference the model and get results as follows:


|    Name    |  Method  | MOTA |
| :-----------: | :-----: | :--: |
|pp_pv_40e_mul_A|Fusion Module A| 77.57|
|pp_pv_40e_mul_B|Fusion Module B| 77.62|
|pp_pv_40e_mul_C|Fusion Module C| 78.18|
|pp_pv_40e_dualadd_subabs_C|Fusion Module C++| 80.08|

The results of Fusion Module A,B and C are the same as those in the Table 1 of the paper.
The Fusion Module C++ indicates that it uses `absolute subtraction` and `softmax with addition` to improve the results, and has the same MOTA as that in the last row of Table 3 of the [paper](https://arxiv.org/abs/1909.03850).


## Data

Currently it supports [PointPillar](https://github.com/nutonomy/second.pytorch)/[SECOND](https://github.com/traveller59/second.pytorch) detector, and also support [RRC-Net](https://github.com/xiaohaoChen/rrc_detection) detector.

In the [paper](https://arxiv.org/abs/1909.03850), we train a PointPillars model to obtain the train/val detection results for ablation study, using the [official codebase](https://github.com/nutonomy/second.pytorch). The detection data are provided in the [google drive](https://drive.google.com/open?id=1IJ6rWSJw-BExQP-N25RNmQzUeTYSmwj6). Once you download the two pkl files, put them in the `data` directory.

We also provide the data split used in our paper in the `data` directory. You need to download and unzip the data from the [KITTI Tracking Benchmark](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) and put them in the `kitti_t_o` directory or any path you like.
Do remember to change the path in the configs.

The RRC detection are obtained from the [link](https://drive.google.com/file/d/1ZR1qEf2qjQYA9zALLl-ZXuWhqG9lxzsM/view) provided by [MOTBeyondPixels](https://github.com/JunaidCS032/MOTBeyondPixels). We use RRC detection for the [KITTI Tracking Benchmark](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).


## Citation

If you use this codebase or model in your research, please cite:
```
@InProceedings{mmMOT_2019_ICCV,
    author = {Zhang, Wenwei and Zhou, Hui and Sun, Shuyang, and Wang, Zhe and Shi, Jianping and Loy, Chen Change},
    title = {Robust Multi-Modality Multi-Object Tracking},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Acknowledgement

This code benefits a lot from [SECOND](https://github.com/traveller59/second.pytorch) and use the detection results provided by [MOTBeyondPixels](https://github.com/JunaidCS032/MOTBeyondPixels). The GHM loss implementation is from [GHM_Detection](https://github.com/libuyu/GHM_Detection).

