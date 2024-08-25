# PSNet-pytorch

### Prerequisites
Ubuntu 18.04\
Python==3.8.3\
Torch==1.8.2+cu111\
Torchvision==0.9.2+cu111\


### Dataset
For all datasets, they should be organized in below's fashion:
```
|__dataset_name
   |__train
      |__images xxx.jpg ...
      |__masks xxx.jpg ...
   |__test
      |__images xxx.jpg ...
      |__masks xxx.jpg ...
```
For training, put your dataset folder under:
```
dataset/
```

### Train & Test
**Make sure you have enough GPU RAM**.\
With default setting (batchsize=8), 10GB RAM is required, but you can always reduce the batchsize to fit your hardware.

Please download the ISIC dataset from [[EGE-UNet]](https://github.com/JCruan519/EGE-UNet)

Download the pre-trained PVT-V2 model 'PVT-V2-B2' from [[PVT-V2]](https://github.com/whai362/PVT/tree/v2/classification) and put it in the '/model/pretrain' folder

Default values in option.py are already set to the same configuration as our paper, so \
after setting the ```--dataset_root``` flag in **option.py**, to train the model (default dataset: ISIC2018), simply:
```
python main.py 
```
to test the model located in the **ckpt** folder (default dataset: ISIC2018), simply:
```
python main.py --test_only --pretrain "ckpt/best_for_isic18.pt" --save_result --save_msg "abc"
```
If you want to train/test with different settings, please refer to **option.py** for more control options.\
Currently only support training on single GPU.

### Pretrain Model & Pre-calculated Saliency Map
Pretrained models can be downloaded from [[Google Drive]](https://drive.google.com/drive/folders/1l5HJWXid7JETz4gxuireVXgoMjm14jEX?usp=sharing)


### Acknowledgments
We thank the authors of [[EGE-UNet]](https://github.com/JCruan519/EGE-UNet) and [[SelfReformer]](https://github.com/BarCodeReader/SelfReformer/tree/main) for their open-source codes.

