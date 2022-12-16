# MVSFormer
Codes of MVSFormer: Multi-View Stereo by Learning Robust Image Features and Temperature-based Depth

[arxiv paper](https://arxiv.org/abs/2208.02541)

- [x] Releasing codes of training and testing
- [x] Adding dynamic pointcloud fusion for T&T
- [ ] Releasing pre-trained models

## Installation

```
git clone https://github.com/ewrfcas/MVSFormer.git
cd MVSFormer
pip install -r requirements.txt
```

We also highly recommend to install fusibile from (https://github.com/YoYo000/fusibile) for the depth fusion.

```
git clone https://github.com/YoYo000/fusibile.git
cd fusibile
cmake .
make
```

**Tips:** Your should revise CUDA_NVCC_FLAGS in CMakeLists.txt according the gpu device you used. We set ```-gencode arch=compute_70,code=sm_70``` instead of ```-gencode arch=compute_60,code=sm_60``` with V100 GPUs.

## Datasets

### DTU

1. Download preprocessed poses from [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view), 
and depth from [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip).
2. We also need original rectified images from [the official website](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip).
3. DTU testing set can be downloaded from [MVSNet](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_).

```
dtu_training
 ├── Cameras
 ├── Depths
 ├── Depths_raw
 └── DTU_origin/Rectified (downloaded from the official website with origin image size)
```

### BlendedMVS

Download high-resolution images from [BlendedMVS](https://onedrive.live.com/?authkey=%21ADb9OciQ4zKwJ%5Fw&id=35CFA9803D6F030F%21123&cid=35CFA9803D6F030F)

```
BlendedMVS_raw
 ├── 57f8d9bbe73f6760f10e916a
 .   └── 57f8d9bbe73f6760f10e916a
 .       └── 57f8d9bbe73f6760f10e916a
 .           ├── blended_images
             ├── cams
             └── rendered_depth_maps
```

### Tank-and-Temples (T&T)
Download preprocessed [T&T](https://drive.google.com/file/d/1gAfmeoGNEFl9dL4QcAU4kF0BAyTd-r8Z/view).
Note that users should use the short depth range of cameras Run the evaluation script to produce the point clouds.

```
tankandtemples
 ├── advanced
 │  ├── Auditorium
 │  ├── Ballroom
 │  ├── ...
 │  └── Temple
 └── intermediate
        ├── Family
        ├── Francis
        ├── ...
        ├── Train
        └── short_range_cameras
```

## Training

Training MVSFormer (Twins-based) on DTU with 2 32GB V100 GPUs cost 2 days.
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/config_mvsformer.json \
                                         --exp_name MVSFormer \
                                         --data_path ${YOUR_DTU_PATH} \
                                         --DDP
```
MVSFormer-P (frozen DINO-based).
```
                                         
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/config_mvsformer-p.json \
                                         --exp_name MVSFormer-p \
                                         --data_path ${YOUR_DTU_PATH} \
                                         --DDP
```

We should finetune our model based on BlendedMVS before the testing on T&T.
```
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/config_mvsformer_blendmvs.json \
                                         --exp_name MVSFormer-blendedmvs \
                                         --data_path ${YOUR_BLENDEMVS_PATH} \
                                         --dtu_model_path ${YOUR_DTU_MODEL_PATH} \
                                         --DDP
```

## Test

Our pretrained models will be released soon!

For testing on DTU:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset dtu --batch_size 1 \
                                       --testpath ${dtu_test_path} \
                                       --testlist ./lists/dtu/test.txt \
                                       --resume ${MODEL_WEIGHT_PATH} \
                                       --outdir ${OUTPUT_DIR} \
                                       --fusibile_exe_path ./fusibile/fusibile \
                                       --interval_scale 1.06 --num_view 5 \
                                       --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma \
                                       --disp_threshold 0.1 --num_consistent 2 --prob_threshold 0.5,0.5,0.5,0.5 \
                                       --combine_conf \
                                       --tmps 5.0,5.0,5.0,1.0
```

For testing on T&T
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset tt --batch_size 1 \
                                      --testpath ${tt_test_path}/intermediate(or advanced) \
                                      --testlist ./lists/tanksandtemples/intermediate.txt(or advanced.txt)
                                      --resume ${MODEL_WEIGHT_PATH} \
                                      --outdir ${OUTPUT_DIR} \ 
                                      --interval_scale 1.0 --num_view 20 --numdepth 256 \
                                      --max_h 1088 --max_w 1920 --filter_method dpcd \
                                      --prob_threshold 0.5,0.5,0.5,0.5 \
                                      --use_short_range --combine_conf --tmps 5.0,5.0,5.0,1.0
```

Our codes are partially based on [CDS-MVSNet](https://github.com/TruongKhang/cds-mvsnet).
