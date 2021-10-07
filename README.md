# DeepCAD-Breast-Cancer

## Install Dependencies
### Install python packages
```posh
pip install pandas
```


### Install mmdetection
```posh
conda create -n mmdet
conda activate mmdet

nvcc -V
gcc --version

pip install -U torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install Pillow==7.0.0
```


### Install cocoapi
```posh
cd ..

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make -j20 install

* Add this line to ~/.bashrc 'export PYTHONPATH="${PYTHONPATH}:/home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/libs/cocoapi/PythonAPI/"'

python setup.py build_ext --inplace

* For MatlabAPI, follow the link in the file README.txt of the forked COCOAPI repository
```


### Install Caffe & py-faster-rcnn (optional, for running frcnn-cad only)
Install `Caffe` by following the instructions here: http://caffe.berkeleyvision.org/install_apt.html

(If you want to compile `Caffe` without root privileges: https://infinitescript.com/2019/07/compile-caffe-without-root-privileges/)

Install `py-faster-rcnn` by following the README.md in this repo: github.com/rbgirshick/py-faster-rcnn


## How to run
### Training MMDet Detection Model
First, change to directory `libs/mmdetection` by running: `cd libs/mmdetection`

Create a config file by following the instructions in the MMDet repository. For example, I have created a config file `faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py` in `configs/cbis_ddsm_mass/`. To train MMDet model using this config file, use this command:
```posh
sh tools/dist_train.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py 4
```

### Testing MMDet Detection Model
Create a bash script test.sh with the following commands:
```posh
save_root=/path/to/mmdet/experiment/result/directory

# Visualize Train/Val Loss Curves
python tools/analysis_tools/analyze_logs.py plot_curve ${save_root}/*.json --keys loss_cls loss_bbox --legend loss_cls loss_bbox --out ${save_root}/losses.pdf

# Get Bounding Boxes predictions for test set (either in `.pkl` or `.json` format)
sh  tools/dist_test.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
    ${save_root}/best_bbox_mAP_epoch_11.pth 1 \
    --out ${save_root}/result.pkl

sh  tools/dist_test.sh configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
    ${save_root}/best_bbox_mAP_epoch_11.pth 1 \
    --format-only --eval-options "jsonfile_prefix=${save_root}/result"

# To visualize images with the highest and lowest detection scores. This is to debug your model.
python  tools/analysis_tools/analyze_results.py \
        configs/cbis_ddsm_mass/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm.py \
        ${save_root}/result.pkl \
        ${save_root}/results

# Plot ROC curve and PR curve for evaluation
mass_test_gt="/home/hqvo2/Datasets/processed_data2/mass/test/annotation_coco_with_classes.json"
python plot_eval_curve.py -gt ${mass_test_gt} -p ${save_root}/result.bbox.json -bb all -s ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/
```
