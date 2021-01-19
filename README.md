# DeepCAD-Breast-Cancer

## Install Dependencies
### Install mmdetection
```
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
```
cd ..

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make -j20 install

* Add this line to ~/.bashrc 'export PYTHONPATH="${PYTHONPATH}:/home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/libs/cocoapi/PythonAPI/"'

python setup.py build_ext --inplace

* For MatlabAPI, follow the link in the file README.txt of the forked COCOAPI repository
```

### Install python packages
```
pip install pandas
```
