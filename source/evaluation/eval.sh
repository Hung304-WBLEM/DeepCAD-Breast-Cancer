cd /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/libs/Object-Detection-Metrics

# declare -a configs=("cascade_mask_rcnn_r50_caffe_fpn_1x_ddsm" "detectors_htc_r50_1x_ddsm" "mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_ddsm" "detection_groundtruth" "faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm" "retinanet_r50_nasfpn_crop640_50e_ddsm")
# 
# 
# for config in ${configs[@]}; do
# 
#     mkdir -p /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/$config/plots/train
#     python pascalvoc.py \
#            -gt /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/detection_groundtruth/train \
#            -det /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/$config/detections/train \
#            -gtformat xywh \
#            -detformat xyrb \
#            -sp /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/$config/plots/train \
#            -np
# 
#     mkdir -p /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/$config/plots/test
#     python pascalvoc.py \
#            -gt /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/detection_groundtruth/test \
#            -det /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/$config/detections/test \
#            -gtformat xywh \
#            -detformat xyrb \
#            -sp /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/$config/plots/test \
#            -np
# 
# done

# mkdir -p /home/cougarnet.uh.edu/hqvo1/Projects/Breast_Cancer/experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/plots/train
# python pascalvoc.py \
#        -gt /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/detection_groundtruth/train \
#        -det /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/detections/train \
#        -gtformat xywh \
#        -detformat xyrb \
#        -sp /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/plots/train \
#        -np
# 
# mkdir -p /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/plots/test
# python pascalvoc.py \
#        -gt /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/detection_groundtruth/test \
#        -det /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/detections/test \
#        -gtformat xywh \
#        -detformat xyrb \
#        -sp /home/cougarnet.uh.edu/hqvo2/Projects/Breast_Cancer/experiments/mmdet_processed_data/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/plots/test \
#        -np
