# python plot_eval_curve.py -gt /home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/annotation_coco_with_classes.json -p /home/hqvo2/Projects/Breast_Cancer/libs/mmdetection/res_ohem.bbox.json -bb all -s ./
# python plot_eval_curve.py -gt /home/hqvo2/Projects/Breast_Cancer/data/processed_data/mass/test/annotation_coco_with_classes.json -p /home/hqvo2/Projects/Breast_Cancer/libs/mmdetection/res_ohem.bbox.json -bb opi -s ./

mass_test_gt="/home/hqvo2/Datasets/processed_data2/mass/test/annotation_coco_with_classes.json"
calc_test_gt="/home/hqvo2/Datasets/processed_data2/calc/test/annotation_coco_with_classes.json"

mass_detection_root="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/mass"
calc_detection_root="/home/hqvo2/Projects/Breast_Cancer/experiments/cbis_ddsm_detection/calc"

# python plot_eval_curve.py -gt ${mass_test_gt} -p ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes.bbox.json -bb all -s ${mass_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/
python plot_eval_curve.py -gt ${calc_test_gt} -p ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/test_bboxes.bbox.json -bb all -s ${calc_detection_root}/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm/
