concat_gt="/home/hqvo2/Projects/Breast_Cancer/data/methodist_data/train_test_folds/12_08_2021/test_folds/concat_gt.json"
concat_pred="/home/hqvo2/Projects/Breast_Cancer/experiments/methodist_data_detection/mass/12_08_2021/faster_rcnn_r50_caffe_fpn_mstrain_1x_ddsm"
python plot_eval_curve.py -gt ${concat_gt} -p ${concat_pred}/concat_pred.json -bb all -s ${concat_pred}/
