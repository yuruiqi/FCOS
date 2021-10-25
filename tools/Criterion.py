import json
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from sklearn.metrics import f1_score, roc_auc_score
from fcos_core.config.paths_catalog import DatasetCatalog

from Data.Preprocess import join_path


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.

    box1: [b1_y1, b1_x1, b1_y2, b1_x2]
    box2: [b2_y1, b2_x1, b2_y2, b2_x2]
    return: float
    """
    # Compute intersection
    b1_y1, b1_x1, b1_h, b1_w = box1
    b2_y1, b2_x1, b2_h, b2_w = box2
    b1_y2, b1_x2 = b1_y1+b1_h, b1_x1+b1_w
    b2_y2, b2_x2 = b2_y1+b2_h, b2_x1+b2_w

    y1 = max(b1_y1, b2_y1)
    x1 = max(b1_x1, b2_x1)
    y2 = min(b1_y2, b2_y2)
    x2 = min(b1_x2, b2_x2)
    intersection = max(x2 - x1, 0)*max(y2 - y1, 0)

    # Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    return iou


def iou_matrix(boxes1, boxes2):
    n = len(boxes1)
    m = len(boxes2)
    matrix = np.zeros([n, m])

    for i in range(n):
        for j in range(m):
            matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    return matrix


def get_cls_results(gt_boxes, pred_boxes, iou_th):
    gt = [x['category_id'] for x in gt_boxes]
    gt_bbox = [x['bbox'] for x in gt_boxes]

    pred = [x['category_id'] for x in pred_boxes]
    pred_bbox = [x['bbox'] for x in pred_boxes]
    pred_score = [x['score'] for x in pred_boxes]

    matrix = iou_matrix(pred_bbox, gt_bbox)  # (n_pred, n_label)

    out_label = []
    # out_score = []
    out_pred = []

    # tp
    tp_index = np.nonzero(matrix>iou_th)
    for i in range(tp_index[0].size):
        pred_index = tp_index[0][i]
        label_index = tp_index[1][i]

        # best pred in duplicated preds
        if matrix[pred_index, label_index] == np.max(matrix[:,label_index]).item():
            out_label.append(gt[label_index])
            # out_score.append(pred_score[tp_index[0][i]])
            out_pred.append(pred[pred_index])
        # duplicate preds, taken as fp
        else:
            out_label.append(0)
            # out_score.append(pred_score[fp_index[i]])
            out_pred.append(pred[pred_index])


    # fp
    fp_index = np.nonzero(np.max(matrix, axis=1)<=iou_th)
    for i in range(fp_index[0].size):
        out_label.append(0)
        # out_score.append(pred_score[fp_index[i]])
        out_pred.append(pred[fp_index[0][i]])

    # fn
    if len(pred)>0:
        fn_index = np.nonzero(np.max(matrix, axis=0)<=iou_th)
        for i in range(fn_index[0].size):
            out_label.append(gt[fn_index[0][i]])
            # out_score.append()
            out_pred.append(0)
    else:
        out_label.extend(gt)
        # out_score.append()
        out_pred.extend([0,]*len(gt))

    return out_label, out_pred


def compute_auc(pred, label, negative_th):
    pred = pred/4
    label = np.where(label>negative_th, 1, 0)
    auc = roc_auc_score(label, pred)
    return auc


def confusion_metrix(pred, label, negative_th):
    pred = np.array(pred)
    label = np.array(label)
    f1 = f1_score(label, pred, average='macro')
    auc = compute_auc(pred, label, negative_th)
    acc_allclass = np.count_nonzero(pred==label)/pred.size
    acc_ap1 = np.count_nonzero(np.abs(pred-label)<2)/pred.size

    pred = np.where(pred>negative_th, 1, 0)
    label = np.where(label>negative_th, 1, 0)

    tp = np.count_nonzero(pred*label)
    fp = np.count_nonzero(pred*(1-label))
    fn = np.count_nonzero((1-pred)*label)
    tn = np.count_nonzero((1-pred)*(1-label))

    sen = tp/(tp+fn)
    ppv = tp/(tp+fp)
    spe = tn/(tn+fp)
    acc = (tp+tn)/pred.size
    print(f'sen:{round(sen,2)}, '
          f'ppv:{round(ppv,2)}, '
          f'spe:{round(spe,2)}, '
          f'acc:{round(acc,2)}, '
          f'f1:{round(f1,2)}, '
          f'auc:{round(auc,2)}, '
          f'acc:{round(acc_allclass,2)}, '
          f'acc+-1:{round(acc_ap1,2)}')


def assess(datasets, output_dir, negative_th=1, iou_th=0.3):
    for dataset in datasets:
        with open(join_path(output_dir, 'inference', dataset, 'bbox.json')) as f:
            bbox_json = json.load(f)

        data_args = DatasetCatalog().get(dataset)['args']
        img_dir = data_args['root']
        ann_f = data_args['ann_file']
        with open(ann_f) as f:
            ann_json = json.load(f)

        label = []
        pred = []
        for img_dict in ann_json['images']:
            img_id = img_dict['id']
            img_name = img_dict['file_name']
            gt_box = [x for x in ann_json['annotations'] if x['image_id'] == img_id]
            pred_box = [x for x in bbox_json if x['image_id'] == img_id]

            case_label, case_pred = get_cls_results(gt_box, pred_box, iou_th)
            label.extend(case_label)
            pred.extend(case_pred)

        confusion_metrix(pred, label, negative_th)


if __name__ == '__main__':
    negative_th = 1
    iou_th = 0.3

    assess(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/fcos_R_50_FPN_1x/',
           negative_th, iou_th)
    assess(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/fcos_R_50_FPN_1x_allclsnms/',
           negative_th, iou_th)
    assess(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/udm/',
           negative_th, iou_th)
    assess(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/udm2/',
           negative_th, iou_th)
    assess(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/udm2_cls1th/',
           negative_th, iou_th)

