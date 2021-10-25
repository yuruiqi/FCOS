import json
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from fcos_core.config.paths_catalog import DatasetCatalog

from Data.Preprocess import join_path


def draw_box(box_dict, colour='green'):
    for box in box_dict:
        y1, x1, h, w = box['bbox']
        if 'score' in box.keys():
            score = box['score']
        else:
            score = None
        category = box['category_id']

        plt_box = plt.Rectangle((y1, x1), h, w, color=colour, fill=False, linewidth=2)
        plt.gca().add_patch(plt_box)

        if score is not None:
            box_tag = '{} {:.2f}'.format(category, score)
        else:
            box_tag = '{}'.format(category)
        plt.gca().text(y1, x1 - 1, box_tag, color='white', bbox={'facecolor': colour, 'alpha': 0.5})


def draw(datasets, output_dir):
    # datasets = cfg['DATASETS']['TEST']

    for dataset in datasets:
        with open(join_path(output_dir, 'inference', dataset, 'bbox.json')) as f:
            bbox_json = json.load(f)

        data_args = DatasetCatalog().get(dataset)['args']
        img_dir = data_args['root']
        ann_f = data_args['ann_file']
        with open(ann_f) as f:
            ann_json = json.load(f)

        save_dir = join_path(output_dir, 'inference', dataset, 'figures')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for img_dict in ann_json['images']:
            img_id = img_dict['id']
            img_name = img_dict['file_name']
            gt = [x for x in ann_json['annotations'] if x['image_id'] == img_id]
            pred = [x for x in bbox_json if x['image_id'] == img_id]

            img = plt.imread(join_path(img_dir, img_name))
            plt.figure()
            plt.imshow(img)
            draw_box(gt)
            draw_box(pred, colour='blue')

            plt.savefig(join_path(save_dir, img_name))


if __name__ == '__main__':
    # with open('/homes/rqyu/PycharmProjects/FCOS/configs/fcos/fcos_R_50_FPN_1x.yaml') as f:
    #     cfg = yaml.load(f)

    # draw(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/fcos_R_50_FPN_1x/')
    # draw(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/fcos_R_50_FPN_1x_allclsnms/')
    draw(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/udm/')
    # draw(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/udm2/')
    # draw(("jsph_test_coco_style",), '/homes/rqyu/PycharmProjects/FCOS/training_dir/udm2_cls1th/')

