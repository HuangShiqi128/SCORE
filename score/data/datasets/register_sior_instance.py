import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
import os
from detectron2.data.datasets.coco import load_coco_json
import torch

# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
SIOR_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "airport"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseball field"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "basketball court"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "bridge"},
    {"color": [0, 0, 192], "isthing": 1, "id": 6, "name": "chimney"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "expressway-service-area"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "expressway-toll-station"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "dam"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "golffield"},
    {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "ground track field"},
    {"color": [72, 0, 118], "isthing": 1, "id": 12, "name": "harbor"},
    {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "overpass"},
    {"color": [209, 0, 151], "isthing": 1, "id": 14, "name": "ship"},
    {"color": [92, 0, 73], "isthing": 1, "id": 15, "name": "stadium"},
    {"color": [0, 228, 0], "isthing": 1, "id": 16, "name": "storage tank"},
    {"color": [145, 148, 174], "isthing": 1, "id": 17, "name": "tennis court"},
    {"color": [197, 226, 255], "isthing": 1, "id": 18, "name": "train station"},
    {"color": [9, 80, 61], "isthing": 1, "id": 19, "name": "vehicle"},
    {"color": [84, 105, 51], "isthing": 1, "id": 20, "name": "windmill"}
]

def _get_sior_instances_meta():
    thing_ids = [k["id"] for k in SIOR_CATEGORIES]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SIOR_CATEGORIES]
    thing_colors = [k["color"] for k in SIOR_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_sior_train_instance(root):
    metadata = _get_sior_instances_meta()
    name = "sior_train_instance"
    image_root = "/data/SCORE/SIOR/train/images"
    json_file = "/data/SCORE/SIOR/train/instances_train.json"

    json_file = os.path.join(root,
                             json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file,
                                                             image_root,
                                                             name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco", **metadata)

def register_sior_val_instance(root):
    metadata = _get_sior_instances_meta()
    name = "sior_val_instance"
    image_root = "/data/SCORE/SIOR/val/images"
    json_file = "/data/SCORE/SIOR/val/instances_val.json"

    json_file = os.path.join(root,
                             json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file,
                                                             image_root,
                                                             name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco", **metadata)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_sior_train_instance(_root)
register_sior_val_instance(_root)








