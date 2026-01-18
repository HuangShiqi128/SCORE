import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
import os
from detectron2.data.datasets.coco import load_coco_json
import torch

# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
NWPU_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "airplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "ship"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "storage_tank"},
    {"color": [255, 179, 240], "isthing": 1, "id": 4, "name": "baseball_diamond"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "tennis_court"},
    {"color": [92, 0, 73], "isthing": 1, "id": 6, "name": "basketball_court"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "ground_track_field"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "harbor"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "bridge"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "vehicle"}
]

def _get_nwpu_instances_meta():
    thing_ids = [k["id"] for k in NWPU_CATEGORIES]
    assert len(thing_ids) == 10, len(thing_ids)
    # Mapping from the incontiguous ADE category id to an id in [0, 99]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in NWPU_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret


def register_nwpu_instance(root):
    metadata = _get_nwpu_instances_meta()
    name = "nwpu_val_instance"

    image_root = "/data/SCORE/NWPU/image_patches"
    json_file = "/data/SCORE/NWPU/instances_val.json"

    json_file = os.path.join(root, json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)
    DatasetCatalog.register(name, lambda: load_coco_json(json_file,
                                                                 image_root,
                                                                 name))

    MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            evaluator_type="coco", **metadata)
    

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_nwpu_instance(_root)







