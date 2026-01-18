import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

SOTA_CATEGORIES = [ 
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": 'large-vehicle'},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": 'swimming-pool'},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": 'helicopter'},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": 'bridge'},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": 'plane'},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": 'ship'},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": 'soccer-ball-field'},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": 'basketball-court'},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": 'ground-track-field'},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": 'small-vehicle'},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": 'baseball-diamond'},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": 'tennis-court'},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": 'roundabout'},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": 'storage-tank'},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": 'harbor'},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": 'container-crane'},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": 'airport'},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": 'helipad'},
]

def _get_sota_instances_meta():
    thing_ids = [k["id"] for k in SOTA_CATEGORIES]
    assert len(thing_ids) == 18, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOTA_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def register_sota_instance(root):
    metadata = _get_sota_instances_meta()
    name = "sota_val_instance"
    image_root = "/data/SCORE/SOTA/val/images"
    json_file = "/data/SCORE/SOTA/val/instances_val.json"

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
register_sota_instance(_root)