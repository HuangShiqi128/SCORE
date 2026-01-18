import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

iSAID_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "ship"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "storage_tank"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "baseball_diamond"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "tennis_court"},
    {"color": [0, 0, 70], "isthing": 1, "id": 5, "name": "basketball_court"},
    {"color": [0, 0, 192], "isthing": 1, "id": 6, "name": "Ground_Track_Field"},
    {"color": [250, 0, 30], "isthing": 1, "id": 7, "name": "Bridge"},
    {"color": [165, 42, 42], "isthing": 1, "id": 8, "name": "Large_Vehicle"},
    {"color": [182, 182, 255], "isthing": 1, "id": 9, "name": "Small_Vehicle"},
    {"color": [0, 82, 0], "isthing": 1, "id": 10, "name": "Helicopter"},
    {"color": [199, 100, 0], "isthing": 1, "id": 11, "name": "Swimming_pool"},
    {"color": [72, 0, 118], "isthing": 1, "id": 12, "name": "Roundabout"},
    {"color": [255, 179, 240], "isthing": 1, "id": 13, "name": "Soccer_ball_field"},
    {"color": [209, 0, 151], "isthing": 1, "id": 14, "name": "plane"},
    {"color": [92, 0, 73], "isthing": 1, "id": 15, "name": "Harbor"},
]

def _get_isaid_instances_meta():
    thing_ids = [k["id"] for k in iSAID_CATEGORIES]
    assert len(thing_ids) == 15, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in iSAID_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def register_isaid_instance(root):
    metadata = _get_isaid_instances_meta()
    name = "isaid_train_instance"
    #TODO
    image_root = "/data/SCORE/iSAID/train/images"
    json_file = "/data/SCORE/iSAID/train/instances_train.json"

    json_file = os.path.join(root, json_file) if "://" not in json_file else json_file
    image_root = os.path.join(root, image_root)
    DatasetCatalog.register(name, lambda: load_coco_json(json_file,
                                                                 image_root,
                                                                 name))

    MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            evaluator_type="coco", **metadata)
    
def register_isaid_val_instance(root):
    metadata = _get_isaid_instances_meta()
    name = "isaid_val_instance"
    image_root = "/data/SCORE/iSAID/val/images"
    json_file = "/data/SCORE/iSAID/val/instances_val.json"

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
register_isaid_instance(_root)
register_isaid_val_instance(_root)