import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

FAST_CATEGORIES = [ 
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": 'A220'},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": 'A321'},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": 'A330'},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": 'A350'},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": 'ARJ21'},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": 'Baseball-Field'},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": 'Basketball-Court'},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": 'Boeing737'},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": 'Boeing747'},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": 'Boeing777'},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": 'Boeing787'},
    {"color": [220, 220, 0], "isthing": 1, "id": 12, "name": 'Bridge'},
    {"color": [175, 116, 175], "isthing": 1, "id": 13, "name": 'Bus'},
    {"color": [250, 0, 30], "isthing": 1, "id": 14, "name": 'C919'},
    {"color": [165, 42, 42], "isthing": 1, "id": 15, "name": 'Cargo-Truck'},
    {"color": [255, 77, 255], "isthing": 1, "id": 16, "name": 'Dry-Cargo-Ship'},
    {"color": [0, 226, 252], "isthing": 1, "id": 17, "name": 'Dump-Truck'},
    {"color": [182, 182, 255], "isthing": 1, "id": 18, "name": 'Engineering-Ship'},
    {"color": [0, 82, 0], "isthing": 1, "id": 19, "name": 'Excavator'},
    {"color": [120, 166, 157], "isthing": 1, "id": 20, "name": 'Fishing-Boat'},
    {"color": [110, 76, 0], "isthing": 1, "id": 21, "name": 'Football-Field'},
    {"color": [174, 57, 255], "isthing": 1, "id": 22, "name": 'Intersection'},
    {"color": [199, 100, 0], "isthing": 1, "id": 23, "name": 'Liquid-Cargo-Ship'},
    {"color": [72, 0, 118], "isthing": 1, "id": 24, "name": 'Motorboat'},
    {"color": [255, 179, 240], "isthing": 1, "id": 25, "name": 'other-airplane'},
    {"color": [0, 125, 92], "isthing": 1, "id": 26, "name": 'other-ship'},
    {"color": [209, 0, 151], "isthing": 1, "id": 27, "name": 'other-vehicle'},
    {"color": [188, 208, 182], "isthing": 1, "id": 28, "name": 'Passenger-Ship'},
    {"color": [0, 220, 176], "isthing": 1, "id": 29, "name": 'Roundabout'},
    {"color": [255, 99, 164], "isthing": 1, "id": 30, "name": 'Small-Car'},
    {"color": [92, 0, 73], "isthing": 1, "id": 31, "name": 'Tennis-Court'},
    {"color": [133, 129, 255], "isthing": 1, "id": 32, "name": 'Tractor'},
    {"color": [78, 180, 255], "isthing": 1, "id": 33, "name": 'Trailer'},
    {"color": [0, 228, 0], "isthing": 1, "id": 34, "name": 'Truck-Tractor'},
    {"color": [174, 255, 243], "isthing": 1, "id": 35, "name": 'Tugboat'},
    {"color": [45, 89, 255], "isthing": 1, "id": 36, "name": 'Van'},
    {"color": [134, 134, 103], "isthing": 1, "id": 37, "name": 'Warship'},
]

def _get_fast_instances_meta():
    thing_ids = [k["id"] for k in FAST_CATEGORIES]
    assert len(thing_ids) == 37, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in FAST_CATEGORIES]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }
    return ret

def register_fast_instance(root):
    metadata = _get_fast_instances_meta()
    name = "fast_val_instance"
    image_root = "/data/SCORE/FAST/val/images"
    json_file = "/data/SCORE/FAST/val/instances_val.json"

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
register_fast_instance(_root)