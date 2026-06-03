CLASSES = {'pascal': ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 
                      'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 
                      'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'],
           
           'cityscapes': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                          'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                          'truck', 'bus', 'train', 'motorcycle', 'bicycle'],
           
           'coco': ['void', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
                    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 
                    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 
                    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                    'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 
                    'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 
                    'ceiling-tile', 'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
                    'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 
                    'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 
                    'grass', 'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 
                    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 
                    'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 
                    'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
                    'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other', 'table', 'tent',
                    'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 
                    'wall-panel', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
                    'window-blind', 'window-other', 'wood'],
           
           'endovis2017_parts': ['background', 'shaft', 'wrist', 'clasper'],

           'endovis2017_type': ['background', 'Bipolar Forceps',
                                 'Prograsp Forceps', 'Large Needle Driver',
                                 'Vessel Sealer', 'Grasping Retractor',
                                 'Monopolar Curved Scissors',
                                 'Ultrasound Probe'],

           # Merged vocabulary for Stage-I affinity training only — parts (3) + type (7)
           # = 10 non-bg classes; same images as endovis2017_parts / _type, but each
           # frame's multi-label is built from BOTH masks.  Stage-II keeps parts and
           # type as separate segmentation tasks; both warm-start from this ckpt.
           'endovis2017_merged': ['background',
                                  # parts
                                  'shaft', 'wrist', 'clasper',
                                  # types
                                  'Bipolar Forceps', 'Prograsp Forceps',
                                  'Large Needle Driver', 'Vessel Sealer',
                                  'Grasping Retractor',
                                  'Monopolar Curved Scissors',
                                  'Ultrasound Probe'],

           'endovis2018': ['background-tissue', 'instrument-shaft',
                            'instrument-clasper', 'instrument-wrist',
                            'kidney-parenchyma', 'covered-kidney',
                            'thread', 'clamps', 'suturing-needle',
                            'suction-instrument', 'small-intestine',
                            'ultrasound-probe'],

           # Surg-SegFormer style 7-class "Scene" remap of EndoVis 2018:
           # shaft+clasper+wrist  →  robotic-instrument-part
           # thread / clamps / suction-instrument  →  ignored (255)
           # Use this with endovis2018_scene_processed produced by
           # tools/remap_endovis2018_scene.py.
           'endovis2018_scene': ['background-tissue',
                                  'robotic-instrument-part',
                                  'kidney-parenchyma',
                                  'covered-kidney',
                                  'suturing-needle',
                                  'small-intestine',
                                  'ultrasound-probe'],

           'endoscapes_seg50': ['background', 'cystic_plate', 'calot_triangle',
                                'cystic_artery', 'cystic_duct', 'gallbladder',
                                'tool'],

           # Endoscapes (full): same 7 classes, different unlabeled pool
           'endoscapes':       ['background', 'cystic_plate', 'calot_triangle',
                                'cystic_artery', 'cystic_duct', 'gallbladder',
                                'tool'],

           'needle': ['background', 'class_0', 'class_1'],

           'cholecseg8k': ['background', 'abdominal_wall', 'liver',
                           'gastrointestinal_tract', 'fat', 'grasper',
                           'connective_tissue', 'blood', 'cystic_duct',
                           'l_hook_electrocautery', 'gallbladder',
                           'hepatic_vein', 'liver_ligament'],

           'ade20k': ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
                      'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
                      'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
                      'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
                      'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
                      'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                      'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                      'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
                      'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
                      'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
                      'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
                      'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
                      'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                      'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
                      'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
                      'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
                      'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
                      'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
                      'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
                      'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
                      'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
                      'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
                      'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 
                      'clock', 'flag']
           }
