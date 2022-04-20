import os
import yaml
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


col2anno = {'officehome_train':'concepts65.txt','officehome_test':'concepts65.txt', 'domainnet_train':'concepts345.txt', 'domainnet_test':'concepts345.txt'}
ROOT_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"VisualSearch")

officehome_domain_dic={"Art":0,"Clipart":1,"Product":2,"Real_World":3}
domainnet_domain_dic={"clipart":0,"painting":1,"real":2,"sketch":4}

officehome_classes_number=65
domainnet_classes_number=345

officehome_classes=['Bottle', 'Kettle', 'Sneakers', 'Glasses', 'Pen', 'Batteries', 'Lamp_Shade', 'Desk_Lamp', 'Push_Pin', 'Toys', 'Bike', 'Notebook', 'Ruler', 'Fork', 'Radio', 'Helmet', 'Printer', 'Backpack', 'Scissors', 'Paper_Clip', 'Calendar', 'Knives', 'Monitor', 'TV', 'Telephone', 'ToothBrush', 'Flipflops', 'Refrigerator', 'Mouse', 'Couch', 'Hammer', 'Calculator', 'Clipboards', 'Trash_Can', 'Pencil', 'Oven', 'Fan', 'Spoon', 'Sink', 'Laptop', 'Bucket', 'Computer', 'Candles', 'Eraser', 'Postit_Notes', 'Pan', 'Screwdriver', 'Keyboard', 'Curtains', 'Alarm_Clock', 'Marker', 'Webcam', 'Exit_Sign', 'Chair', 'Mop', 'Soda', 'Bed', 'Folder', 'Flowers', 'Drill', 'Table', 'Speaker', 'Mug', 'File_Cabinet', 'Shelf']
domainnet_classes=['knife', 'crab', 'submarine', 'piano', 'skull', 'light_bulb', 'rhinoceros', 'sock', 'hamburger', 'picture_frame', 'garden_hose', 'beach', 'crown', 'spider', 'mailbox', 'pig', 'tooth', 'headphones', 'bench', 'ear', 'cello', 'boomerang', 'toothpaste', 'paper_clip', 'hospital', 'bandage', 'hat', 'belt', 'van', 'wine_glass', 'bat', 'smiley_face', 'mermaid', 'waterslide', 'microphone', 'carrot', 'toaster', 'birthday_cake', 'motorbike', 'The_Mona_Lisa', 'harp', 'string_bean', 'soccer_ball', 'stethoscope', 'watermelon', 'tiger', 'swing_set', 'pickup_truck', 'sleeping_bag', 'donut', 'screwdriver', 'floor_lamp', 'hand', 'snail', 'pond', 'firetruck', 'golf_club', 'apple', 'swan', 'aircraft_carrier', 'helmet', 'house', 'paintbrush', 'traffic_light', 'barn', 'snake', 'baseball', 'power_outlet', 'crocodile', 'remote_control', 'lobster', 'peanut', 'postcard', 'broccoli', 'bulldozer', 'jail', 'canoe', 'bicycle', 'wheel', 'stop_sign', 'fire_hydrant', 'mountain', 'tennis_racquet', 'train', 'fireplace', 'square', 'arm', 'snorkel', 'flamingo', 'bowtie', 'brain', 'mushroom', 'The_Great_Wall_of_China', 'nose', 'suitcase', 'rabbit', 'sandwich', 'panda', 'mouth', 'streetlight', 'dishwasher', 'palm_tree', 'star', 'giraffe', 'bush', 'lollipop', 'drums', 'hourglass', 'cloud', 'circle', 'finger', 'saw', 'helicopter', 'truck', 'garden', 'stairs', 'bread', 'sword', 'frying_pan', 'triangle', 'toilet', 'lighthouse', 'church', 'ice_cream', 'basketball', 'cup', 'river', 'matches', 'hurricane', 'dumbbell', 'cake', 'sun', 'microwave', 'bridge', 'cow', 'elbow', 'parachute', 'zebra', 'nail', 'duck', 'leaf', 'sink', 'chair', 'map', 'crayon', 'binoculars', 'tent', 'elephant', 'lion', 'banana', 'see_saw', 'pants', 'teddy-bear', 'teapot', 'beard', 'fan', 'bus', 'cell_phone', 'cat', 'door', 'jacket', 'owl', 'knee', 'alarm_clock', 'sweater', 'broom', 'diving_board', 'saxophone', 'lightning', 'rainbow', 'popsicle', 'book', 'cookie', 'toe', 'monkey', 'computer', 'dolphin', 'squiggle', 'backpack', 'school_bus', 'underwear', 'table', 'grass', 'vase', 'marker', 'scorpion', 'television', 'hedgehog', 'washing_machine', 'radio', 'chandelier', 'shoe', 'violin', 'baseball_bat', 'whale', 'feather', 'anvil', 'cooler', 'line', 'toothbrush', 'skateboard', 'calendar', 'bird', 'oven', 'telephone', 'The_Eiffel_Tower', 'diamond', 'rifle', 'keyboard', 'house_plant', 'skyscraper', 'grapes', 'angel', 'snowman', 'cruise_ship', 'hockey_stick', 'bracelet', 'hot_tub', 'animal_migration', 'raccoon', 'potato', 'hammer', 'blackberry', 'frog', 'asparagus', 'mouse', 'squirrel', 'sheep', 'laptop', 'goatee', 'airplane', 'hot_air_balloon', 'zigzag', 'onion', 'windmill', 'kangaroo', 'moustache', 'sailboat', 'octagon', 'spreadsheet', 'castle', 'ant', 'parrot', 'police_car', 'spoon', 'face', 'purse', 'lighter', 'yoga', 'pliers', 'sea_turtle', 'strawberry', 'couch', 'guitar', 'camouflage', 'speedboat', 'lantern', 'peas', 'stove', 'dresser', 'clarinet', 'eyeglasses', 'penguin', 'flower', 'camera', 'hockey_puck', 'dog', 'shark', 'ambulance', 'scissors', 'lipstick', 'paint_can', 'car', 'mosquito', 'rain', 'mug', 'camel', 'axe', 'fence', 'envelope', 'clock', 'bee', 'bathtub', 'key', 'foot', 'coffee_cup', 'horse', 'moon', 'dragon', 'bear', 'roller_coaster', 'calculator', 'cactus', 'drill', 'ocean', 'candle', 'bed', 'trombone', 'bottlecap', 'wine_bottle', 'pillow', 'steak', 'bucket', 'flashlight', 'hexagon', 'pencil', 'shovel', 'trumpet', 'tornado', 'tree', 'megaphone', 'rollerskates', 'snowflake', 'eraser', 'flip_flops', 'pool', 'tractor', 'syringe', 'butterfly', 'shorts', 'octopus', 'ladder', 'stitches', 'stereo', 'basket', 'pear', 'ceiling_fan', 'umbrella', 'flying_saucer', 'pizza', 'campfire', 'rake', 'leg', 'cannon', 'hot_dog', 'pineapple', 'compass', 't-shirt', 'fish', 'blueberry', 'passport', 'fork', 'necklace', 'wristwatch', 'eye']

domainnet_classes_label={0: 'aircraft_carrier', 1: 'airplane', 2: 'alarm_clock', 3: 'ambulance', 4: 'angel', 5: 'animal_migration', 6: 'ant', 7: 'anvil', 8: 'apple', 9: 'arm', 10: 'asparagus', 11: 'axe', 12: 'backpack', 13: 'banana', 14: 'bandage', 15: 'barn', 16: 'baseball', 17: 'baseball_bat', 18: 'basket', 19: 'basketball', 20: 'bat', 21: 'bathtub', 22: 'beach', 23: 'bear', 24: 'beard', 25: 'bed', 26: 'bee', 27: 'belt', 28: 'bench', 29: 'bicycle', 30: 'binoculars', 31: 'bird', 32: 'birthday_cake', 33: 'blackberry', 34: 'blueberry', 35: 'book', 36: 'boomerang', 37: 'bottlecap', 38: 'bowtie', 39: 'bracelet', 40: 'brain', 41: 'bread', 42: 'bridge', 43: 'broccoli', 44: 'broom', 45: 'bucket', 46: 'bulldozer', 47: 'bus', 48: 'bush', 49: 'butterfly', 50: 'cactus', 51: 'cake', 52: 'calculator', 53: 'calendar', 54: 'camel', 55: 'camera', 56: 'camouflage', 57: 'campfire', 58: 'candle', 59: 'cannon', 60: 'canoe', 61: 'car', 62: 'carrot', 63: 'castle', 64: 'cat', 65: 'ceiling_fan', 66: 'cello', 67: 'cell_phone', 68: 'chair', 69: 'chandelier', 70: 'church', 71: 'circle', 72: 'clarinet', 73: 'clock', 74: 'cloud', 75: 'coffee_cup', 76: 'compass', 77: 'computer', 78: 'cookie', 79: 'cooler', 80: 'couch', 81: 'cow', 82: 'crab', 83: 'crayon', 84: 'crocodile', 85: 'crown', 86: 'cruise_ship', 87: 'cup', 88: 'diamond', 89: 'dishwasher', 90: 'diving_board', 91: 'dog', 92: 'dolphin', 93: 'donut', 94: 'door', 95: 'dragon', 96: 'dresser', 97: 'drill', 98: 'drums', 99: 'duck', 100: 'dumbbell', 101: 'ear', 102: 'elbow', 103: 'elephant', 104: 'envelope', 105: 'eraser', 106: 'eye', 107: 'eyeglasses', 108: 'face', 109: 'fan', 110: 'feather', 111: 'fence', 112: 'finger', 113: 'fire_hydrant', 114: 'fireplace', 115: 'firetruck', 116: 'fish', 117: 'flamingo', 118: 'flashlight', 119: 'flip_flops', 120: 'floor_lamp', 121: 'flower', 122: 'flying_saucer', 123: 'foot', 124: 'fork', 125: 'frog', 126: 'frying_pan', 127: 'garden', 128: 'garden_hose', 129: 'giraffe', 130: 'goatee', 131: 'golf_club', 132: 'grapes', 133: 'grass', 134: 'guitar', 135: 'hamburger', 136: 'hammer', 137: 'hand', 138: 'harp', 139: 'hat', 140: 'headphones', 141: 'hedgehog', 142: 'helicopter', 143: 'helmet', 144: 'hexagon', 145: 'hockey_puck', 146: 'hockey_stick', 147: 'horse', 148: 'hospital', 149: 'hot_air_balloon', 150: 'hot_dog', 151: 'hot_tub', 152: 'hourglass', 153: 'house', 154: 'house_plant', 155: 'hurricane', 156: 'ice_cream', 157: 'jacket', 158: 'jail', 159: 'kangaroo', 160: 'key', 161: 'keyboard', 162: 'knee', 163: 'knife', 164: 'ladder', 165: 'lantern', 166: 'laptop', 167: 'leaf', 168: 'leg', 169: 'light_bulb', 170: 'lighter', 171: 'lighthouse', 172: 'lightning', 173: 'line', 174: 'lion', 175: 'lipstick', 176: 'lobster', 177: 'lollipop', 178: 'mailbox', 179: 'map', 180: 'marker', 181: 'matches', 182: 'megaphone', 183: 'mermaid', 184: 'microphone', 185: 'microwave', 186: 'monkey', 187: 'moon', 188: 'mosquito', 189: 'motorbike', 190: 'mountain', 191: 'mouse', 192: 'moustache', 193: 'mouth', 194: 'mug', 195: 'mushroom', 196: 'nail', 197: 'necklace', 198: 'nose', 199: 'ocean', 200: 'octagon', 201: 'octopus', 202: 'onion', 203: 'oven', 204: 'owl', 205: 'paintbrush', 206: 'paint_can', 207: 'palm_tree', 208: 'panda', 209: 'pants', 210: 'paper_clip', 211: 'parachute', 212: 'parrot', 213: 'passport', 214: 'peanut', 215: 'pear', 216: 'peas', 217: 'pencil', 218: 'penguin', 219: 'piano', 220: 'pickup_truck', 221: 'picture_frame', 222: 'pig', 223: 'pillow', 224: 'pineapple', 225: 'pizza', 226: 'pliers', 227: 'police_car', 228: 'pond', 229: 'pool', 230: 'popsicle', 231: 'postcard', 232: 'potato', 233: 'power_outlet', 234: 'purse', 235: 'rabbit', 236: 'raccoon', 237: 'radio', 238: 'rain', 239: 'rainbow', 240: 'rake', 241: 'remote_control', 242: 'rhinoceros', 243: 'rifle', 244: 'river', 245: 'roller_coaster', 246: 'rollerskates', 247: 'sailboat', 248: 'sandwich', 249: 'saw', 250: 'saxophone', 251: 'school_bus', 252: 'scissors', 253: 'scorpion', 254: 'screwdriver', 255: 'sea_turtle', 256: 'see_saw', 257: 'shark', 258: 'sheep', 259: 'shoe', 260: 'shorts', 261: 'shovel', 262: 'sink', 263: 'skateboard', 264: 'skull', 265: 'skyscraper', 266: 'sleeping_bag', 267: 'smiley_face', 268: 'snail', 269: 'snake', 270: 'snorkel', 271: 'snowflake', 272: 'snowman', 273: 'soccer_ball', 274: 'sock', 275: 'speedboat', 276: 'spider', 277: 'spoon', 278: 'spreadsheet', 279: 'square', 280: 'squiggle', 281: 'squirrel', 282: 'stairs', 283: 'star', 284: 'steak', 285: 'stereo', 286: 'stethoscope', 287: 'stitches', 288: 'stop_sign', 289: 'stove', 290: 'strawberry', 291: 'streetlight', 292: 'string_bean', 293: 'submarine', 294: 'suitcase', 295: 'sun', 296: 'swan', 297: 'sweater', 298: 'swing_set', 299: 'sword', 300: 'syringe', 301: 'table', 302: 'teapot', 303: 'teddy-bear', 304: 'telephone', 305: 'television', 306: 'tennis_racquet', 307: 'tent', 308: 'The_Eiffel_Tower', 309: 'The_Great_Wall_of_China', 310: 'The_Mona_Lisa', 311: 'tiger', 312: 'toaster', 313: 'toe', 314: 'toilet', 315: 'tooth', 316: 'toothbrush', 317: 'toothpaste', 318: 'tornado', 319: 'tractor', 320: 'traffic_light', 321: 'train', 322: 'tree', 323: 'triangle', 324: 'trombone', 325: 'truck', 326: 'trumpet', 327: 't-shirt', 328: 'umbrella', 329: 'underwear', 330: 'van', 331: 'vase', 332: 'violin', 333: 'washing_machine', 334: 'watermelon', 335: 'waterslide', 336: 'whale', 337: 'wheel', 338: 'windmill', 339: 'wine_bottle', 340: 'wine_glass', 341: 'wristwatch', 342: 'yoga', 343: 'zebra', 344: 'zigzag'}
officehome_classes_label={33: 'Alarm_Clock', 32: 'Backpack', 36: 'Batteries', 15: 'Bed', 19: 'Bike', 2: 'Bottle', 46: 'Bucket', 49: 'Calculator', 48: 'Calendar', 53: 'Candles', 47: 'Chair', 54: 'Clipboards', 4: 'Computer', 18: 'Couch', 57: 'Curtains', 23: 'Desk_Lamp', 0: 'Drill', 45: 'Eraser', 1: 'Exit_Sign', 38: 'Fan', 5: 'File_Cabinet', 13: 'Flipflops', 50: 'Flowers', 11: 'Folder', 58: 'Fork', 3: 'Glasses', 16: 'Hammer', 25: 'Helmet', 10: 'Kettle', 12: 'Keyboard', 61: 'Knives', 51: 'Lamp_Shade', 9: 'Laptop', 64: 'Marker', 28: 'Monitor', 29: 'Mop', 26: 'Mouse', 21: 'Mug', 31: 'Notebook', 62: 'Oven', 40: 'Pan', 35: 'Paper_Clip', 27: 'Pen', 14: 'Pencil', 20: 'Postit_Notes', 43: 'Printer', 34: 'Push_Pin', 37: 'Radio', 63: 'Refrigerator', 39: 'Ruler', 55: 'Scissors', 41: 'Screwdriver', 6: 'Shelf', 8: 'Sink', 30: 'Sneakers', 59: 'Soda', 44: 'Speaker', 52: 'Spoon', 56: 'TV', 60: 'Table', 24: 'Telephone', 17: 'ToothBrush', 7: 'Toys', 42: 'Trash_Can', 22: 'Webcam'}


def parse_args(config_path,args):
    with open(config_path) as f:
        config=yaml.load(f,Loader=yaml.FullLoader)
    config=EasyDict(config)
    config.run = args.run
    config.mode = args.mode
    config.source = args.source
    config.target = args.target
    config.dataset = args.dataset
    config.data.annotation = args.annotation
    config.path.datasetroot = args.datasetroot
    config.network.num_class = args.num_class
    if config.mode=="train":
        config.network.model_weight = "None"
        config.network.sup_model_weight = "VisualSearch/{}_train/Checkpoints/{}/{}/ResNet50_{}_{}/run_1/[{}].pth".format(config.dataset,config.data.annotation,config.source,config.source,config.target,config.source[0])
        config.network.da_model_weight = "VisualSearch/{}_train/Checkpoints/{}/{}/{}_ResNet50_{}_{}/run_1/[{}2{}].pth".format(config.dataset,config.data.annotation,config.source,config.network.da_model,config.source,config.target,config.source[0],config.target[0])
    elif config.mode=="predict":
        config.network.model_weight ="VisualSearch/{}_train/Checkpoints/{}/{}/{}_ResNet50_{}_{}/run_{}/[{}2{}].pth".format(config.dataset,config.data.annotation,config.source,config.network.model,config.source,config.target,config.run,config.source[0],config.target[0])
        config.network.sup_model_weight = "None"
        config.network.da_model_weight = "None"
    else:
        Exception("input error mode")
    return config

def save_score(path,ids,argmax_indexs,argmax_scores,dataset):
    if dataset=="officehome":
        classes_label=officehome_classes_label
    elif dataset=="domainnet":
        classes_label=domainnet_classes_label
    lines=[]
    for i in range(len(ids)):
        line="{} {} {}\n".format(ids[i],classes_label[argmax_indexs[i]],argmax_scores[i])
        lines.append(line)
    f=open(path,"w")
    f.writelines(lines)
    f.close()


def get_anno_path(collection, domain, annotation_name, concept, rootpath=ROOT_PATH):
    anno_path = os.path.join(rootpath, collection, 'Annotations', 'Image', annotation_name, domain, '%s.txt' % concept)
    return anno_path

def get_pred_path(test_collection, domain, model_name, run, rootpath=ROOT_PATH):
    annotation_name = col2anno[test_collection]
    pred_path = os.path.join(rootpath, test_collection, 'Predictions', annotation_name, domain, model_name, "run_{}".format(run), 'id.concept.score.txt')
    return pred_path

def read_imset(collection, domain, rootpath=ROOT_PATH):
    imset_file = os.path.join(rootpath, collection, 'ImageSets', '%s.txt' % domain)
    imset = [x.strip() for x in open(imset_file) if x.strip()]
    assert(len(set(imset)) == len(imset))
    return imset


def read_domain_list(collection, rootpath=ROOT_PATH):
    domain_file = os.path.join(rootpath, collection, 'Annotations', 'domains.txt')
    domain_list = [x.strip() for x in open(domain_file) if x.strip()]
    assert(len(set(domain_list)) == len(domain_list))
    return domain_list


def read_concept_list(collection, annotation_name, rootpath=ROOT_PATH):
    concept_file = os.path.join(rootpath, collection, 'Annotations', annotation_name)
    concepts = [x.strip() for x in open(concept_file).readlines() if x.strip()]
    assert(len(set(concepts)) == len(concepts))
    return concepts


def read_anno(anno_path):
    pos_img_list = []
    for line in open(anno_path):
        imgid, label = line.strip().split()
        assert(int(label) == 1)
        pos_img_list.append(imgid)
    assert(len(set(pos_img_list)) == len(pos_img_list))
    return pos_img_list


def read_full_anno(collection, domain, annotation_name, rootpath=ROOT_PATH):
    concepts = read_concept_list(collection, annotation_name, rootpath)
    gt = {}

    for concept in concepts:
        anno_path = get_anno_path(collection, domain, annotation_name, concept, rootpath)
        pos_img_list = read_anno(anno_path)
        for im in pos_img_list:
            assert (im not in gt) 
            gt[im] = concept
    return gt

def read_pred(pred_path):
    lines = open(pred_path).readlines()
    pred = {}

    for line in lines:
        imgid, concept, score = line.strip().split()
        pred[imgid] = concept
    return pred

def compute_accuracy(preds, gts):
    assert(len(preds) == len(gts))
    count=0.0
    for imgid,pred_y in preds.items():
        if pred_y == gts[imgid]:
            count = count + 1
    accuracy = count / len(preds)
    return accuracy



def mixup_data(sourcex,targetx, sourcey,targety):
    lam = np.random.beta(1, 1)
    mixed_x = lam * sourcex + (1 - lam) * targetx
    mixed_y = lam * sourcey + (1 - lam) * targety
    return mixed_x, mixed_y

def cross_entropy_loss(input, target=None, pseudo=False, temperature=None):
    # target is unknown, maximize the largest logit
    if target is None:
        target = torch.argmax(input, dim=1)
        loss = F.cross_entropy(input, target)

    # softmax
    if target is not None and target.dtype == torch.int64:
        loss = F.cross_entropy(input, target)    

    # soft label is known, use the class of the highest score as the target
    if target is not None and target.dtype != torch.int64 and pseudo is True:
        target = target.argmax(dim=1)
        loss = F.cross_entropy(input, target)
    
    # soft label softmax
    if target is not None and target.dtype != torch.int64 and pseudo is False:
        if temperature is not None:
            input = input / temperature
            target = target / temperature
        log_input = torch.log_softmax(input, dim=1)
        loss = -(log_input * target).sum(1).mean()
    return loss

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count