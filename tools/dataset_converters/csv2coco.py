# -*- coding: utf-8 -*-
'''
@time: 2019/01/11 11:28
spytensor
'''

import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
# from IPython import embed
from sklearn.model_selection import train_test_split
np.random.seed(41)

#0为背景
classname_to_id = {'Bus': 1, 'Small_Medium_Car': 2, 'Trucks': 3, 'Motors': 4, 'Special_vehicle': 5, 'Tiny_car': 6, 'Lorry': 7, 'Two-Wheels':8}
class_category = {'Bus': ['Bus'], 'Small_Medium_Car': ['Sedan_Car', 'SUV', 'MiniVan', 'Small_Medium','Vehicle_light'], 'Trucks': ['BigTruck', 'SmallTruck', 'Trucks'],  \
     'Motors': ['Motor-Tricycle', 'Tricycle', 'SmallTruckWithPerson', 'Motors'],  'Special_vehicle': ['Special_vehicle'], 'Tiny_car': ['Tiny_car'], 'Lorry': ['Lorry'], \
     'Two-Wheels': ['Motorcycle']}

occ_to_id = {'full_visible': 1, 'occluded': 2, 'heavily_occluded': 3, 'invisible': 4, 'unknown': 5}
occ_category = {'full_visible': ['full_visible', 'full_visible_a'], 'occluded':['occluded', 'occluded_c'], 'heavily_occluded':['heavily_occluded', 'heavily_occluded_c'], \
    'invisible':['invisible', 'invisible_d'], 'unknown':['unknown']}

age_to_id = {'Adult': 1, 'Child': 2, 'Teenagers': 3, 'unknown': 4}
age_category = {'Adult': ['Adult'], 'Child':['Child'], 'Teenagers':['Teenagers'], 'unknown':['unknown']}

ori_to_id = {'back': 1, 'front': 2, 'left': 3, 'left_anterior': 4, 'left_back': 5, 'right': 6, 'right_back': 7, 'right_front': 8, 'unknow': 9}
ori_category = {'back':['back'], 'front': ['front'], 'left': ['left'], 'left_anterior': ['left_anterior'], 'left_back':['left_back'], 'right':['right'], 'right_back':['right_back'], \
    'right_front':['right_front'], 'unknow':['unknow']}

pose_to_id = {'Bended': 1, 'Cyclist': 2, 'Lier': 3, 'Pedestrian': 4, 'Sitter': 5, 'Unknown': 6, 'NonTarget': 7, 'InVehicle': 8, 'PersonRideCycle':9}
pose_category = {'Bended':['Bended'], 'Cyclist':['Cyclist', 'Bicyclist', 'Motorcyclist', 'Tricyclist', 'WithPrsonalTransporter'], 'Lier': ['Lier'], 'Pedestrian':['Pedestrian'], \
    'Sitter':['Sitter', 'Squatter', 'WithWheelchair'], 'Unknown':['Unknown', 'Dummy','Others'], 'NonTarget':['NonTarget'], 'InVehicle':['InVehicle'], 'PersonRideCycle':['PersonRideMotorcycle', 'PersonRideTricycle', \
        'PersonPushBicycle', 'PersonRideBicycle']}
class Csv2CoCo:

    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.occlusion = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # files = open(save_path, 'w')
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示
    

    def get_label(self, org_label):
        # for k, v in class_category.items():
        for k, v in occ_category.items():
        # for k, v in age_category.items():
        # for k, v in ori_category.items():
        # for k, v in pose_category.items():
            # if org_label == 'PersonRideTricycle':
            #     print(org_label)
            if org_label in v:
                return k
            

    # 由txt文件构建COCO
    def to_coco(self, keys):
        self._init_categories()
        self._init_occlusion()
        for key in keys:
            self.images.append(self._image(key))
            # shapes = self.total_annos[key]['vehicle']  # 一张图像的所有box集合 
            if 'person' not in self.total_annos[key].keys():
                continue
            shapes = self.total_annos[key]['person']
            for shape in shapes:
                # ignore = shape['attrs']['ignore']
                # occlusion = shape['attrs']['occlusion']
                # truncation = shape['attrs']['truncation']
                # label = shape['attrs']['type']
                # print(label)
                label = shape['attrs']['occlusion']
                # print(label)
                # label = shape['attrs']['age']
                # label = shape['attrs']['orientation']

                # if ignore == 'yes' or truncation=='VeryLow' or label == 'Non-Vehicle_others':
                #     continue
                bboxi = shape['data']
                label = self.get_label(label)
                annotation = self._annotation(bboxi, label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1


        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        # instance['occlusion'] = self.occlusion
        return instance

    # 构建类别
    def _init_categories(self):
        # for k, v in classname_to_id.items():
        for k, v in occ_to_id.items():
        # for k, v in age_to_id.items():
        # for k, v in ori_to_id.items():
        # for k, v in pose_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    def _init_occlusion(self):
        for k, v in occ_to_id.items():
            occlusion = {}
            occlusion['id'] = v
            occlusion['name'] = k
            self.occlusion.append(occlusion)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(self.image_dir + path)
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, bbox, label):
        # label = shape[-1]
        # points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        # print(label)
        print(label)
        # annotation['category_id'] = int(classname_to_id[label])

        annotation['category_id'] = int(occ_to_id[label])
        # annotation['category_id'] = int(age_to_id[label])
        # annotation['category_id'] = int(ori_to_id[label])
        # annotation['category_id'] = int(pose_to_id[label])
        # annotation['occ_id'] = int(occ_to_id[occlusion])
        # annotation['category_id'] = int(occ_to_id[occlusion])
        # annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(bbox)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(bbox)
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # 计算面积
    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a
   

if __name__ == '__main__':

    org_json_file = 'D:/DataSet/adas_mini/adas_test/tococo/2pe_person_occlusion_classification/data.json'
    image_dir = 'D:/DataSet/adas_mini/adas_test/tococo/2pe_person_occlusion_classification/data/'
    saved_path = 'D:/DataSet/adas_mini/adas_test/tococo/2pe_person_occlusion_classification'

    total_annotations = {}
    for annotation in open(org_json_file, 'r', encoding='utf-8'):
        annotation_dict = json.loads(annotation)
        total_annotations[annotation_dict['image_key']] = annotation_dict

    l2c_train = Csv2CoCo(image_dir=image_dir, total_annos=total_annotations)
    total_keys = list(total_annotations.keys())
    train_instance = l2c_train.to_coco(total_keys)

    # if not os.path.exists('%s/annotations/' % saved_path):
    #     os.makedirs('%s/annotations/' % saved_path)
    # if not os.path.exists('%s/images/'%saved_path):
    #     os.makedirs('%s/images/'%saved_path)
    # if not os.path.exists('%s/images/'%saved_path):
    #     os.makedirs('%s/images/'%saved_path)

    l2c_train.save_coco_json(train_instance, '%s/person_occlusion_classification.json'%saved_path)



    # csv_file = "train.csv"
    # image_dir = "images/"
    # saved_coco_path = "./"
    # # 整合csv格式标注文件
    # total_csv_annotations = {}
    # annotations = pd.read_csv(csv_file,header=None).values
    # for annotation in annotations:
    #     key = annotation[0].split(os.sep)[-1]
    #     value = np.array([annotation[1:]])
    #     if key in total_csv_annotations.keys():
    #         total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
    #     else:
    #         total_csv_annotations[key] = value
    # # 按照键值划分数据
    # total_keys = list(total_csv_annotations.keys())
    # train_keys, val_keys = train_test_split(total_keys, test_size=0.2)
    # print("train_n:", len(train_keys), 'val_n:', len(val_keys))
    # # 创建必须的文件夹
    # if not os.path.exists('%scoco/annotations/'%saved_coco_path):
    #     os.makedirs('%scoco/annotations/'%saved_coco_path)
    # if not os.path.exists('%scoco/images/train2017/'%saved_coco_path):
    #     os.makedirs('%scoco/images/train2017/'%saved_coco_path)
    # if not os.path.exists('%scoco/images/val2017/'%saved_coco_path):
    #     os.makedirs('%scoco/images/val2017/'%saved_coco_path)
    # # 把训练集转化为COCO的json格式
    # l2c_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    # train_instance = l2c_train.to_coco(train_keys)
    # l2c_train.save_coco_json(train_instance, '%scoco/annotations/instances_train2017.json'%saved_coco_path)
    # for file in train_keys:
    #     shutil.copy(image_dir+file,"%scoco/images/train2017/"%saved_coco_path)
    # for file in val_keys:
    #     shutil.copy(image_dir+file,"%scoco/images/val2017/"%saved_coco_path)
    # # 把验证集转化为COCO的json格式
    # l2c_val = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    # val_instance = l2c_val.to_coco(val_keys)
    # l2c_val.save_coco_json(val_instance, '%scoco/annotations/instances_val2017.json'%saved_coco_path)

