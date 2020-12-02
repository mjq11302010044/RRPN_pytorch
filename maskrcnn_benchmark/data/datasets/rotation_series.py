import os
import pickle
import torch
import torch.utils.data
from PIL import Image
import sys
import numpy as np
import time
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
import json
from maskrcnn_benchmark.data.transforms import transforms as T
from maskrcnn_benchmark.structures.bounding_box import RBoxList
from maskrcnn_benchmark.utils.visualize import vis_image
import cv2


def get_ICDAR2013(mode, dataset_dir):
    DATASET_DIR = dataset_dir

    img_dir = "/ch2_training_images/"
    gt_dir = "/ch2_training_localization_transcription_gt"

    # gt_list = []
    # img_list = []

    im_infos = []
    image_dir = DATASET_DIR + img_dir
    gt_file_list = os.listdir(image_dir)

    gt_words = []

    if mode == 'train':
        cache_pkl = './data_cache/IC13_training.pkl'

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    for image in gt_file_list:

        prefix = image[:-4]
        im_path = os.path.join(image_dir, image)
        gt_path = os.path.join(dataset_dir + gt_dir, 'gt_' + prefix + '.txt')
        print(im_path)
        gt_list = open(gt_path, 'r', encoding='utf-8').readlines()
        im = cv2.imread(im_path)
        if im is None:
            print(im_path + '--> None')
            continue

        boxes = []
        for gt_ele in gt_list:
            gt_ele = gt_ele.replace('\n', '').replace('\ufeff', '')
            gt = gt_ele.split(',')

            if len(gt) > 1:
                gt_ind = np.array(gt[:8], dtype=np.float32)
                gt_ind = np.array(gt_ind, dtype=np.int32)
                words = gt[8]

                pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:

                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                if height * width * (800 / float(im.shape[0])) < 16 * 16 and mode == "train":
                    continue
                # return to width, height
                # if '###' in words:
                #    continue
                boxes.append([x_ctr, y_ctr, width, height, angle, words])
                gt_words.append(words)
        cls_num = 2
        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # prob
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

        # print ("boxes_size:", gt_boxes.shape[0])
        if gt_boxes.shape[0] > 0:
            max_overlaps = overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = overlaps.argmax(axis=1)
        else:
            continue

        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': im_path,
            'boxes': gt_boxes,
            'gt_words': gt_words,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")

    return im_infos


def get_ICDAR2015_RRC_PICK_TRAIN(mode, dataset_dir):
    # dir_path = "/home/shiki-alice/Downloads/ICDAR2015/ch4_training_images/"
    img_file_type = "jpg"
    a=dataset_dir
    dataset_dir=os.path.join(os.getcwd()+a[2:])
    image_dir = os.path.join(dataset_dir, 'ch4_training_images/')
    gt_dir = os.path.join(dataset_dir, 'ch4_training_localization_transcription_gt/')

    image_list = os.listdir(image_dir)
    image_list.sort()
    im_infos = []

    cache_file = './data_cache/IC15_training.pkl'
    if os.path.isfile(cache_file):
        return pickle.load(open(cache_file, 'rb'))

    for image in image_list:

        prefix = image[:-4]
        img_name = os.path.join(image_dir, image)
        gt_name = os.path.join(gt_dir, 'gt_' + prefix + '.txt')

        # img_name = dir_path + img_list[idx]
        # gt_name = gt_dir + gt_list[idx]

        easy_boxes = []
        hard_boxes = []

        boxes = []
        # print gt_name
        gt_obj = open(gt_name, 'r',encoding='utf-8')
        gt_txt = gt_obj.read()
        gt_split = gt_txt.split('\n')
        img = cv2.imread(img_name)
        print(img_name)
        f = False
        # print '-------------'
        for gt_line in gt_split:

            if not f:
                gt_ind = gt_line.split('\\')

                f = True
            else:
                gt_ind = gt_line.split(',')
            if len(gt_ind) > 3 and '###' not in gt_ind[8]:
                # condinate_list = gt_ind[2].split(',')
                # print ("easy: ", gt_ind)

                pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:

                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            if len(gt_ind) > 3 and '###' in gt_ind[8]:
                # condinate_list = gt_ind[2].split(',')

                # print "hard: ", gt_ind

                pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                angle = 0

                if edge1 > edge2:

                    width = edge1
                    height = edge2
                    if pt1[0] - pt2[0] != 0:
                        angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                elif edge2 >= edge1:
                    width = edge2
                    height = edge1
                    # print pt2[0], pt3[0]
                    if pt2[0] - pt3[0] != 0:
                        angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                    else:
                        angle = 90.0
                if angle < -45.0:
                    angle = angle + 180

                x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                hard_boxes.append([x_ctr, y_ctr, width, height, angle])

        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes) / 3)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': img_name,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': img.shape[0],
            'width': img.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_file, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print ("Save pickle done.")

    return im_infos


def get_ICDAR2017_mlt(mode, dataset_dir):
    DATASET_DIR = dataset_dir
    task = 'double_class'
    prefetched = True if os.path.isfile('./data_cache/ICDAR2017_training_cache.pkl') else False
    im_infos = []

    data_list = []
    gt_list = []
    img_type = ['jpg', 'png', 'gif']
    cls_list = {'background': 0, 'Arabic': 1, 'English': 2, 'Japanese': 3, 'French': 4, 'German': 5, 'Chinese': 6,
                'Korean': 7, 'Italian': 8, 'Bangla': 9}

    if not prefetched:
        # training set contains 7200 images with
        if mode == "train":
            for i in range(7200):
                img_candidate_path = DATASET_DIR + "ch8_training_images_" + str(int(i / 1000) + 1) + "/" + 'img_' + str(
                    i + 1) + "."
                if os.path.isfile(img_candidate_path + img_type[0]):
                    img_candidate_path += img_type[0]
                elif os.path.isfile(img_candidate_path + img_type[1]):
                    img_candidate_path += img_type[1]
                elif os.path.isfile(img_candidate_path + img_type[2]):
                    im = Image.open(img_candidate_path + img_type[2])
                    im = im.convert('RGB')
                    im.save(img_candidate_path + "jpg", "jpeg")
                    img_candidate_path = img_candidate_path + "jpg"
                data_list.append(img_candidate_path)
                # print ("data_list:", len(data_list))

                gt_candidate_path = DATASET_DIR + "ch8_training_localization_transcription_gt/" + 'gt_img_' + str(
                    i + 1) + ".txt"
                if os.path.isfile(gt_candidate_path):
                    gt_list.append(gt_candidate_path)
                # print ("gt_list:", len(gt_list))

                f_gt = open(gt_candidate_path)
                f_content = f_gt.read()

                lines = f_content.split('\n')
                print (img_candidate_path)
                img = cv2.imread(img_candidate_path)
                boxes = []
                for gt_line in lines:
                    # print (gt_line)
                    gt_ind = gt_line.split(',')

                    if len(gt_ind) > 3:

                        pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                        pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                        pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                        pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                        angle = 0

                        if edge1 > edge2:

                            width = edge1
                            height = edge2
                            if pt1[0] - pt2[0] != 0:
                                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        elif edge2 >= edge1:
                            width = edge2
                            height = edge1
                            # print pt2[0], pt3[0]
                            if pt2[0] - pt3[0] != 0:
                                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        if angle < -45.0:
                            angle = angle + 180

                        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                        if height * width < 32 * 32:
                             continue

                        if not gt_ind[8].replace('\n', '') in ['English', 'French', 'German', 'Italian']:
                            continue

                        boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])

                # print ("line_size:", len(lines))

                cls_num = 2
                if task == "multi_class":
                    cls_num = len(cls_list.keys())

                len_of_bboxes = len(boxes)
                gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
                gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
                overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
                seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

                if task == "multi_class":
                    gt_boxes = []  # np.zeros((len_of_bboxes, 5), dtype=np.int16)
                    gt_classes = []  # np.zeros((len_of_bboxes), dtype=np.int32)
                    overlaps = []  # np.zeros((len_of_bboxes, cls_num), dtype=np.float32) #text or non-text
                    seg_areas = []  # np.zeros((len_of_bboxes), dtype=np.float32)

                for idx in range(len(boxes)):

                    if task == "multi_class":
                        if not boxes[idx][5] in cls_list:
                            print (boxes[idx][5] + " not in list")
                            continue
                        gt_classes.append(cls_list[boxes[idx][5]])  # cls_text
                        overlap = np.zeros((cls_num))
                        overlap[cls_list[boxes[idx][5]]] = 1.0  # prob
                        overlaps.append(overlap)
                        seg_areas.append((boxes[idx][2]) * (boxes[idx][3]))
                        gt_boxes.append([boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]])
                    else:
                        gt_classes[idx] = 1  # cls_text
                        overlaps[idx, 1] = 1.0  # prob
                        seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
                        gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

                if task == "multi_class":
                    gt_classes = np.array(gt_classes)
                    overlaps = np.array(overlaps)
                    seg_areas = np.array(seg_areas)
                    gt_boxes = np.array(gt_boxes)

                # print ("boxes_size:", gt_boxes.shape[0])
                if gt_boxes.shape[0] > 0:
                    max_overlaps = overlaps.max(axis=1)
                    # gt class that had the max overlap
                    max_classes = overlaps.argmax(axis=1)
                else:
                    continue
                im_info = {
                    'gt_classes': gt_classes,
                    'max_classes': max_classes,
                    'image': img_candidate_path,
                    'boxes': gt_boxes,
                    'flipped': False,
                    'gt_overlaps': overlaps,
                    'seg_areas': seg_areas,
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'max_overlaps': max_overlaps,
                    'rotated': True
                }
                im_infos.append(im_info)

            f_save_pkl = open('./data_cache/ICDAR2017_training_cache.pkl', 'wb')
            pickle.dump(im_infos, f_save_pkl)
            f_save_pkl.close()
            print ("Save pickle done.")
        elif mode == "validation":
            for i in range(1800):
                img_candidate_path = DATASET_DIR + "ch8_validation_images/" + 'img_' + str(i + 1) + "."
                if os.path.isfile(img_candidate_path + img_type[0]):
                    img_candidate_path += img_type[0]
                elif os.path.isfile(img_candidate_path + img_type[1]):
                    img_candidate_path += img_type[1]
                elif os.path.isfile(img_candidate_path + img_type[2]):
                    im = Image.open(img_candidate_path + img_type[2])
                    im = im.convert('RGB')
                    im.save(img_candidate_path + "jpg", "jpeg")
                    img_candidate_path = img_candidate_path + "jpg"
                data_list.append(img_candidate_path)
                # print ("data_list:", len(data_list))

                gt_candidate_path = DATASET_DIR + "ch8_validation_localization_transcription_gt/" + 'gt_img_' + str(
                    i + 1) + ".txt"
                if os.path.isfile(gt_candidate_path):
                    gt_list.append(gt_candidate_path)
                # print ("gt_list:", len(gt_list))

                f_gt = open(gt_candidate_path)
                f_content = f_gt.read()

                lines = f_content.split('\n')
                print (img_candidate_path)
                img = cv2.imread(img_candidate_path)
                boxes = []

                for gt_line in lines:
                    # print (gt_line)
                    gt_ind = gt_line.split(',')
                    if len(gt_ind) > 3:

                        pt1 = (int(gt_ind[0]), int(gt_ind[1]))
                        pt2 = (int(gt_ind[2]), int(gt_ind[3]))
                        pt3 = (int(gt_ind[4]), int(gt_ind[5]))
                        pt4 = (int(gt_ind[6]), int(gt_ind[7]))

                        edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
                        edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

                        angle = 0

                        if edge1 > edge2:

                            width = edge1
                            height = edge2
                            if pt1[0] - pt2[0] != 0:
                                angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        elif edge2 >= edge1:
                            width = edge2
                            height = edge1
                            # print pt2[0], pt3[0]
                            if pt2[0] - pt3[0] != 0:
                                angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                            else:
                                angle = 90.0
                        if angle < -45.0:
                            angle = angle + 180

                        x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
                        y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

                        if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
                            continue

                        boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])

                cls_num = 2
                if task == "multi_class":
                    cls_num = len(cls_list.keys())

                len_of_bboxes = len(boxes)
                gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
                gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
                overlaps = np.zeros((len_of_bboxes, cls_num), dtype=np.float32)  # text or non-text
                seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

                for idx in range(len(boxes)):

                    if task == "multi_class":
                        if not boxes[idx][5] in cls_list:
                            break
                        gt_classes[idx] = cls_list[boxes[idx][5]]  # cls_text
                        overlaps[idx, cls_list[boxes[idx][5]]] = 1.0  # prob
                    else:
                        gt_classes[idx] = 1  # cls_text
                        overlaps[idx, 1] = 1.0  # prob
                    seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])
                    gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]

                max_overlaps = overlaps.max(axis=1)
                # gt class that had the max overlap
                max_classes = overlaps.argmax(axis=1)

                im_info = {
                    'gt_classes': gt_classes,
                    'max_classes': max_classes,
                    'image': img_candidate_path,
                    'boxes': gt_boxes,
                    'flipped': False,
                    'gt_overlaps': overlaps,
                    'seg_areas': seg_areas,
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'max_overlaps': max_overlaps,
                    'rotated': True
                }
                im_infos.append(im_info)

            f_save_pkl = open('ICDAR2017_validation_cache.pkl', 'wb')
            pickle.dump(im_infos, f_save_pkl)
            f_save_pkl.close()
            print ("Save pickle done.")
    else:
        if mode == "train":
            f_pkl = open('./data_cache/ICDAR2017_training_cache.pkl', 'rb')
            im_infos = pickle.load(f_pkl)
        if mode == "validation":
            f_pkl = open('ICDAR2017_validation_cache.pkl', 'rb')
            im_infos = pickle.load(f_pkl)
    return im_infos


def get_ICDAR_LSVT_full(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[0, 3000],
        'train':[3000, 30000],
        'full':[0, 30000]
    }

    vis = False

    cache_file = './data_cache/LSVT_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    im_codes = range(data_split[mode][0], data_split[mode][1])
    gt_json = os.path.join(dataset_dir, 'train_full_labels.json')

    gt_dict = json.load(open(gt_json, 'r'))

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    for imnum in im_codes:
        forder = int(imnum / 15000)
        imfolder = os.path.join(dataset_dir, 'train_full_images_'+str(forder), 'train_full_images_'+str(forder))
        impath = os.path.join(imfolder, 'gt_' + str(imnum) + '.jpg')
        gt_code = 'gt_' + str(imnum)
        gt_anno = gt_dict[gt_code]

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []

        print(str(imnum) + '/' + str(data_split[mode][0] + num_samples), impath)

        for i in range(inst_num):
            inst = gt_anno[i]
            # print(inst.keys())
            poly = np.array(inst['points'])
            words = inst['transcription']
            illegibility = inst['illegibility']

            color = (255, 0, 255) if illegibility else (0, 0, 255)
            if poly.shape[0] > 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)
                if vis:
                    rect_pt_num = rect.shape[0]
                    for i in range(rect.shape[0]):
                        cv2.line(im, (rect[i % rect_pt_num][0], rect[i % rect_pt_num][1]),
                                 (rect[(i + 1) % rect_pt_num][0], rect[(i + 1) % rect_pt_num][1]), (0, 255, 0), 2)
            if vis:
                pt_num = poly.shape[0]
                for i in range(poly.shape[0]):
                    cv2.line(im, (poly[i % pt_num][0], poly[i % pt_num][1]),
                             (poly[(i + 1) % pt_num][0], poly[(i + 1) % pt_num][1]), color, 2)

            poly = poly.reshape(-1)
            pt1 = (int(poly[0]), int(poly[1]))
            pt2 = (int(poly[2]), int(poly[3]))
            pt3 = (int(poly[4]), int(poly[5]))
            pt4 = (int(poly[6]), int(poly[7]))

            edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

            angle = 0

            if edge1 > edge2:

                width = edge1
                height = edge2
                if pt1[0] - pt2[0] != 0:
                    angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            elif edge2 >= edge1:
                width = edge2
                height = edge1
                # print pt2[0], pt3[0]
                if pt2[0] - pt3[0] != 0:
                    angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
            y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

            # if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
            #    continue

            if illegibility:
                hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            else:
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
        # img_pil = Image.fromarray(im)
        boxes = []
        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes) / 5)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0 or gt_boxes.shape[0] > 100:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos


def get_ICDAR_ReCTs_full(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[0, 3000],
        'train':[0, 18000],
        'full':[0, 30000]
    }

    vis = False

    cache_file = './data_cache/ReCTs_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    # im_codes = range(data_split[mode][0], data_split[mode][1])
    # gt_json = os.path.join(dataset_dir, 'train_full_labels.json')

    # gt_dict = json.load(open(gt_json, 'r'))

    gt_dir = os.path.join(dataset_dir, mode, 'gt')
    im_dir = os.path.join(dataset_dir, mode, 'image')

    imlist = os.listdir(im_dir)

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    cnt = 0

    for imname in imlist:
        # forder = int(imnum / 15000)
        # imfolder = os.path.join(dataset_dir, 'train_full_images_'+str(forder), 'train_full_images_'+str(forder))
        impath = os.path.join(im_dir, imname)
        gtpath = os.path.join(gt_dir, imname.split('.')[0] + '.json')
        gt_anno = open(gtpath, 'r')

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []
        cnt += 1
        print(str(cnt) + '/' + str(data_split[mode][0] + num_samples), impath)

        # using lines
        lines = gt_anno['lines']

        for i in range(len(lines)):
            inst = lines[i]
            # print(inst.keys())
            poly = np.array(inst['points']).reshape(-1, 2)
            words = inst['transcription']
            ignore = inst['ignore']

            color = (255, 0, 255) if not ignore else (0, 0, 255)
            if poly.shape[0] > 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)
                if vis:
                    rect_pt_num = rect.shape[0]
                    for i in range(rect.shape[0]):
                        cv2.line(im, (rect[i % rect_pt_num][0], rect[i % rect_pt_num][1]),
                                 (rect[(i + 1) % rect_pt_num][0], rect[(i + 1) % rect_pt_num][1]), (0, 255, 0), 2)
            if vis:
                pt_num = poly.shape[0]
                for i in range(poly.shape[0]):
                    cv2.line(im, (poly[i % pt_num][0], poly[i % pt_num][1]),
                             (poly[(i + 1) % pt_num][0], poly[(i + 1) % pt_num][1]), color, 2)

            poly = poly.reshape(-1)
            pt1 = (int(poly[0]), int(poly[1]))
            pt2 = (int(poly[2]), int(poly[3]))
            pt3 = (int(poly[4]), int(poly[5]))
            pt4 = (int(poly[6]), int(poly[7]))

            edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

            angle = 0

            if edge1 > edge2:

                width = edge1
                height = edge2
                if pt1[0] - pt2[0] != 0:
                    angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            elif edge2 >= edge1:
                width = edge2
                height = edge1
                # print pt2[0], pt3[0]
                if pt2[0] - pt3[0] != 0:
                    angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
            y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

            # if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
            #    continue

            if ignore:
                hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            else:
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
        # img_pil = Image.fromarray(im)
        boxes = []
        boxes.extend(easy_boxes)
        boxes.extend(hard_boxes[0: int(len(hard_boxes) / 5)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0 or gt_boxes.shape[0] > 100:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos



def get_ICDAR_ArT(mode, dataset_dir):

    assert mode in ['train', 'val', 'full'], 'mode not in ' + str(['train', 'val', 'full'])

    data_split = {
        'val':[4000, 5603],
        'train':[0, 4000],
        'full':[0, 5603]
    }

    vis = False

    dataset_dir = os.path.join(dataset_dir, 'ArT_detect_train')

    cache_file = './data_cache/ArT_det_' + mode + '.pkl'
    if os.path.isfile(cache_file):
        print('dataset cache found, loading from it...')
        im_infos = pickle.load(open(cache_file, 'rb'))
        print('load done')
        return im_infos

    im_codes = range(data_split[mode][0], data_split[mode][1])
    gt_json = os.path.join(dataset_dir, 'train_labels.json')

    gt_dict = json.load(open(gt_json, 'r'))

    im_infos = []

    num_samples = data_split[mode][1] - data_split[mode][0]

    for imnum in im_codes:
        # forder = int(imnum / 15000)
        imfolder = os.path.join(dataset_dir, 'train_images')
        impath = os.path.join(imfolder, 'gt_' + str(imnum) + '.jpg')
        gt_code = 'gt_' + str(imnum)
        gt_anno = gt_dict[gt_code]

        inst_num = len(gt_anno)

        im = cv2.imread(impath)

        easy_boxes = []
        hard_boxes = []

        print(str(imnum) + '/' + str(data_split[mode][0] + num_samples), impath)

        for i in range(inst_num):
            inst = gt_anno[i]
            # print(inst.keys())
            poly = np.array(inst['points'])
            words = inst['transcription']
            illegibility = inst['illegibility']
            language = inst['language']

            color = (255, 0, 255) if illegibility else (0, 0, 255)
            if poly.shape[0] > 4:
                # print('polygon:', poly.shape[0])
                rect = cv2.minAreaRect(poly)
                poly = np.array(cv2.boxPoints(rect), np.int)
                # print('rect:', rect)
                if vis:
                    rect_pt_num = rect.shape[0]
                    for i in range(rect.shape[0]):
                        cv2.line(im, (rect[i % rect_pt_num][0], rect[i % rect_pt_num][1]),
                                 (rect[(i + 1) % rect_pt_num][0], rect[(i + 1) % rect_pt_num][1]), (0, 255, 0), 2)
            if vis:
                pt_num = poly.shape[0]
                for i in range(poly.shape[0]):
                    cv2.line(im, (poly[i % pt_num][0], poly[i % pt_num][1]),
                             (poly[(i + 1) % pt_num][0], poly[(i + 1) % pt_num][1]), color, 2)

            if poly.shape[0] < 4:
                print('poly:', poly.shape, np.array(inst['points']).shape)
                continue
            poly = poly.reshape(-1)
            pt1 = (int(poly[0]), int(poly[1]))
            pt2 = (int(poly[2]), int(poly[3]))
            pt3 = (int(poly[4]), int(poly[5]))
            pt4 = (int(poly[6]), int(poly[7]))

            edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
            edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

            angle = 0

            if edge1 > edge2:

                width = edge1
                height = edge2
                if pt1[0] - pt2[0] != 0:
                    angle = -np.arctan(float(pt1[1] - pt2[1]) / float(pt1[0] - pt2[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            elif edge2 >= edge1:
                width = edge2
                height = edge1
                # print pt2[0], pt3[0]
                if pt2[0] - pt3[0] != 0:
                    angle = -np.arctan(float(pt2[1] - pt3[1]) / float(pt2[0] - pt3[0])) / 3.1415926 * 180
                else:
                    angle = 90.0
            if angle < -45.0:
                angle = angle + 180

            x_ctr = float(pt1[0] + pt3[0]) / 2  # pt1[0] + np.abs(float(pt1[0] - pt3[0])) / 2
            y_ctr = float(pt1[1] + pt3[1]) / 2  # pt1[1] + np.abs(float(pt1[1] - pt3[1])) / 2

            # if height * width * (800 / float(img.shape[0])) < 16 * 16 and mode == "train":
            #     continue

            if illegibility:
                hard_boxes.append([x_ctr, y_ctr, width, height, angle])
            else:
                easy_boxes.append([x_ctr, y_ctr, width, height, angle])

            # boxes.append([x_ctr, y_ctr, width, height, angle, gt_ind[8]])
        # img_pil = Image.fromarray(im)
        boxes = []
        boxes.extend(easy_boxes)
        # boxes.extend(hard_boxes[0: int(len(hard_boxes) / 3)])

        len_of_bboxes = len(boxes)
        gt_boxes = np.zeros((len_of_bboxes, 5), dtype=np.int16)
        gt_classes = np.zeros((len_of_bboxes), dtype=np.int32)
        overlaps = np.zeros((len_of_bboxes, 2), dtype=np.float32)  # text or non-text
        seg_areas = np.zeros((len_of_bboxes), dtype=np.float32)

        for idx in range(len(boxes)):
            gt_boxes[idx, :] = [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]]
            gt_classes[idx] = 1  # cls_text
            overlaps[idx, 1] = 1.0  # cls_text
            seg_areas[idx] = (boxes[idx][2]) * (boxes[idx][3])

        # img_pil = vis_image(img_pil, gt_boxes)
        # img_pil.save('gt_LSVT.jpg', 'jpeg')
        # break
        max_overlaps = overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = overlaps.argmax(axis=1)
        if gt_boxes.shape[0] <= 0:
            continue
        # print('gt_boxes:', gt_boxes)
        im_info = {
            'gt_classes': gt_classes,
            'max_classes': max_classes,
            'image': impath,
            'boxes': gt_boxes,
            'flipped': False,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': im.shape[0],
            'width': im.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    print('Saving pkls...')
    pkl_f = open(cache_file, 'wb')
    pickle.dump(im_infos, pkl_f)
    pkl_f.close()
    print('done')
    return im_infos


DATASET = {
    'IC13':get_ICDAR2013,
    'IC15':get_ICDAR2015_RRC_PICK_TRAIN,
    'IC17mlt':get_ICDAR2017_mlt,
    'LSVT':get_ICDAR_LSVT_full,
    'ArT':get_ICDAR_ArT,
    'ReCTs':get_ICDAR_ReCTs_full,
}


_DEBUG = False
class RotationDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "text"
    )

    def __init__(self, database, use_difficult=False, transforms=None):
        # database:{dataset_name, dataset_dir}

        self.transforms = transforms

        self.annobase = []

        for dataset_name in database:
            if dataset_name in DATASET:
                self.annobase.extend(DATASET[dataset_name]('train', database[dataset_name]))

        print('DATASET: Total samples from:', database.keys(), len(self.annobase))

        self.ids = [anno['image'][:-4] for anno in self.annobase]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = RotationDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.mixup = T.MixUp(mix_ratio=0.1)
        self.num_samples = len(self.annobase)

    def __getitem__(self, index):

        # if _DEBUG:
        # index = 0

        # img_id = self.ids[index]

        im_path = self.annobase[index]['image']# os.path.join(self.root, img_id + '.jpg')
        img = Image.open(im_path).convert("RGB")
        # im = cv2.imread(im_path)
        anno = self.annobase[index]
        target = RBoxList(torch.from_numpy(anno["boxes"]), (anno['width'], anno['height']), mode="xywha")
        target.add_field("labels", torch.from_numpy(anno["gt_classes"]))
        target.add_field("difficult", torch.Tensor([0 for i in range(len(anno["gt_classes"]))]))

        target = target.clip_to_image(remove_empty=True)
        # print('target:', target, im_path)
        if self.transforms is not None:
            # off = int(self.num_samples * np.random.rand())
            # mix_index = (off + index) % self.num_samples
            # img_mix = Image.open(self.annobase[mix_index]['image']).convert("RGB")
            # img, target = self.mixup(img, img_mix, target)
            img, target = self.transforms(img, target)
        if _DEBUG:
            if not target is None:
                self.show_boxes(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, index):

        return {"height": self.annobase[index]['height'], "width": self.annobase[index]['width']}

    def map_class_id_to_class_name(self, class_id):
        return RotationDataset.CLASSES[class_id]

    def show_boxes(self, img, target):
        bbox_np = target.bbox.data.cpu().numpy()
        # print('image shape:', img.size())
        np_img = np.transpose(np.uint8(img.data.cpu().numpy()), (1, 2, 0))
        img_pil = Image.fromarray(np_img)
        # print('bbox_np:', bbox_np)
        draw_img = vis_image(img_pil, bbox_np)
        draw_img.save('gt_show.jpg', 'jpeg')
        # print('Sleep for show...')
        # time.sleep(2)

if __name__ == '__main__':
    get_ICDAR_LSVT_full('train', '../datasets/LSVT/')
