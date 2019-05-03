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


'''
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

'''

def get_irra_XXX(mode, dataset_dir, XXX):
    im_infos = []

    img_dir = os.path.join(dataset_dir, 'imshow_picture/', XXX)
    anno_dir = os.path.join(dataset_dir, 'coordinates/', XXX)

    anno_list = os.listdir(anno_dir)
    anno_list.sort()

    #split = {
    #    'train': [0, 100],
    #    'val': [100, 147]
    #}

    base_list = anno_list# [split[mode][0]:split[mode][1]]
    cache_pkl = ''
    if mode == 'train':
        cache_pkl = './data_cache/nofilter_irra_training_' + XXX + '.pkl'
    else:
        cache_pkl = './data_cache/nofilter_irra_val' + XXX + '.pkl'

    if os.path.isfile(cache_pkl):
        return pickle.load(open(cache_pkl, 'rb'))

    print('XXX:', XXX, len(base_list))

    for anno_name in base_list:
        anno_path = os.path.join(anno_dir, anno_name)
        # f_impath = os.path.join(img_dir, 'usefilter_' + anno_name.split('.')[0] + '.jpg')
        f_impath = os.path.join(img_dir, 'nofilter_' + anno_name.split('.')[0] + '.jpg')

        f_img = cv2.imread(f_impath)
        # nf_img = cv2.imread(nf_impath)

        anno_lines = open(anno_path, 'r').readlines()

        polys = []
        boxes = []
        for anno in anno_lines:
            if len(anno) > 10:
                print('annos:', anno.replace('\n', '').split('\t'), len(anno))
                anno_split = anno.replace('\n', '').split('\t')

                # lt, lb, rt, rb --> lt, rt, rb, lb
                poly = np.array(
                    [anno_split[1], anno_split[0],
                     anno_split[5], anno_split[4],
                     anno_split[7], anno_split[6],
                     anno_split[3], anno_split[2]
                     ]
                ).astype(np.int)
                polys.append(poly)
                print('poly:', poly.shape)

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
                if height * width * (800 / float(f_img.shape[0])) < 16 * 16 and mode == "train":
                    continue
                boxes.append([x_ctr, y_ctr, width, height, angle])

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
            'image': f_impath,
            'boxes': gt_boxes,
            'flipped': False,
            'polys':polys,
            'gt_overlaps': overlaps,
            'seg_areas': seg_areas,
            'height': f_img.shape[0],
            'width': f_img.shape[1],
            'max_overlaps': max_overlaps,
            'rotated': True
        }
        im_infos.append(im_info)

    f_save_pkl = open(cache_pkl, 'wb')
    pickle.dump(im_infos, f_save_pkl)
    f_save_pkl.close()
    print("Save pickle done.")

    return im_infos

DATASET = {
    'AAA':get_irra_XXX,
    'BBB':get_irra_XXX,
    'CCC':get_irra_XXX,
    'DDD':get_irra_XXX,
    'EEE':get_irra_XXX,
}


_DEBUG = True
class IrraRotationDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "irra"
    )

    def __init__(self, database, use_difficult=False, transforms=None):
        # database:{dataset_name, dataset_dir}

        self.transforms = transforms

        self.annobase = []

        for dataset_name in database:
            if dataset_name in DATASET:
                self.annobase.extend(DATASET[dataset_name]('train', database[dataset_name], dataset_name))

        print('DATASET: Total samples from:', database.keys(), len(self.annobase))

        self.ids = [anno['image'][:-4] for anno in self.annobase]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = IrraRotationDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.mixup = T.MixUp(mix_ratio=0.1)
        self.num_samples = len(self.annobase)

    def __getitem__(self, index):

        # if _DEBUG:
        #     index = 0

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
        return IrraRotationDataset.CLASSES[class_id]

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