import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg
from predictor import ICDARDemo


def write_result_ICDAR(im_file, dets, result_dir):
    file_spl = im_file.split('/')
    file_name = file_spl[len(file_spl) - 1]
    file_name_arr = file_name.split(".")

    file_name_str = file_name_arr[0]

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    result = os.path.join(result_dir, "res_" + file_name_str + ".txt")

    return_bboxes = []

    if not os.path.isfile(result):
        os.mknod(result)
    result_file = open(result, "w")

    result_str = ""

    for idx in range(len(dets)):

        l, t, r, b = dets[idx].astype(np.int32)[0:4]

        rotated_pts = [
            [l, t], [r, t], [r, b], [l, b]
        ]

        #det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
        #          str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
        #          str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
        #          str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "\r\n"

        # rotated_pts = rotated_pts[:,0:2]

        # if (dets[idx][5] > threshold):
        # rotated_pts = over_bound_handle(rotated_pts, height, width)
        det_str = str(int(l)) + "," + str(int(t)) + "," + \
                  str(int(r)) + "," + str(int(b)) + "\r\n"

        result_str = result_str + det_str
        return_bboxes.append(dets[idx])

        # print rotated_pts.shape

    result_file.write(result_str)
    result_file.close()

    return return_bboxes


model_file = 'text_IC13'

result_dir = os.path.join('results', model_file)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

config_file = "../configs/e2e_faster_rcnn_R_50_C4_1x_ICDAR13_test.yaml"
print('config_file:', config_file)
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = ICDARDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image_dir = '../../datasets/ICDAR13/Challenge2_Test_Task12_Images/'

imlist = os.listdir(image_dir)

for image in imlist:
    impath = os.path.join(image_dir, image)
    print('image:', impath)
    img = cv2.imread(impath)
    predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
    # print('predictions:', predictions.shape)

    bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
    write_result_ICDAR(image[:-4], bboxes_np, result_dir)
    #cv2.imshow('win', predictions)
    #cv2.waitKey(0)