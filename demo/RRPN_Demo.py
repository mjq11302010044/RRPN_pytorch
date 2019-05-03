import os
import cv2
import numpy as np
from maskrcnn_benchmark.config import cfg
from demo.predictor import ICDARDemo, RRPNDemo
from maskrcnn_benchmark.utils.visualize import vis_image, write_result_ICDAR_RRPN2polys, zip_dir
from PIL import Image
import time


config_file = "./configs/rrpn/e2e_rrpn_R_50_C4_1x_ICDAR13_15_17_trial_again_test.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
# cfg.freeze()
# cfg.MODEL.WEIGHT = 'models/IC-13-15-17-Trial/model_0155000.pth'

result_dir = os.path.join('results', config_file.split('/')[-1].split('.')[0], cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0])

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)


coco_demo = RRPNDemo(
    cfg,
    min_image_size=1000,
    confidence_threshold=0.85,
)

dataset_name = 'IC15'

testing_dataset = {
    'IC13': {
        'testing_image_dir': '../datasets/ICDAR13/Challenge2_Test_Task12_Images',
        'test_vocal_dir': '../datasets/ICDAR13/ch2_test_vocabularies_per_image'
    },
    'IC15': {
        'testing_image_dir': '../datasets/ICDAR15/ch4_test_images',
        'test_vocal_dir': '../datasets/ICDAR15/ch4_test_vocabularies_per_image'
    },
}

image_dir = testing_dataset[dataset_name]['testing_image_dir']
vocab_dir = testing_dataset[dataset_name]['test_vocal_dir']

# load image and then run prediction
# image_dir = '../datasets/ICDAR13/Challenge2_Test_Task12_Images/'
imlist = os.listdir(image_dir)

print('************* META INFO ***************')
print('config_file:', config_file)
print('result_dir:', result_dir)
print('image_dir:', image_dir)
print('weights:', cfg.MODEL.WEIGHT)
print('***************************************')

vis = True

num_images = len(imlist)
cnt = 0

for image in imlist:
    impath = os.path.join(image_dir, image)
    # print('image:', impath)
    img = cv2.imread(impath)
    cnt += 1
    tic = time.time()
    predictions, bounding_boxes = coco_demo.run_on_opencv_image(img)
    toc = time.time()

    print('time cost:', str(toc - tic)[:6], '|', str(cnt) + '/' + str(num_images))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes_np = bounding_boxes.bbox.data.cpu().numpy()
    bboxes_np[:, 2:4] /= cfg.MODEL.RRPN.GT_BOX_MARGIN

    width, height = bounding_boxes.size

    if vis:
        pil_image = vis_image(Image.fromarray(img), bboxes_np)
        pil_image.show()
        time.sleep(10)
    write_result_ICDAR_RRPN2polys(image[:-4], bboxes_np, threshold=0.7, result_dir=result_dir, height=height, width=width)
    #im_file, dets, threshold, result_dir, height, width
    #cv2.imshow('win', predictions)
    #cv2.waitKey(0)

'''
if dataset_name == 'IC15':
    zipfilename = os.path.join(result_dir, 'submit_' + config_file.split('/')[-1].split('.')[0] + '_' + cfg.MODEL.WEIGHT.split('/')[-1].split('.')[0] + '.zip')
    if os.path.isfile(zipfilename):
        print('Zip file exists, removing it...')
        os.remove(zipfilename)
    zip_dir(result_dir, zipfilename)
    comm = 'curl -i -F "submissionFile=@' + zipfilename + '" http://127.0.0.1:8080/evaluate'
    # print(comm)
    print(os.popen(comm, 'r'))
else:
    pass
'''