import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
import cv2
import editdistance


def vis_masks(convex_polys, img_size):

    canvas = Image.new('RGB', img_size)
    pdraw = ImageDraw.Draw(canvas)
    for poly in convex_polys:
        # poly = poly.polygons
        for subpoly in poly.polygons:
            poly = subpoly.data.cpu().numpy().astype(np.int).tolist()
            # print('poly:', poly)
            pdraw.polygon(poly, fill=(255, 255, 255), outline=(0, 255, 0))

    del pdraw
    return canvas


def vis_image(img, boxes, cls_prob=None, mode=0, font_file='./fonts/ARIAL.TTF'):
    # img = cv2.imread(image_path)
    # cv2.setMouseCallback("image", trigger)
    font = ImageFont.truetype(font_file, 32)
    draw = ImageDraw.Draw(img)

    for idx in range(len(boxes)):
        cx, cy, w, h, angle = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]
        # need a box score larger than thresh

        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        # rotated_pts[rotated_pts <= 0] = 1
        # rotated_pts[rotated_pts > img.shape[1]] = img.shape[1] - 1

        if mode == 1:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])), fill=(0, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 255, 0))

        elif mode == 0:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 0, 255))

        elif mode == 2:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(255, 255, 0))

        if not cls_prob is None:
            score = cls_prob[idx]
            if idx == 0:
                draw.text((int(rotated_pts[0, 0]+30), int(rotated_pts[0, 1]+30)), str(idx), fill=(255, 255, 255, 128), font=font)

    # cv2.imshow("image", cv2.resize(img, (1024, 768)))
    # cv2.wait Key(0)
    del draw
    return img


#img, boxes, cls_prob=None, mode=0
def vis_image_with_words(img, boxes, word_labels, coder, voc_list, cls_prob=None, mode=1, font_file='./fonts/ARIAL.TTF'):
    # img = cv2.imread(image_path)
    # cv2.setMouseCallback("image", trigger)
    font = ImageFont.truetype(font_file, 32)
    draw = ImageDraw.Draw(img)
    # print('voc_list:', voc_list)
    for idx in range(len(boxes)):

        cx, cy, w, h, angle = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3], boxes[idx][4]
        # need a box score larger than thresh

        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        #rotated_pts[rotated_pts <= 0] = 1
        #rotated_pts[rotated_pts > img.shape[1]] = img.shape[1] - 1

        # rotated_pts = np.array(boxes[idx]).reshape(-1, 2)

        if mode == 1:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])), fill=(0, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 255, 0))

        elif mode == 0:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(0, 0, 255))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(0, 0, 255))

        elif mode == 2:
            draw.line((int(rotated_pts[0, 0]), int(rotated_pts[0, 1]), int(rotated_pts[1, 0]), int(rotated_pts[1, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[1, 0]), int(rotated_pts[1, 1]), int(rotated_pts[2, 0]), int(rotated_pts[2, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[2, 0]), int(rotated_pts[2, 1]), int(rotated_pts[3, 0]), int(rotated_pts[3, 1])),
                      fill=(255, 255, 0))
            draw.line((int(rotated_pts[3, 0]), int(rotated_pts[3, 1]), int(rotated_pts[0, 0]), int(rotated_pts[0, 1])),
                      fill=(255, 255, 0))
        if not cls_prob is None:
            score = cls_prob[idx]
            draw.text((int(rotated_pts[0, 0]), int(rotated_pts[0, 1])), str(score),
                        font, (255, 0, 122))
        context_word = ''
        if not voc_list is None:
            # print('voc_list:', voc_list)
            res_word = word_labels[idx]
            edisarr = []
            min_dix_arg = -1
            min_dis = 1000000000000000
            for i_voc in range(len(voc_list)):
                dis = editdistance.eval(res_word.upper(), voc_list[i_voc].replace('\ufeff', '').replace('\n', '').upper())
                edisarr.append(dis)
                if dis < min_dis:
                    min_dis = dis
                    min_dix_arg = i_voc
            # edisarr = np.array(edisarr, np.float32)
            # min_dix_arg = np.argmin(edisarr, axis=0)
            context_word = voc_list[min_dix_arg]

        if not word_labels is None:
            word_label = word_labels[idx] if voc_list is None else context_word
            # print(word_label)
            x, y = int(rotated_pts[0, 0]+3), int(rotated_pts[0, 1]-11)
            draw.rectangle(((x-5, y-30), (x + 22 * len(word_label) + 3, y+10)), fill='yellow', outline=None)
            # cv2.rectangle(img, (x-5, y-30), (x + 20 * len(word_label) + 3, y+10), (0, 255, 255), thickness=-1)
            draw.text((x+10, y-25), word_label, fill=(255, 255, 255, 128), font=font) #(255, 0, 255)

    del draw
    return img


def write_result_ICDAR_FRCN_ltrb(im_file, dets, result_dir):
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

        det_str = str(int(l)) + "," + str(int(t)) + "," + \
                  str(int(r)) + "," + str(int(b)) + "\r\n"

        result_str = result_str + det_str
        return_bboxes.append(dets[idx])

    result_file.write(result_str)
    result_file.close()

    return return_bboxes


def write_result_ICDAR_RRPN2polys(im_file, dets, threshold, result_dir, height, width):
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
        cx, cy, w, h, angle = dets[idx][0:5]
        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        # angle = angle * 0.45

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        # if angle != 0:
        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        # else :
        #	cos_cita = 1
        #	sin_cita = 0

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        # print im
        # print im.shape
        #			im = im.transpose(2,0,1)

        # det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
        #          str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
        #          str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
        #          str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "\r\n"

        # rotated_pts = rotated_pts[:,0:2]


        # if (dets[idx][5] > threshold):
        rotated_pts = over_bound_handle(rotated_pts, height, width)
        det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
                  str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
                  str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
                  str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) + "\r\n"

        result_str = result_str + det_str
        return_bboxes.append(dets[idx])

        # print rotated_pts.shape

    result_file.write(result_str)
    result_file.close()

    return return_bboxes


def write_result_ICDAR_MASKRRPN2polys(im_file, polys, threshold, result_dir, height, width):
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

    for idx in range(len(polys)):

        poly_str = ''
        # print('poly:', polys[idx])
        poly = polys[idx].reshape(-1, 2)

        # if (dets[idx][5] > threshold):
        poly = over_bound_handle(poly, height, width).reshape(-1)

        # result_str = ''
        for i in range(poly.shape[0]):
            cood = str(int(poly[i]))
            poly_str += (cood + ',')

        result_str += (poly_str[:-1] + "\r\n")

        # result_str = result_str + det_str
        return_bboxes.append(polys[idx])

        # print rotated_pts.shape

    result_file.write(result_str)
    result_file.close()

    return return_bboxes


def write_result_ICDAR_RRPN2polys_with_words(im_file, dets, word_labels, voc_list, result_dir, height, width):
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
        cx, cy, w, h, angle = dets[idx][0:5]
        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        # angle = angle * 0.45

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        # if angle != 0:
        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        # else :
        #	cos_cita = 1
        #	sin_cita = 0

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)
        context_word = ''
        if not voc_list is None:
            # print('voc_list:', voc_list)
            res_word = word_labels[idx]
            edisarr = []
            min_dix_arg = -1
            min_dis = 1000000000000000
            for i_voc in range(len(voc_list)):
                dis = editdistance.eval(res_word.upper(), voc_list[i_voc].replace('\ufeff', '').replace('\n', '').upper())
                edisarr.append(dis)
                if dis < min_dis:
                    min_dis = dis
                    min_dix_arg = i_voc

            context_word = voc_list[min_dix_arg]

        write_prediction = word_labels[idx] if voc_list is None else context_word

        rotated_pts = over_bound_handle(rotated_pts, height, width)
        det_str = str(int(rotated_pts[0][0])) + "," + str(int(rotated_pts[0][1])) + "," + \
                  str(int(rotated_pts[1][0])) + "," + str(int(rotated_pts[1][1])) + "," + \
                  str(int(rotated_pts[2][0])) + "," + str(int(rotated_pts[2][1])) + "," + \
                  str(int(rotated_pts[3][0])) + "," + str(int(rotated_pts[3][1])) +  "," + write_prediction + "\r\n"

        result_str = result_str + det_str
        return_bboxes.append(dets[idx])

    result_file.write(result_str)
    result_file.close()

    return return_bboxes


def write_result_ICDAR_RRPN2ltrb(im_file, dets, threshold, result_dir, height, width):
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
        cx, cy, h, w, angle = dets[idx][0:5]
        lt = [cx - w / 2, cy - h / 2, 1]
        rt = [cx + w / 2, cy - h / 2, 1]
        lb = [cx - w / 2, cy + h / 2, 1]
        rb = [cx + w / 2, cy + h / 2, 1]

        pts = []

        # angle = angle * 0.45

        pts.append(lt)
        pts.append(rt)
        pts.append(rb)
        pts.append(lb)

        angle = -angle

        # if angle != 0:
        cos_cita = np.cos(np.pi / 180 * angle)
        sin_cita = np.sin(np.pi / 180 * angle)

        # else :
        #	cos_cita = 1
        #	sin_cita = 0

        M0 = np.array([[1, 0, 0], [0, 1, 0], [-cx, -cy, 1]])
        M1 = np.array([[cos_cita, sin_cita, 0], [-sin_cita, cos_cita, 0], [0, 0, 1]])
        M2 = np.array([[1, 0, 0], [0, 1, 0], [cx, cy, 1]])
        rotation_matrix = M0.dot(M1).dot(M2)

        rotated_pts = np.dot(np.array(pts), rotation_matrix)

        rotated_pts = over_bound_handle(rotated_pts, height, width)

        left = min(int(rotated_pts[0][0]), int(rotated_pts[1][0]), int(rotated_pts[2][0]), int(rotated_pts[3][0]))
        top = min(int(rotated_pts[0][1]), int(rotated_pts[1][1]), int(rotated_pts[2][1]), int(rotated_pts[3][1]))
        right = max(int(rotated_pts[0][0]), int(rotated_pts[1][0]), int(rotated_pts[2][0]), int(rotated_pts[3][0]))
        bottom = max(int(rotated_pts[0][1]), int(rotated_pts[1][1]), int(rotated_pts[2][1]), int(rotated_pts[3][1]))

        det_str = str(int(left)) + "," + str(int(top)) + "," + \
                  str(int(right)) + "," + str(int(bottom)) + "\r\n"

        result_str = result_str + det_str
        return_bboxes.append(dets[idx])

        # print rotated_pts.shape

    result_file.write(result_str)
    result_file.close()

    return return_bboxes


def over_bound_handle(pts, img_height, img_width):

    pts[np.where(pts < 0)] = 1

    pts[np.where(pts[:,0] > img_width), 0] = img_width-1
    pts[np.where(pts[:,1] > img_height), 1] = img_height-1

    return pts


def zip_dir(dirname, zipfilename):
    filelist = []
    if os.path.isfile(dirname):
        filelist.append(dirname)
    else:
        for root, dirs, files in os.walk(dirname):
            for name in files:
                filelist.append(os.path.join(root, name))

    zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
    print('Compressing ' + zipfilename + '..')
    for tar in filelist:
        arcname = tar[len(dirname):]
        # print arcname
        zf.write(tar, arcname)
    zf.close()
