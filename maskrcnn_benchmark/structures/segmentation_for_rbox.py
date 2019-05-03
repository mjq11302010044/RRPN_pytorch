import torch
import pycocotools.mask as mask_utils

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
import math

class Mask(object):
    """
    This class is unfinished and not meant for use yet
    It is supposed to contain the mask for an object as
    a 2d tensor
    """

    def __init__(self, masks, size, mode):
        self.masks = masks
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 2
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        flip_idx = list(range(dim)[::-1])
        flipped_masks = self.masks.index_select(dim, flip_idx)
        return Mask(flipped_masks, self.size, self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]

        cropped_masks = self.masks[:, box[1]: box[3], box[0]: box[2]]
        return Mask(cropped_masks, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        pass


def rotate_pts(pts, angle, ctpt):

    # pts: [K*2]
    # angles: angle degree
    # print('pts:', pts)
    pt_x = pts[0::2]
    pt_y = pts[1::2]

    ptx_norm = pt_x - ctpt[0]
    pty_norm = pt_y - ctpt[1]

    angle_trans = -angle / 180 * 3.1415926535
    cosA = torch.cos(angle_trans)
    sinA = torch.sin(angle_trans)

    new_x_groups = ptx_norm * cosA - pty_norm * sinA
    new_y_groups = pty_norm * cosA + ptx_norm * sinA

    # print('pts:', pts)
    # print('cosA:', cosA, sinA, ptx_norm * cosA)
    # print('new_x_groups:', new_x_groups, new_x_groups)

    new_x_groups += ctpt[0]
    new_y_groups += ctpt[1]

    return torch.cat([new_x_groups[..., None], new_y_groups[..., None]], 1).reshape(-1)


class Polygons(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size, mode):
        # assert isinstance(polygons, list), '{}'.format(polygons)
        if isinstance(polygons, list):
            # print('init:', polygons)
            polygons = [torch.as_tensor(p, dtype=torch.float32) for p in polygons]
        elif isinstance(polygons, Polygons):
            polygons = polygons.polygons

        self.polygons = polygons

        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return Polygons(flipped_polygons, size=self.size, mode=self.mode)

    def crop(self, rbox):
        ctx, cty, w, h, theta =  rbox[0], rbox[1], rbox[2], rbox[3], rbox[4]

        # TODO chck if necessary
        w = max(w, 1)
        h = max(h, 1)

        cropped_polygons = []
        for poly in self.polygons:
            # rotate_poly = rotate_pts(poly, theta, [ctx, cty])
            p = poly.clone()
            p[0::2] = p[0::2] - (ctx - (w // 2))  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - (cty - (h // 2))  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return Polygons(cropped_polygons, size=(w, h), mode=self.mode)

    def rotate(self, angle, ctpt):
        rotated_polygons = []
        # print('self.polygons:', self.polygons)
        for poly in self.polygons:
            # rotate_poly = rotate_pts(poly, theta, [ctx, cty])
            p = poly.clone()
            p = rotate_pts(p, angle, ctpt)
            rotated_polygons.append(p)

        return Polygons(rotated_polygons, size=self.size, mode=self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        # print('ratios:', ratios)
        if math.fabs(ratios[0] - ratios[1]) < 0.01:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return Polygons(scaled_polys, size, mode=self.mode)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return Polygons(scaled_polygons, size=size, mode=self.mode)

    def convert(self, mode):
        width, height = self.size
        if mode == "mask":
            # print('self.polygons:', self.polygons)
            rles = mask_utils.frPyObjects(
                [p.numpy() for p in self.polygons], height, width
            )
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            mask = torch.from_numpy(mask)
            # TODO add squeeze?
            return mask

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_polygons={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class SegmentationMask(object):
    """
    This class stores the segmentations for all objects in the image
    """

    def __init__(self, polygons, size, mode=None):
        """
        Arguments:
            polygons: a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.
        """
        assert isinstance(polygons, list)

        self.polygons = [Polygons(p, size, mode) for p in polygons]
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped = []
        for polygon in self.polygons:
            flipped.append(polygon.transpose(method))
        return SegmentationMask(flipped, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2], box[3]
        cropped = []
        for polygon in self.polygons:
            cropped.append(polygon.crop(box))
        return SegmentationMask(cropped, size=(w, h), mode=self.mode)

    def rotate(self, angle, ctpt):
        rotated = []
        for polygon in self.polygons:
            rotated.append(polygon.rotate(angle, ctpt))
        return SegmentationMask(rotated, size=self.size, mode=self.mode)

    def resize(self, size, *args, **kwargs):
        scaled = []
        for polygon in self.polygons:
            scaled.append(polygon.resize(size, *args, **kwargs))
        return SegmentationMask(scaled, size=size, mode=self.mode)

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_polygons = [self.polygons[item]]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return SegmentationMask(selected_polygons, size=self.size, mode=self.mode)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
