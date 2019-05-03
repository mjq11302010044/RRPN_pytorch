// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "RROIAlign.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
  m.def("roi_align_forward", &ROIAlign_forward, "ROIAlign_forward");
  m.def("roi_align_backward", &ROIAlign_backward, "ROIAlign_backward");
  m.def("roi_pool_forward", &ROIPool_forward, "ROIPool_forward");
  m.def("roi_pool_backward", &ROIPool_backward, "ROIPool_backward");

  m.def("rroi_align_forward", &RROIAlign_forward, "RROIAlign_forward");
  m.def("rroi_align_backward", &RROIAlign_backward, "RROIAlign_backward");

}
