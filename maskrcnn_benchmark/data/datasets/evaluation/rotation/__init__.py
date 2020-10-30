import logging

from .rotation_eval import do_rotation_evaluation


def rotation_evaluation(dataset, predictions, output_folder, box_only, **kwargs):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    return do_rotation_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )

