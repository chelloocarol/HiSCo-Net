"""Compute standard COCO bbox metrics from GT and prediction JSON files."""
import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def validate_inputs(ground_truth, predictions):
    """Reject mismatched IDs instead of silently dropping detections."""
    images = ground_truth['images']
    categories = ground_truth['categories']
    image_ids = {image['id'] for image in images}
    category_ids = {category['id'] for category in categories}
    annotation_ids = [annotation['id'] for annotation in ground_truth['annotations']]
    if not image_ids or not category_ids:
        raise ValueError('Ground truth must contain images and categories')
    if len(image_ids) != len(images) or len(category_ids) != len(categories):
        raise ValueError('Image and category IDs must be unique')
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError('Annotation IDs must be unique')
    if not isinstance(predictions, list):
        raise ValueError('Predictions must be a COCO detection list')
    for record in ground_truth['annotations'] + predictions:
        if record['image_id'] not in image_ids or record['category_id'] not in category_ids:
            raise ValueError('Unknown image_id or category_id; GT/prediction mappings must match')
        box = record['bbox']
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) < 0:
            raise ValueError('Expected finite [x, y, width, height] boxes with nonnegative extents')
    for record in predictions:
        if not math.isfinite(record['score']) or not 0 <= record['score'] <= 1:
            raise ValueError('Expected finite prediction scores in [0, 1]')


def coco_metrics(ground_truth, predictions):
    validate_inputs(ground_truth, predictions)
    gt = COCO()
    gt.dataset = deepcopy(ground_truth)
    gt.dataset.setdefault('info', {})
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[])
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR100']
    metrics = {name: float(evaluator.stats[index])
               for name, index in zip(names, [0, 1, 2, 3, 4, 5, 8])}
    metrics['per_class_AP'], metrics['per_class_AP50'] = {}, {}
    for index, category_id in enumerate(evaluator.params.catIds):
        name = gt.cats[category_id]['name']
        precision = evaluator.eval['precision'][:, :, index, 0, -1]
        for key, values in [('per_class_AP', precision), ('per_class_AP50', precision[0])]:
            valid = values[values > -1]
            metrics[key][name] = float(valid.mean()) if valid.size else None
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True, help='COCO ground-truth JSON')
    parser.add_argument('--pred', type=Path, required=True, help='COCO detection-list JSON')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--output', type=Path, required=True, help='New output summary JSON')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output path to preserve prior records')
    gt = json.loads(args.gt.read_text(encoding='utf-8-sig'))
    predictions = json.loads(args.pred.read_text(encoding='utf-8-sig'))
    metrics = coco_metrics(gt, predictions)
    summary = dict(split=args.split, evaluator='pycocotools.COCOeval',
                   gt=str(args.gt.resolve()), predictions=str(args.pred.resolve()),
                   iou_thresholds=[round(.5+.05*i, 2) for i in range(10)],
                   coco_max_dets=[1, 10, 100], n_images=len(gt['images']),
                   n_annotations=len(gt['annotations']), metrics=metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
