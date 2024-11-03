
import argparse
import csv
import json
import os
from collections import defaultdict

from pycocotools.coco import COCO

from coco.coco_eval import CocoEvaluator


def get_args_parser():
    parser = argparse.ArgumentParser('ClipsEval', add_help=False)
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--result_file', type=str)
    return parser


def main(args):
    # Check if the annotation file and result file exist
    if not os.path.isfile(args.annotation_file):
        print(f"Error: Annotation file '{args.annotation_file}' does not exist.")
        return

    if not os.path.isfile(args.result_file):
        print(f"Error: Result file '{args.result_file}' does not exist.")
        return

    # Read result file and construct a nested dictionary
    results = defaultdict(lambda: defaultdict(dict))
    with open(args.result_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            experiment = row['experiment']
            frame_number = int(row['frame_number'])
            image_id = int(row['image_id'])
            box = json.loads(row['box'])  # Assuming box is saved as a JSON list in the CSV file
            score = float(row['score'])
            label = int(row['label'])

            # Store results in the nested dictionary structure
            if image_id not in results[experiment][frame_number]:
                results[experiment][frame_number][image_id] = {
                    "boxes": [],
                    "scores": [],
                    "labels": []
                }
            results[experiment][frame_number][image_id]["boxes"].append(box)
            results[experiment][frame_number][image_id]["scores"].append(score)
            results[experiment][frame_number][image_id]["labels"].append(label)

    coco_evaluators_per_experiment_and_timestamp = defaultdict(lambda: defaultdict(dict))
    coco = COCO(args.annotation_file)

    stats = {}

    for experiment, experiment_results in results.items():
        for frame_number, r in experiment_results.items():
            if frame_number not in coco_evaluators_per_experiment_and_timestamp[experiment]:
                coco_evaluators_per_experiment_and_timestamp[experiment][frame_number] = CocoEvaluator(coco)

            ce = coco_evaluators_per_experiment_and_timestamp[experiment][frame_number]
            ce.update(r)
            ce.accumulate()
            ce.summarize()

            stats[f'coco_eval_bbox_{experiment}_{frame_number}'] = ce.coco_eval['bbox'].stats.tolist()

    flat_stats_result = flat_stats(stats)

    print(json.dumps(flat_stats_result, indent=2))


def flat_stats(stats_dict):
    prefix_dict = {'coco_eval_bbox': [
        'AP_IoU_50_to_95_area_all_maxDet_100',
        'AP_IoU_50_area_all_maxDet_100',
        'AP_IoU_75_area_all_maxDet_100',
        'AP_IoU_50_to_95_area_small_maxDet_100',
        'AP_IoU_50_to_95_area_medium_maxDet_100',
        'AP_IoU_50_to_95_area_large_maxDet_100',
        'AR_IoU_50_to_95_area_all_maxDet_1',
        'AR_IoU_50_to_95_area_all_maxDet_10',
        'AR_IoU_50_to_95_area_all_maxDet_100',
        'AR_IoU_50_to_95_area_small_maxDet_100',
        'AR_IoU_50_to_95_area_medium_maxDet_100',
        'AR_IoU_50_to_95_area_large_maxDet_100'
    ]
    }
    flattened_test_stats = _flatten_stats(stats_dict, prefix_dict)
    return flattened_test_stats


def _flatten_stats(stats, prefix_dict=None):
    """
    Flattens list values in the test_stats dictionary.

    Args:
    - test_stats (dict): Dictionary containing test statistics.
    - prefix_dict (dict, optional): Dictionary containing prefix for specific keys.

    Returns:
    - dict: Flattened test statistics with new keys.
    """
    flattened_stats = {}

    for k, v in stats.items():
        if isinstance(v, list) and all(isinstance(i, (int, float)) for i in v):
            # If the value is a list of numeric values
            for idx, num in enumerate(v):
                # Determine the prefix to use: either from prefix_dict or fallback to the index
                if prefix_dict and (k in prefix_dict
                                    or any(k.startswith(prefix_key) for prefix_key in prefix_dict.keys())):
                    prefix = None

                    for prefix_key in prefix_dict.keys():
                        if k.startswith(prefix_key):
                            prefix = prefix_dict[prefix_key]
                            break

                    if prefix:
                        if isinstance(prefix, list) and len(prefix) > idx:
                            prefix = prefix[idx]
                        else:
                            prefix = f'{prefix}_{idx}'
                else:
                    prefix = idx  # Fallback to the index if no custom prefix
                flattened_key = f'{k}_{prefix}'
                flattened_stats[flattened_key] = num
        else:
            # If the value is not a list of numeric values, just use the original key-value
            flattened_stats[f'{k}'] = v
    return flattened_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ClipsEval script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

