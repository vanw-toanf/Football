"""
    @author: Van Toan <damtoan321@gmail.com>
"""
import os
import glob
import json
import shutil
import cv2
from pprint import pprint
import argparse

def get_arg():
    parse = argparse.ArgumentParser(description='Football')
    parse.add_argument('-m', '--mode', type=str, default="train", help="train or test")

    args = parse.parse_args()
    return args

if __name__ == '__main__':
    args = get_arg()
    root = "dataset/football_{}".format(args.mode)
    output_path = "football_dataset_yolo"
    is_train = True if args.mode == "train" else False
    mode = "train" if is_train else "val"

    if not os.path.isdir(output_path) and is_train:
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "images", mode))
        os.makedirs(os.path.join(output_path, "labels", mode))
    elif not is_train:
        os.makedirs(os.path.join(output_path, "images", mode))
        os.makedirs(os.path.join(output_path, "labels", mode))

    video_paths = list(glob.iglob("{}/*/*.mp4".format(root)))
    anno_paths = list(glob.iglob("{}/*/*.json".format(root)))

    video_wo_ext = [video_path.replace(".mp4", "") for video_path in video_paths]
    anno_wo_ext = [anno_path.replace(".json", "") for anno_path in anno_paths]

    paths = list(set(video_wo_ext) & set(anno_wo_ext))
    for idx, path in enumerate(paths):
        cap = cv2.VideoCapture("{}.mp4".format(path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with open("{}.json".format(path), "r") as json_file:
            json_data = json.load(json_file)
        if num_frames != len(json_data["images"]):
            print("Something is wrong with game {}".format(path))
            paths.remove(path)

        # Size frame
        width = json_data["images"][0]["width"]
        height = json_data["images"][0]["height"]

        all_objects = [{"image_id": obj["image_id"], "bbox": obj["bbox"], "category_id": obj["category_id"]} for obj in
                       json_data["annotations"] if obj["category_id"] in [3, 4]]

        frame_counter = 0
        while cap.isOpened():
            print(idx, frame_counter)
            flag, frame = cap.read()
            if not flag:
                break

            current_object = [obj for obj in all_objects if obj["image_id"] - 1 == frame_counter]

            # Write frame
            cv2.imwrite(os.path.join(output_path, "images", mode, "{}_{}.jpg".format(idx, frame_counter)), frame)

            with open(os.path.join(output_path, "labels", mode, "{}_{}.txt".format(idx, frame_counter)), "w") as fo:
                for obj in current_object:
                    xmin, ymin, w, h = obj["bbox"]

                    # Normalize
                    xmin /= width
                    ymin /= height
                    w /= width
                    h /= height

                    if obj["category_id"] == 4:
                        category = 0
                    else:
                        category = 1
                    fo.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(category, xmin+w/2, ymin+h/2, w, h))

            frame_counter += 1
