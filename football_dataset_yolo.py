import os
import glob
import json
import shutil
import cv2
from pprint import pprint
import argparse

if __name__ == '__main__':
    root = "dataset/football_train"
    # root = "dataset/football_test"
    output_path = "football_dataset_yolo"
    is_train = True
    # is_train = False
    mode = "train" if is_train else "val"

    # if os.path.isdir(output_path):
    #     shutil.rmtree(output_path)
    #     os.makedirs(output_path)
    if not os.path.isdir(output_path) and is_train:
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "images", mode))
        os.makedirs(os.path.join(output_path, "labels", mode))
    elif not is_train:
        os.makedirs(os.path.join(output_path, "images", mode))
        os.makedirs(os.path.join(output_path, "labels", mode))

    video_paths = list(glob.iglob("{}/*/*.mp4".format(root)))
    anno_paths = list(glob.iglob("{}/*/*.json".format(root)))
    # print(anno_paths)

    video_wo_ext = [video_path.replace(".mp4", "") for video_path in video_paths]
    anno_wo_ext = [anno_path.replace(".json", "") for anno_path in anno_paths]

    paths = list(set(video_wo_ext) & set(anno_wo_ext))
    # print(paths)
    for idx, path in enumerate(paths):
        cap = cv2.VideoCapture("{}.mp4".format(path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(num_frames)
        with open("{}.json".format(path), "r") as json_file:
            json_data = json.load(json_file)
            # print(len(json_data["images"]))
        if num_frames != len(json_data["images"]):
            print("Something is wrong with game {}".format(path))
            paths.remove(path)

        # Lay kich thuoc frame
        width = json_data["images"][0]["width"]
        height = json_data["images"][0]["height"]

        # all_players = [{"image_id": obj["image_id"], "bbox": obj["bbox"]} for obj in json_data["annotations"] if
        #                obj["category_id"] == 4]
        # all_balls = [obj for obj in json_data["annotations"] if obj["category_id"] == 3]
        # pprint(all_players)
        # # print(len(all_players))
        # # print(all_players[10])
        # exit()
        all_objects = [{"image_id": obj["image_id"], "bbox": obj["bbox"], "category_id": obj["category_id"]} for obj in
                       json_data["annotations"] if obj["category_id"] in [3, 4]]

        frame_counter = 0
        while cap.isOpened():
            print(idx, frame_counter)
            flag, frame = cap.read()
            if not flag:
                break

            current_object = [obj for obj in all_objects if obj["image_id"] - 1 == frame_counter]
            # print(path)
            # pprint(current_object)
            # for obj in current_object:
            #     xmin, ymin, w, h = obj["bbox"]
            #     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmin+w), int(ymin+h)), (255, 0, 0), thickness=2)

            # ghi all frame vao file
            cv2.imwrite(os.path.join(output_path, "images", mode, "{}_{}.jpg".format(idx, frame_counter)), frame)

            with open(os.path.join(output_path, "labels", mode, "{}_{}.txt".format(idx, frame_counter)), "w") as fo:
                for obj in current_object:
                    xmin, ymin, w, h = obj["bbox"]

                    # chuan hoa
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




