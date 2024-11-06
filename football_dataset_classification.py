import os
import glob
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
import cv2
from utils import collate_fn


# class footballDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.transform = transform
#         self.images = []
#         self.labels = []
#         self.num_frames = []
#         self.file_names = []
#
#         matches = os.listdir(root)
#         for match in matches:
#             folder_path = os.path.join(root, match)
#             json_path, video_path = sorted(os.listdir(folder_path))  # xep file json trc
#             self.file_names.append(os.path.join(folder_path, json_path.replace(".json", "")))  # lay duong dan bo .json
#             with open(os.path.join(folder_path, json_path), "r") as json_file:
#                 json_data = json.load(json_file)
#             # print(json_data.keys())
#             self.num_frames.append(len(json_data["images"]))
#         # print(self.num_frames)
#
#     def __len__(self):
#         return sum(self.num_frames)
#
#     def __getitem__(self, index):
#         if index < self.num_frames[0]:
#             frame_id = index
#             video_id = 0
#         elif self.num_frames[0] <= index < self.num_frames[0] + self.num_frames[1]:
#             frame_id = index - self.num_frames[0]
#             video_id = 1
#         elif self.num_frames[0] + self.num_frames[1] <= index < self.num_frames[0] + self.num_frames[1] + \
#                 self.num_frames[2]:
#             frame_id = index - self.num_frames[0] - self.num_frames[1]
#             video_id = 2
#         else:
#             frame_id = index - self.num_frames[0] - self.num_frames[1] - self.num_frames[2]
#             video_id = 3
#         video_path = "{}.mp4".format(self.file_names[video_id])
#         json_path = "{}.json".format(self.file_names[video_id])
#         # print(self.num_frames, video_path, json_path)
#
#         # mo file mp4
#         cap = cv2.VideoCapture(video_path)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # doc chinh xac frame_id
#         flag, image = cap.read()
#         # cv2.imwrite("sample.png", image)
#
#         # mo dile json
#         with open(json_path, "r") as json_file:
#             json_data = json.load(json_file)
#         # print(json_data["annotations"])
#         bboxes = [anno["bbox"] for anno in json_data["annotations"] if
#                               anno["image_id"] - 1 == frame_id and anno["category_id"] == 4]    # lay bbox 10 ng trong 1 frame
#         jersey = [int(anno["attributes"]["jersey_number"])-1 for anno in json_data["annotations"] if
#                               anno["image_id"] - 1 == frame_id and anno["category_id"] == 4]
#         # print(jersey)
#         colors = [anno["attributes"]["team_jersey_color"] for anno in json_data["annotations"] if
#                               anno["image_id"] - 1 == frame_id and anno["category_id"] == 4]
#         # print(colors)
#         colors = [0 if color == "black" else 1 for color in colors]
#
#         # visualize xem bbox nao dung
#         # for anno in bbox:
#         #     # print(anno)
#         #     xcent, ycent, width, height = anno
#         #     xmin = int(xcent - width/2)
#         #     xmax = int(xcent + width/2)
#         #     ymin = int(ycent - height/2)
#         #     ymax = int(ycent + height/2)
#         #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = (0, 0, 255), thickness=2)
#         # cv2.imwrite("sample_1.png", image)
#
#         # for anno in bbbox:
#         #     xmin, ymin, width, height = anno
#         #     xmin = int(xmin)
#         #     ymin = int(ymin)
#         #     xmax = int(xmin + width)
#         #     ymax = int(ymin + height)
#         #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = (0, 0, 255), thickness=2)
#         # cv2.imwrite("sample_1.png", image)
#
#         cropped_images = [image[int(ymin):int(ymin+height), int(xmin):int(xmin+width), :] for (xmin, ymin, width, height) in bboxes]
#         # for i, cropped_image in enumerate(cropped_images):    # visualize thu anh crop
#         #     cv2.imwrite("{}.jpg".format(i), cropped_image)
#
#         if self.transform:
#             cropped_images = [self.transform(image) for image in cropped_images]
#
#         return cropped_images, jersey, colors
#
#
# # tach anh sang 1 ben, label sang 1 ben
# def collate_fn(batch_size):
#     # images, labels = zip(*batch_size)
#     images, labels, colors = zip(*batch_size)
#
#     # gop lai thanh list co batch_size*10 phan tu
#     final_images = []
#     for image in images:
#         final_images.extend(image)
#     final_images = torch.stack(final_images)
#
#     final_labels = []
#     final_labels.extend([label for label in labels])
#     final_labels = torch.IntTensor(final_labels)
#
#     final_colors = []
#     final_colors.extend([color for color in colors])
#     final_colors = torch.IntTensor(final_colors)
#
#     return final_images, final_labels, final_colors


class footballDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.num_frames = []
        self.file_names = []

        video_paths = list(glob.iglob("{}/*/*.mp4".format(root)))
        json_paths = list(glob.iglob("{}/*/*.json".format(root)))
        video_wo_ext = [video_path.replace(".mp4", "") for video_path in video_paths]
        json_wo_ext = [json_path.replace(".json", "") for json_path in json_paths]
        matches = list(set(video_wo_ext) & set(json_wo_ext))

        for match in matches:
            self.file_names.append(match)
            with open("{}.json".format(match), "r") as json_file:
                json_data = json.load(json_file)
            # print(json_data.keys())
            self.num_frames.append(len(json_data["images"]))
        # print(self.num_frames)

    def __len__(self):
        return sum(self.num_frames)

    def __getitem__(self, index):
        if index < self.num_frames[0]:
            frame_id = index
            video_id = 0
        elif self.num_frames[0] <= index < self.num_frames[0] + self.num_frames[1]:
            frame_id = index - self.num_frames[0]
            video_id = 1
        elif self.num_frames[0] + self.num_frames[1] <= index < self.num_frames[0] + self.num_frames[1] + \
                self.num_frames[2]:
            frame_id = index - self.num_frames[0] - self.num_frames[1]
            video_id = 2
        else:
            frame_id = index - self.num_frames[0] - self.num_frames[1] - self.num_frames[2]
            video_id = 3
        video_path = "{}.mp4".format(self.file_names[video_id])
        json_path = "{}.json".format(self.file_names[video_id])
        # print(self.num_frames, video_path, json_path)

        # mo file mp4
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)  # doc chinh xac frame_id
        flag, image = cap.read()
        # cv2.imwrite("sample.png", image)

        # mo dile json
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)
        # print(json_data["annotations"])
        bboxes = [anno["bbox"] for anno in json_data["annotations"] if
                              anno["image_id"] - 1 == frame_id and anno["category_id"] == 4]    # lay bbox 10 ng trong 1 frame
        jersey = [int(anno["attributes"]["jersey_number"])-1 for anno in json_data["annotations"] if
                              anno["image_id"] - 1 == frame_id and anno["category_id"] == 4]
        # print(jersey)
        colors = [anno["attributes"]["team_jersey_color"] for anno in json_data["annotations"] if
                              anno["image_id"] - 1 == frame_id and anno["category_id"] == 4]
        # print(colors)
        colors = [0 if color == "black" else 1 for color in colors]

        # visualize xem bbox nao dung
        # for anno in bbbox:
        #     xmin, ymin, width, height = anno
        #     xmin = int(xmin)
        #     ymin = int(ymin)
        #     xmax = int(xmin + width)
        #     ymax = int(ymin + height)
        #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color = (0, 0, 255), thickness=2)
        # cv2.imwrite("sample_1.png", image)

        cropped_images = [image[int(ymin):int(ymin+height), int(xmin):int(xmin+width), :] for (xmin, ymin, width, height) in bboxes]
        # for i, cropped_image in enumerate(cropped_images):    # visualize thu anh crop
        #     cv2.imwrite("{}.jpg".format(i), cropped_image)

        if self.transform:
            cropped_images = [self.transform(image) for image in cropped_images]

        return cropped_images, jersey, colors



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])

    path = "dataset/football_train"
    dataset = footballDataset(root=path, transform=transform)

    cropped_images, jersey, color = dataset[1506]
    for image in cropped_images:
        print(image.shape)
    #
    # params = {
    #     "batch_size": 2,
    #     "shuffle": True,
    #     "num_workers": 6,
    #     "drop_last": True,
    #     "collate_fn": collate_fn,       # dua [batch, so label, 3, 224, 224] ve [batch*10, 3, 224, 224]
    # }
    # dataloader = DataLoader(dataset, **params)
    #
    # model = models.resnet50(pretrained=True)
    # model.fc = nn.Linear(in_features=2048, out_features=10)
    # model.to(device)
    #
    # for images, labels, colors in dataloader:
    #     # print(images.shape)
    #     # print(labels)
    #     # print("--------------")
    #     images = images.to(device)
    #     output = model(images)
    #     print(output.shape)
    #     print(labels)
    #     print(colors)


