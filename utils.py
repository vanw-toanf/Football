import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torchvision.models as models


def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20,20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i,j] > threshold else "black"
            plt.text(j, i, cm[i,j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

def get_arg():
    parse = argparse.ArgumentParser(description='Football Classification')
    parse.add_argument('-p', '--data_path', type=str, default="dataset/football_train")
    parse.add_argument('-b', '--batch_size', type=int, default=2)
    parse.add_argument('-e', '--epochs', type=int, default=100)
    parse.add_argument('-l', '--lr', type=float, default=1e-3)
    parse.add_argument('-s', '--image_size', type=int, default=224)
    parse.add_argument('-c', '--checkpoint_path', type=str, default=None)
    parse.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard")
    parse.add_argument('-r', '--train_path', type=str, default="trained_models")
    args = parse.parse_args()
    return args

def collate_fn(batch_size):
    images, labels, colors = zip(*batch_size)

    final_images = []
    for image in images:
        final_images.extend(image)
    final_images = torch.stack(final_images)

    final_labels = []
    for label in labels:
        final_labels.extend(label)
    final_labels = torch.IntTensor(final_labels)

    final_colors = []
    for color in colors:
        final_colors.extend(color)
    final_colors = torch.IntTensor(final_colors)

    return final_images, final_labels, final_colors
