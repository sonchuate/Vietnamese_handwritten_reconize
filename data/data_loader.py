import random as rd
import cv2
from common import orcPreprocess, encodeText, pad_listints
import numpy as np

def divide_data_gt():
    """divide train_gt.txt into training_gt.txt and valid_gt.txt"""

    with open('training_gt.txt', 'w', encoding='UTF-8') as wf1:
        with open('valid_gt.txt', 'w', encoding='UTF-8') as wf2:
            f = open("train_gt.txt","r", encoding='UTF-8')
            for l in f:
                if rd.random() < 0.02:
                    wf2.write(l)
                else:
                    wf1.write(l)
            f.close()


#data loader

def data_loader(prob = 10000/103000, path_gt = "data/train_gt.txt"):
    f = open(path_gt,"r", encoding='UTF-8')

    images = []
    labels =[]
    file_names = []
    with open(path_gt, 'r', encoding='utf8') as f:
        s = f.read()
        s = s.split("\n")
        i = 0#to limit number of image
        for l in s:
            if np. random. uniform (0, 1) > prob:
                continue
            [file_name, label] = l.split('\t')
            img_path = "vietnamese_hcr/raw/new_train/"+file_name
            img = cv2.imread(img_path)
            img = orcPreprocess(img, default_fixed_size = (800, 100), default_filter_size = (2, 2))
            images.append(img)
            labels.append(label)
            file_names.append(file_name)
            # i+=1
            # if i == 100:
            #   break

    return images, labels, file_names
