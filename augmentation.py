#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 19:07:49 2021

@author: gitanshwadhwa
"""
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import cv2
import glob

all_img = glob.glob(r'./training/valid/nodr_s/*.jpeg')
other_dir = r'./training/valid/nodr'


for img_id, img_path in enumerate(all_img):

    img = cv2.imread(img_path)

    #rotating
#    rotate=iaa.Affine(rotate=(-50, 30))
#    rotated_image=rotate.augment_image(img)
#    cv2.imwrite(f'{other_dir}/{img_id}1.jpeg',rotated_image)

    crop = iaa.Crop(percent=(0, 0.3)) # crop image
    crop_image=crop.augment_image(img)
    cv2.imwrite(f'{other_dir}/{img_id}2.jpeg',crop_image)

    #shearing image
#    shear = iaa.Affine(shear=(0,40))
#    shear_image=shear.augment_image(img)
#    cv2.imwrite(f'{other_dir}/{img_id}3.jpeg',shear_image)

    #flipping image horizontally
    flip_hr=iaa.Fliplr(p=1.0)
    flip_hr_image= flip_hr.augment_image(img)
    cv2.imwrite(f'{other_dir}/{img_id}4.jpeg',flip_hr_image)

    #flipping image vertically
    flip_vr=iaa.Flipud(p=1.0)
    flip_vr_image= flip_vr.augment_image(img)
    cv2.imwrite(f'{other_dir}/{img_id}5.jpeg',flip_vr_image)

     #brightness change
#    contrast=iaa.GammaContrast(gamma=2.0)
#    contrast_image =contrast.augment_image(img)
#    cv2.imwrite(f'{other_dir}/{img_id}6.jpeg',contrast_image)

    #sacling image
    scale_im=iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
    scale_image =scale_im.augment_image(img)
    cv2.imwrite(f'{other_dir}/{img_id}7.jpeg',scale_image)
