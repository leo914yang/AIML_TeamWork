import cv2
from glob import glob
import os
import numpy as np


#1 圖片讀取 (多資料夾底下全部目錄, 圖檔)
def image_path_AllFolder(file_path):
    path_arr = []
    for dirpath, dirname, filenames in os.walk(file_path):
        for filename in filenames:
            path_arr.append(os.path.join(dirpath, filename))
    return path_arr
    # return dirname (目錄)
            

#2 圖片位置 (單資料夾底下全部圖檔)
def image_path(file_path):
    return glob(file_path + '/*')


#3 圖片讀取/高斯模糊/雙邊濾波/翻轉/存檔
def image_read(file_path, color=0, gaussian=0, bilateral=0, flip=0, flip_direction=-1, save=0, save_path='c:/test/', save_name='img'):
    count = 0
    for i in file_path:
        count += 1
        img_type = i.split('.')[1]
        img = cv2.imread(i, color)
        if gaussian:
            img = cv2.GaussianBlur(img, (7, 7), 0)
        if bilateral:
            img = cv2.bilateralFilter(img, 7, 31, 31)
        if flip:
            img = cv2.flip(img, flip_direction) 
            # flip 1 水平翻轉, 0 垂直翻轉, -1 水平+垂直翻轉
        if save:
            cv2.imwrite(save_path + save_name + f'-{count:03d}.' + img_type, img)
        cv2.imshow('image', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


#1 test
#print(image_path_AllFolder(file := input('Input path:')))

#3 test
print(image_read(image_path(file := input('input image path: ')), color=1, gaussian=1, flip=1, save=1))

