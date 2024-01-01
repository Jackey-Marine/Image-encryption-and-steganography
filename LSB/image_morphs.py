from PIL import Image
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
import LSBSteg
import cv2

def image_selector():                           #returns path to selected image
    path = "NULL"
    root = tk.Tk()
    root.withdraw()                             # we don't want a full GUI, so keep the root window from appearing
    path = filedialog.askopenfilename()         # show an "Open" dialog box and return the path to the selected file
    if path!="NULL":
        print("Image loaded!") 
    else:
        print("Error Image not loaded!")
    return path

def interpolate_image(image_path, gain=8):
    # 读取图像
    print(f'The image morphs to {gain} times the original.')
    I = np.array(Image.open(image_path))

    # 获取图像尺寸
    IH, IW, ID = I.shape

    # 计算亚像素图像大小
    GIH = round(IH * gain)
    GIW = round(IW * gain)

    # 初始化亚像素图像
    GI = np.zeros((GIH, GIW, ID), dtype=np.uint8)

    # 像素预处理
    IPRE = np.zeros((IH + 2, IW + 2, ID), dtype=np.uint8)
    IPRE[1:IH + 1, 1:IW + 1, :] = I

    # 为填充出来的像素赋值
    IPRE[0, 1:IW + 1, :] = I[0, :, :]
    IPRE[IH + 1, 1:IW + 1, :] = I[IH - 1, :, :]

    IPRE[1:IH + 1, 0, :] = I[:, 0, :]
    IPRE[1:IH + 1, IW + 1, :] = I[:, IW - 1, :]

    IPRE[0, 0, :] = I[0, 0, :]
    IPRE[0, IW + 1, :] = I[0, IW - 1, :]

    IPRE[IH + 1, 0, :] = I[IH - 1, 0, :]
    IPRE[IH + 1, IW + 1, :] = I[IH - 1, IW - 1, :]

    # 亚像素图像像素值计算
    for gj in range(GIW):
        for gi in range(GIH):
            ii = gi / gain
            jj = gj / gain
            i = int(ii)
            j = int(jj)

            # 新增点的坐标
            u = ii - i
            v = jj - j
            i = i + 1
            j = j + 1  # 更新

            # 根据上述公式计算亚像素值
            GI[gi, gj, :] = (1 - u) * (1 - v) * IPRE[i, j, :] + (1 - u) * v * IPRE[i, j + 1, :] + u * (1 - v) * IPRE[i + 1, j, :] + u * v * IPRE[i + 1, j + 1, :]

    # 输出图像
    output_image = Image.fromarray(GI)
    return output_image

#program main
if (__name__ == "__main__"):
    # file_path = image_selector()
    # start_time = time.time()
    # interpolated_image = interpolate_image(file_path, gain=8)
    # end_time = time.time()
    # print(f'Interpolate image time:{end_time - start_time}s.')
    # interpolated_image.save('Interpolate_image.jpg')
    
    # try:
    #     im = Image.open(file_path)
    #     im.save('./newData.png')
    #     print('The image conversion from JPG to PNG is successful')
    # except FileNotFoundError:
    #     print('Provided image path is not found')
    
    #LSB
    # encoding
    steg = LSBSteg.LSBSteg(cv2.imread("host.png"))
    new_im = steg.encode_image(cv2.imread("in.png"))
    cv2.imwrite("Steg.png", new_im)

    # decoding
    steg = LSBSteg.LSBSteg(cv2.imread("Steg.png"))
    orig_im = steg.decode_image()
    cv2.imwrite("recovered.png", orig_im)
