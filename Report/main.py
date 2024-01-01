import cv2
import numpy as np
import chaos

# 密钥初始值
x0 = 0.1
y0 = 0.5
z0 = 0.2

# 导入RGB图片
img = cv2.imread('.\input_image\image.jpg')

# 加密图像
encrypted_img = chaos.encrypt_image(img, x0, y0, z0)

# 解密图像
decrypted_img = chaos.decrypt_image(encrypted_img, x0, y0, z0)

# 显示图像
cv2.imshow('Original Image', img)
cv2.imshow('Encrypted Image', encrypted_img)
cv2.imshow('Decrypted Image', decrypted_img)
cv2.waitKey(0)

# # 保存加密后的图片
# cv2.imwrite('.\encrypted_image\encrypted_image.png', encrypted_img)
# # 保存解密后的图片
# cv2.imwrite('.\decrypted_image\decrypted_image.png', decrypted_img)