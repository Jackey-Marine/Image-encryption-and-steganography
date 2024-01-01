import cv2
import numpy as np

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



def generate_chua_deriv(steps, x0, y0, z0, alpha=15.6, beta=28, m0=-1.143, m1=-0.714, dt=0.01):
    print("Generating chua sequences...")

    # Chua 参数和初始值设定
    x = np.empty((steps + 1,))
    y = np.empty((steps + 1,))
    z = np.empty((steps + 1,))

    # 初始值
    x[0], y[0], z[0] = x0, y0, z0

    # 定义 h(x) 函数
    h = lambda x: m1*x + 0.5*(m0-m1)*(abs(x+1)-abs(x-1))

    # Chua 方程迭代
    for i in range(steps):
        dx = alpha * (y[i] - x[i] - h(x[i]))
        dy = x[i] - y[i] + z[i]
        dz = -beta * y[i]
        x[i + 1] = x[i] + dx * dt
        y[i + 1] = y[i] + dy * dt
        z[i + 1] = z[i] + dz * dt

    print("Finished generate chua sequences.")
    print(x)
    print(y)
    print(z)
    print("------------------------")

    return x, y, z

def generate_lorenz_sequences(steps, x0, y0, z0, sigma=10.0, rho=28.0, beta=8/3, dt=0.01):
    
    print("Generating lorenz sequences...")

    # Lorenz参数和初始值设定
    x = np.empty((steps + 1,))
    y = np.empty((steps + 1,))
    z = np.empty((steps + 1,))

    # 初始值
    x[0], y[0], z[0] = x0, y0, z0

    # Lorenz方程迭代
    for i in range(steps):
        dx = sigma * (y[i] - x[i])
        dy = x[i] * (rho - z[i]) - y[i]
        dz = x[i] * y[i] - beta * z[i]
        x[i + 1] = x[i] + dx * dt
        y[i + 1] = y[i] + dy * dt
        z[i + 1] = z[i] + dz * dt

    print("Finished generate lorenz sequences.")
    print(x)
    print(y)
    print(z)
    print("------------------------")

    return x, y, z



def encrypt_image(image, x0, y0, z0):

    # 获取图像的尺寸
    rows, cols, _ = image.shape

    # 计算像素数量
    pixels_num = rows * cols

    # 生成混沌序列
    pos_x, pos_y, pos_z = generate_chua_deriv(pixels_num,x0,y0,z0)
    keys_x, keys_y, keys_z = generate_lorenz_sequences(pixels_num,x0,y0,z0)

    pos_x = (normalize(pos_x) * rows).astype(np.int) % rows
    pos_y = (normalize(pos_y) * cols).astype(np.int) % cols

    print("Encrypting image position...")
    
    # 创建一个空的像素位置加密后的图像
    encrypted_image_temp = np.zeros_like(image)

    # 根据混沌序列中的置换关系，重新排列图像的像素位置
    for old_row in range(rows):
        for old_col in range(cols):
            new_row = pos_x[old_row]
            new_col = pos_y[old_col]
            encrypted_image_temp[new_row][new_col] = image[old_row][old_col]

    print("Encrypting image rgb...")

    # 将图像分解为RGB通道
    r, g, b = cv2.split(encrypted_image_temp)

    # 归一化处理
    normalized_keys_x = (normalize(keys_x) * 255).astype(np.uint8)
    normalized_keys_y = (normalize(keys_y) * 255).astype(np.uint8)
    normalized_keys_z = (normalize(keys_z) * 255).astype(np.uint8)

    # 对RGB通道进行异或加密操作
    encrypted_r = np.bitwise_xor(r, normalized_keys_x[:r.size].reshape(r.shape))
    encrypted_g = np.bitwise_xor(g, normalized_keys_y[:g.size].reshape(g.shape))
    encrypted_b = np.bitwise_xor(b, normalized_keys_z[:b.size].reshape(b.shape))

    # 重新组合RGB通道
    encrypted_img = cv2.merge([encrypted_r, encrypted_g, encrypted_b])

    print("Finished encryption.")

    return encrypted_img



def decrypt_image(encrypted_img, x0, y0, z0):

    # 获取图像的尺寸
    rows, cols, _ = encrypted_img.shape

    # 计算像素数量
    pixels_num = rows * cols

    # 生成混沌序列
    pos_x, pos_y, pos_z = generate_chua_deriv(pixels_num,x0,y0,z0)
    keys_x, keys_y, keys_z = generate_lorenz_sequences(pixels_num,x0,y0,z0)

    pos_x = (normalize(pos_x) * rows).astype(np.int) % rows
    pos_y = (normalize(pos_y) * cols).astype(np.int) % cols

    print("Decrypting image rgb...")

    # 将图像分解为RGB通道
    encrypted_r, encrypted_g, encrypted_b = cv2.split(encrypted_img)

    # 归一化处理
    normalized_keys_x = (normalize(keys_x) * 255).astype(np.uint8)
    normalized_keys_y = (normalize(keys_y) * 255).astype(np.uint8)
    normalized_keys_z = (normalize(keys_z) * 255).astype(np.uint8)

    # 使用之前的秘钥进行XOR操作，恢复RGB通道
    decrypted_r = np.bitwise_xor(encrypted_r, normalized_keys_x[:encrypted_r.size].reshape(encrypted_r.shape))
    decrypted_g = np.bitwise_xor(encrypted_g, normalized_keys_y[:encrypted_g.size].reshape(encrypted_g.shape))
    decrypted_b = np.bitwise_xor(encrypted_b, normalized_keys_z[:encrypted_b.size].reshape(encrypted_b.shape))

    # 重新组合RGB通道
    decrypted_img_temp = cv2.merge([decrypted_r, decrypted_g, decrypted_b])

    print("Finished image position...")

    # 创建一个空的像素位置解密后的图像
    decrypted_img = np.zeros_like(decrypted_img_temp)

    # 根据混沌序列中的置换关系，重新排列图像的像素位置
    for new_row in range(rows):
        for new_col in range(cols):
            old_row = pos_x[new_row]
            old_col = pos_y[new_col]
            decrypted_img[old_row][old_col] = decrypted_img_temp[new_row][new_col]

    print("Finished decryption.")

    return decrypted_img