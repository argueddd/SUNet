import os
import random
from PIL import Image
import numpy as np

file_path = 'datasets/DIV2K_valid_HR'  # 原始图像文件夹路径
output_target_path = 'datasets/Denoising_DIV2K_test/target'
output_input_path= 'datasets/Denoising_DIV2K_test/input'

# 确保输出文件夹存在
os.makedirs(output_target_path, exist_ok=True)
os.makedirs(output_input_path, exist_ok=True)


img_path_list = [f for f in os.listdir(file_path) if f.endswith('.png')]
img_num = len(img_path_list)
count = 0

if img_num > 0:  # 如果有满足条件的图像
    for j, image_name in enumerate(img_path_list, start=1):  # 逐一读取图像
        input_image = Image.open(os.path.join(file_path, image_name))
        a, b, c = 255, 255, 3  # 图像裁剪大小和裁剪次数
        X, Y = input_image.size

        for _ in range(c):
            # 随机选择裁剪位置
            y = random.randint(0, X - a - 1)
            x = random.randint(0, Y - b - 1)
            C = input_image.crop((x, y, x + a, y + b))

            # 定义三个不同的噪声方差
            V1, V2, V3 = (10 / 256) ** 2, (30 / 256) ** 2, (50 / 256) ** 2

            # 添加高斯噪声
            def add_gaussian_noise(image, variance):
                noise = np.random.normal(0, variance ** 0.5, (a, b, 3))
                noisy_image = np.array(image) / 255 + noise
                noisy_image = np.clip(noisy_image, 0, 1) * 255
                return Image.fromarray(noisy_image.astype('uint8'))

            added_noise1 = add_gaussian_noise(C, V1)
            added_noise2 = add_gaussian_noise(C, V2)
            added_noise3 = add_gaussian_noise(C, V3)


            # 保存噪声图像和对应的无噪声目标图像
            count += 1
            added_noise1.save(os.path.join(output_input_path, f'{count}.png'))
            C.save(os.path.join(output_target_path, f'{count}.png'))

            count += 1
            added_noise2.save(os.path.join(output_input_path, f'{count}.png'))
            C.save(os.path.join(output_target_path, f'{count}.png'))

            count += 1
            added_noise3.save(os.path.join(output_input_path, f'{count}.png'))
            C.save(os.path.join(output_target_path, f'{count}.png'))

        print(f'Image {j}')

print('Finished!')