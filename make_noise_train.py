import os
import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np

file_path = 'datasets/DIV2K_train_HR'  # 图像文件夹路径，最后记得加/
output_target_path = 'datasets/Denoising_DIV2K_train/target'
output_input_path = 'datasets/Denoising_DIV2K_train/input'

# 确保输出文件夹存在
os.makedirs(output_target_path, exist_ok=True)
os.makedirs(output_input_path, exist_ok=True)

img_path_list = [f for f in os.listdir(file_path) if f.endswith('.png')]
img_num = len(img_path_list)
count = 0

if img_num > 0:  # 有满足条件的图像
    for j, image_name in enumerate(img_path_list, start=1):  # 逐一读取图像
        input_image = Image.open(os.path.join(file_path, image_name))
        a, b, c = 255, 255, 100
        X, Y = input_image.size

        for c_img_num in range(c):
            y = random.randint(0, X - a - 1)
            x = random.randint(0, Y - b - 1)
            C = input_image.crop((x, y, x + a, y + b))

            sig = round(random.uniform(5, 50))
            V = (sig / 256) ** 2
            noise = np.random.normal(0, V ** 0.5, (a, b, 3))
            added_noise = np.array(C) / 255 + noise
            added_noise = np.clip(added_noise, 0, 1) * 255
            added_noise_image = Image.fromarray(added_noise.astype('uint8'))

            count += 1
            C.save(os.path.join(output_target_path, f'{count}.png'))
            added_noise_image.save(os.path.join(output_input_path, f'{count}.png'))

        print(f'Image {j}')

print('Finished!')