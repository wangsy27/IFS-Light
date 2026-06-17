import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from torchvision import transforms

import os
from time import time

import numpy as np
import torch.utils.data
import sys

sys.path.append('./')
from auxiliary.settings import DEVICE
from classes.core.Evaluator import Evaluator
from classes.fc4.ModelFC4 import ModelFC4
from classes.fc4.repvit import utils

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])


parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', help='Input images', required=True)
parser.add_argument('--out_dir', help='The Dataset name to be output.', required=True)
args = parser.parse_args()



class WildDataset(Dataset):
    def __init__(self, path = None):
        """
        初始化数据集
        :param image_dir: 图像文件夹路径
        :param skin_label_file: 肤色标签文件路径
        :param light_label_file: 光色标签文件路径
        :param transform: 图像预处理
        """

        self.image_dir = path

        self.transform = transform

        # 确保图像和标签匹配
        self.image_names = os.listdir(self.image_dir)
        

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 获取图像路径
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 加载图像
        image = Image.open(image_path).convert('RGB')  # 确保图像是 RGB 格式

        # 图像预处理
        if self.transform:
            image = self.transform(image)

        return image, image_name


def main():
    evaluator = Evaluator()
    eval_data = {"file_names": [], "skin_pred": [], "light_pred": []}
    model = ModelFC4()

    utils.replace_batchnorm(model)

    fold_evaluator = Evaluator()

    test_set = WildDataset(path=args.image_dir)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=16)

    path_to_pretrained = './model'
    model.load(path_to_pretrained)
    model.evaluation_mode()

    

    print(" * Test set size: {}".format(len(test_set)))
    print(" * Using trained model stored at: {} \n".format(path_to_pretrained))

    with torch.no_grad():
        for i, (img, file_name) in enumerate(test_dataloader):
            img = img.to(DEVICE)
            skin_pred, light_pred = model.predict(img, return_steps=False)
            #print(skin_pred, skin_label, light_pred, light_label)
            eval_data["file_names"].append(file_name[0])
            eval_data["skin_pred"].append(skin_pred.cpu().numpy())
            eval_data["light_pred"].append(light_pred.cpu().numpy())
            print('\t - Input: {} - Batch: {}'.format(file_name[0], i))
    
    with open(args.out_dir + '/ffhq-valid-skincolor-anno.txt', 'w') as file:
        for row in range(len(eval_data["file_names"])):
            file_name = eval_data["file_names"][row]
            skin_pred = ' '.join(map(str, eval_data["skin_pred"][row][0]))
            file.write(f"{file_name} {skin_pred}\n")


if __name__ == '__main__':
    main()
