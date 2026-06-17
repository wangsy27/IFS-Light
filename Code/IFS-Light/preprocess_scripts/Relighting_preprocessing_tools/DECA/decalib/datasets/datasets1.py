# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io
import lmdb
from io import BytesIO
from PIL import Image
import pandas as pd

import torchvision.transforms.functional as Ftrans

from . import detectors

def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        # if count%sample_step == 0:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                'utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img



class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10, 
                 path=os.path.expanduser('/dataset/data/ffhq_256_with_anno/ffhq_256/'),
                 image_size=256,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = False,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 **kwargs):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        # if isinstance(testpath, list):
        #     self.imagepath_list = testpath
        # elif os.path.isdir(testpath): 
        #     self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        # elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
        #     self.imagepath_list = [testpath]
        # elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
        #     self.imagepath_list = video2sequence(testpath, sample_step)
        # else:
        #     print(f'please check the test path: {testpath}')
        #     exit()
        # # print('total {} images'.format(len(self.imagepath_list)))
        # self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size

        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()
        
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        #return len(self.imagepath_list)
        return self.length

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        # imagepath = self.imagepath_list[index]
        # imagename, imageext = os.path.splitext(os.path.split(imagepath)[-1])[0], os.path.splitext(os.path.split(imagepath)[-1])[1]
        # image = np.array(imread(imagepath))
        # if len(image.shape) == 2:
        #     image = image[:,:,None].repeat(1,1,3)
        # if len(image.shape) == 3 and image.shape[2] > 3:
        #     image = image[:,:,:3]
        assert index < self.length
        index = index + self.offset
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        image = np.array(image)
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            # kpt_matpath = os.path.splitext(imagepath)[0]+'.mat'
            # kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
            kpt_matpath = ''
            kpt_txtpath = ''
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                # 'imagename': imagename,
                # 'imageext': imageext,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }