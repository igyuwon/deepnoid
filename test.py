import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import math
import cv2

path = 'C:/dataset'
categories = ['train', 'test', 'val', 'auto_test'] # 전처리된 데이터셋을 훈련용, 평가용, 검증용으로 구분
data_dir = path+'/osteoarthritis/'
# device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu') # Mac OS
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

def resize_image(img,size=(128,128)):
    return cv2.resize(img,size)

def he_img(img):
    return cv2.equalizeHist(img)

def clahe_image(img):
    clahe = cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8))
    cl_img = clahe.apply(img)
    return cl_img

def denoise_img(img):
    return cv2.fastNlMeansDenoising(img,None,30,7,21)

def normalize_img(img):
    return cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)

def detect_edge(img):
    return cv2.Canny(img,100,200)

def blur_img(img):
    return cv2.GaussianBlur(img,(5,5),0)

def find_contour(img):
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hiearchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# Data augmentation settings
datagen = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    fill_mode='nearest'
)

def augment_images(data_dir, label, augment_count=500):
    label_path = os.path.join(data_dir, 'train', str(label))
    augmented_dir = os.path.join(label_path, 'augmented')
    os.makedirs(augmented_dir, exist_ok=True)

    img_names = os.listdir(label_path)
    img_names = [name for name in img_names if os.path.isfile(os.path.join(label_path, name))]
    generated_count = 0

    while generated_count < augment_count:
        img_name = np.random.choice(img_names)
        img_path = os.path.join(label_path, img_name)
        try:
            img = load_img(img_path, color_mode='grayscale', target_size=(128, 128))
            img = img_to_array(img)
            img = img.reshape((1,) + img.shape)

            for batch in datagen.flow(img, batch_size=1, save_to_dir=augmented_dir, save_prefix='aug', save_format='png'):
                generated_count += 1
                break  # 한 장만 생성하고 반복문 탈출
        except PermissionError:
            print(f"Permission denied: Unable to access file {img_path}")
        except Exception as e:
            print(f"Error processing file {img_path}: {e}")

# Augment images in label 4
# data_dir = path+'/osteoarthritis/train/'  # 적절한 데이터 디렉토리로 변경
augment_images(data_dir, 4)

# Display augmented images
augmented_img_paths = os.listdir(os.path.join(data_dir, 'train', '4', 'augmented'))
augmented_img_paths = shuffle(augmented_img_paths)[:6]  # 무작위로 6개 선택

for i, img_name in enumerate(augmented_img_paths):
    img_path = os.path.join(data_dir, 'train', '4', 'augmented', img_name)
    img = load_img(img_path, color_mode='grayscale', target_size=(128, 128))
    plt.imshow(img, cmap='gray')
    plt.title(f'Augmented Image {i + 1}')
    plt.show()

def load_data(data_dir):
    images = []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        if os.path.isfile(img_path):  # 파일인지 확인
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # 이미지가 정상적으로 로드되었는지 확인
                img = resize_image(img)
                img = clahe_image(img)
                img = normalize_img(img)
                images.append(img)
    prepared_data = np.array(images)
    return prepared_data

def load_and_augment_data(data_dir, label):
    label_path = os.path.join(data_dir, 'train', str(label))
    augmented_dir = os.path.join(label_path, 'augmented')

    images = []
    # Load original images
    for img_name in os.listdir(label_path):
        if img_name.endswith('.png'):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = resize_image(img)
                img = clahe_image(img)
                img = normalize_img(img)
                images.append(img)
    
    # Load augmented images
    if os.path.exists(augmented_dir):
        for img_name in os.listdir(augmented_dir):
            img_path = os.path.join(augmented_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = resize_image(img)
                img = clahe_image(img)
                img = normalize_img(img)
                images.append(img)
    
    return np.array(images)


# 각 카테고리와 라벨에 따라 이미지를 처리
all_data = {}
for category in categories:
    category_path = os.path.join(data_dir, category)
    all_data[category] = {}
    for label in range(5):
        if category == 'train' and label == 4:
            processed_img_list = load_and_augment_data(data_dir, label)
        else:
            label_path = os.path.join(category_path, str(label))
            if os.path.isdir(label_path):
                processed_img_list = load_data(label_path)
            else:
                processed_img_list = np.array([])
        all_data[category][label] = processed_img_list

# Function to sample 500 images from each label for training
def sample_images(data, num_samples=500):
    sampled_data = []
    sampled_labels = []
    for label, images in data.items():
        if len(images) >= num_samples:
            sampled_data.append(images[:num_samples])
            sampled_labels.append(np.full(num_samples, label))
        else:
            sampled_data.append(images)
            sampled_labels.append(np.full(len(images), label))
    sampled_data = np.concatenate(sampled_data, axis=0)
    sampled_labels = np.concatenate(sampled_labels, axis=0)
    return shuffle(sampled_data, sampled_labels)

# Sample 500 images from each label for training
train_data_sampled, train_labels_sampled = sample_images(all_data['train'])

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Combine validation and test sets without sampling
val_data_combined = []
val_labels_combined = []
test_data_combined = []
test_labels_combined = []

for dataset in ['val', 'test']:
    for label, images in all_data[dataset].items():
        if dataset == 'val':
            val_data_combined.append(images)
            val_labels_combined.append(np.full(images.shape[0], label))
        else:
            test_data_combined.append(images)
            test_labels_combined.append(np.full(images.shape[0], label))

val_data_combined = np.concatenate(val_data_combined, axis=0)
val_labels_combined = np.concatenate(val_labels_combined, axis=0)
test_data_combined = np.concatenate(test_data_combined, axis=0)
test_labels_combined = np.concatenate(test_labels_combined, axis=0)

# PyTorch 텐서로 변환
train_data_tensor = torch.tensor(train_data_sampled, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels_sampled, dtype=torch.long)
val_data_tensor = torch.tensor(val_data_combined, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels_combined, dtype=torch.long)
test_data_tensor = torch.tensor(test_data_combined, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels_combined, dtype=torch.long)

# PyTorch 데이터셋 및 데이터 로더 생성
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)