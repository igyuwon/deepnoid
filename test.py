import os
from glob import glob
from tqdm.notebook import tqdm_notebook as tqdm
import pandas as pd
import seaborn as sns
import xmltodict as xmd
import cv2
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class_name = {
    1: 'Gun',
    2: 'Knife',
    3: 'Pliers',
    4: 'Scissors',
    5: 'Wrench'
}

NUM_CLASSES = len(class_name) + 1

def load_from_directory(path):
    if os.path.isfile(path+'/annot.csv'):
        df = pd.read_csv(path+'/annot.csv')
        print(f"Loaded {path+'/annot.csv'}")
    else:
        df = pd.DataFrame(columns=['img', 'xml'])
        flist = glob(path+'/*.jpg')

        for i, img in tqdm(enumerate(flist), total=len(flist), desc=f'Loading data from {path}...'):
            xml = img[:-3] + 'xml'
            df.loc[i] = [img, xml]

        df.to_csv(path+'/annot.csv', index=False)

    return df

train_df = load_from_directory('./ugku-2/train')
valid_df = load_from_directory('./ugku-2/valid')
test_df  = load_from_directory('./ugku-2/test')

class Data_Structure(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.image_ids = self.dataframe.index

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.dataframe.index[idx]
        fname, xml = self.dataframe.loc[idx]

        image = cv2.imread(fname, cv2.IMREAD_COLOR)[:,:,::-1].copy()

        image = cv2.resize(image, (512, 512), cv2.INTER_LINEAR) / 255.

        # XML annotation 불러오기 및 dict타입 구성
        with open(xml, 'r') as f:
            annot = xmd.parse(f.read())['annotation']

        width, height = int(annot['size']['width']), int(annot['size']['height'])
        boxes = []
        labels = []
        areas = []

        # 여러 개의 object를 처리
        objects = annot['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            object_class_name = obj['name']
            object_class = [k for k, v in class_name.items() if v == object_class_name][0]  # 클래스 이름에서 번호로 매핑

            xmin, xmax, ymin, ymax = [int(obj['bndbox'][value]) for value in obj['bndbox']]

            # Bounding box normalization => 0~1
            box = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
            bbox_w, bbox_h = box[0, 2] - box[0, 0], box[0, 3] - box[0, 1]
            box[0, ::2] = box[0, ::2] / width * 512
            box[0, 1::2] = box[0, 1::2] / height * 512

            boxes.append(box[0])
            labels.append(object_class)
            areas.append(bbox_w * bbox_h)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        areas = np.array(areas, dtype=np.float32)

         # 학습용 레이블 데이터 생성
        target = {
            'image_id': torch.as_tensor(image_id),

            # Bounding box 좌표 구성은 numpy array -> (Number, xmin, ymin, xmax, ymax)
            # 데이터셋에서 탐지할 object 수에 맞춰서 수정 -> (N, xmin, ymin, xmax, ymax) -> N: 탐지 객체 수
            'boxes': torch.from_numpy(boxes),

            # 레이블 클래스 부여
            # 이미지에 탐지 객체의 레이블이 여러 개 존재할 경우 여러 값 작성 -> [1, 3, 2], [1, 1, 1]...
            'labels': torch.as_tensor(labels),
            'area': torch.as_tensor(areas),
            'iscrowd': torch.from_numpy(np.zeros(len(boxes)))
        }

        return (
            torch.from_numpy(image).float().permute(2, 0, 1),  # 순서 변경 -> (channel, horizontal, vertical)
            target
        )

TRAIN_DATA = Data_Structure(train_df)
VAL_DATA = Data_Structure(valid_df)

print(TRAIN_DATA[0])
'''
print(TRAIN_DATA[0]) 출력
(tensor([[[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],
 
         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]],
 
         [[1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          ...,
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.],
          [1., 1., 1.,  ..., 1., 1., 1.]]]),
 {'image_id': tensor(0),
  'boxes': tensor([[143.8202, 237.9419, 444.2447, 491.8174],
          [124.0050, 343.1037, 395.6654, 501.3776]]),
  'labels': tensor([2, 2]),
  'area': tensor([112330.,  63325.]),
  'iscrowd': tensor([0., 0.], dtype=torch.float64)})
'''
def Preview(data, class_name, model=None, device='cuda', iou_threshold=0.5, score_threshold=0):
    plt.figure(figsize=(12,12))
    plt.imshow(data[0].permute(1,2,0).numpy())
    ax = plt.gca()
    for i, box in enumerate(data[1]['boxes']):
        x1, y1, x2, y2 = box
        ground_truth = Rectangle(
                            (x1, y1),
                            x2 - x1, y2 - y1,
                            edgecolor='blue',
                            linewidth=3,
                            fill=False )
        ax.add_patch(ground_truth)
        plt.text(x1, y1 - 10, f'[Ground Truth]: {class_name[data[1]["labels"][i].item()]}', color='blue', fontsize=12)

    if model is not None:
        model.eval()
        with torch.no_grad():
            rtn = model(torch.unsqueeze(data[0], 0).to(device))  # 차원 확장
        class_pred = rtn[0]['labels'].detach().cpu()
        bbox = rtn[0]['boxes'].detach().cpu()
        scores = rtn[0]['scores'].detach().cpu()
        get = torchvision.ops.nms(boxes=bbox, scores=scores, iou_threshold=iou_threshold)
        for i, pred in enumerate(class_pred):
            if i not in get: continue
            if scores[i] < score_threshold: continue

            x1, y1, x2, y2 = bbox[i]
            prediction = Rectangle(
                                (x1, y1), x2 - x1, y2 - y1,
                                edgecolor='red', linewidth=3, fill=False )
            ax.add_patch(prediction)
            plt.text(x1, y1 - 10, f'[Prediction]: {class_name[pred.item()]} - {scores[i]:.4f}', color='red', fontsize=12)
    plt.show()

BATCH_SIZE = 8
NUM_WORKERS = 8

def collate_fn(batch):
    return tuple(zip(*batch)) # 다양한 크기의 shape를 다루기 위해 조정

train_loader = DataLoader(TRAIN_DATA,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          collate_fn=collate_fn)

val_loader = DataLoader(VAL_DATA,
                        batch_size=4,
                        shuffle=False,
                        num_workers=NUM_WORKERS,
                        collate_fn=collate_fn)

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None, min_size=512, max_size=512)

in_features = model.roi_heads.box_predictor.cls_score.in_features
print(in_features) # 1024
model = model.to(device)

import sys
sys.path.append('./torch_ref/torchvision_ref') # 경로 추가
import torch_ref.torchvision_ref.engine as engine

# Params: 가중치 업데이트
params = [p for p in model.parameters() if p.requires_grad]

# SGD Optimizer
optimizer = torch.optim.SGD(params, lr=0.01, weight_decay=0.0005)

# lr_scheduler로 현재 learning rate를 3에폭 마다 95%로 감소
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

num_epochs = 3

for epoch in range(num_epochs):
    engine.train_one_epoch(model, optimizer, train_loader, 'cuda', epoch, print_freq=50)

    # lr_scheduler 실행
    lr_scheduler.step()

    engine.evaluate(model, val_loader, device='cuda')
    torch.save(model.state_dict(), f"./last_epoch_model.pth")
