# from flask import Flask, request, render_template, send_file
# from PIL import Image, ImageDraw, ImageFont
# import io
# import torch
# import torchvision.transforms as T
# import os

# app = Flask(__name__)

# # 모델 정의 함수
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models import ResNet50_Weights

# def get_model_instance_segmentation(num_classes):
#     weights_backbone = ResNet50_Weights.IMAGENET1K_V1
#     model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=weights_backbone)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

# # 클래스 이름
# class_names = {
#     1: 'Knife'
# }

# # 모델 초기화
# num_classes = len(class_names) + 1  # 배경 클래스를 포함하기 위해 +1
# model = get_model_instance_segmentation(num_classes)

# # Load the trained model weights
# checkpoint = torch.load('C:/workspace/deepnoid/deepnoid/PBL2/pth/one_class_detect_model.pth', map_location=torch.device('cpu'))

# # Remove incompatible keys
# del checkpoint['roi_heads.box_predictor.cls_score.weight']
# del checkpoint['roi_heads.box_predictor.cls_score.bias']
# del checkpoint['roi_heads.box_predictor.bbox_pred.weight']
# del checkpoint['roi_heads.box_predictor.bbox_pred.bias']

# model.load_state_dict(checkpoint, strict=False)

# model.eval()

# # 이미지 변환 함수
# transform = T.Compose([
#     T.ToTensor()
# ])

# @app.route('/')
# def upload_file():
#     return render_template('upload.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return 'No file part'
#     file = request.files['file']
#     if file.filename == '':
#         return 'No selected file'
#     image = Image.open(io.BytesIO(file.read())).convert("RGB")
#     transformed_image = transform(image).unsqueeze(0)  # 배치 차원 추가
#     with torch.no_grad():
#         prediction = model(transformed_image)
    
#     print("Prediction:", prediction)  # Debug: print prediction to ensure it's working

#     draw = ImageDraw.Draw(image)
#     # 절대 경로로 글꼴 파일을 불러오기
#     font_path = os.path.join(os.path.dirname(__file__), "fonts", "arial.ttf")
#     font = ImageFont.truetype(font_path, 20)  # 글꼴 및 크기 설정
#     for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
#         if score >= 0.85:  # 신뢰도 점수가 0.9 이상인 경우만 그리기
#             box = [int(b) for b in box.tolist()]
#             draw.rectangle(box, outline="red", width=3)
#             # 텍스트 박스 배경
#             class_name = class_names.get(label.item(), "Unknown")
#             text = f"{class_name} {score:.2f}"
#             text_bbox = draw.textbbox((0, 0), text, font=font)
#             text_width = text_bbox[2] - text_bbox[0]
#             text_height = text_bbox[3] - text_bbox[1]
#             text_location = [box[0], box[1] - text_height]
#             draw.rectangle([tuple(text_location), (text_location[0] + text_width, text_location[1] + text_height)], fill="red")
#             # 텍스트 그리기
#             draw.text((box[0], box[1] - text_height), text, fill="white", font=font)
    
#     img_io = io.BytesIO()
#     image.save(img_io, format='JPEG')
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/jpeg')

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import io
import torch
import torchvision
from flask import Flask, request, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# 모델 로드 및 설정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# 로컬에서 다운로드한 가중치 파일 로드
checkpoint_path = 'C:/workspace/deepnoid/deepnoid/PBL2/pth/final_detention_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)

# Remove the classifier and box predictor from the checkpoint
del checkpoint['roi_heads.box_predictor.cls_score.weight']
del checkpoint['roi_heads.box_predictor.cls_score.bias']
del checkpoint['roi_heads.box_predictor.bbox_pred.weight']
del checkpoint['roi_heads.box_predictor.bbox_pred.bias']

# Load the rest of the state dictionary
model.load_state_dict(checkpoint, strict=False)  # Use strict=False to ignore missing and unexpected keys

# 모델 설정
model.to(device)
model.eval()

# 클래스 매핑
class_mapping = {
    1: 'Knife'
}

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 이미지 변환 함수
transform = T.Compose([
    T.ToTensor()
])

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    transformed_image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스로 이동
    with torch.no_grad():
        prediction = model(transformed_image)

    draw = ImageDraw.Draw(image)
    # 절대 경로로 글꼴 파일을 불러오기
    font_path = os.path.join(os.path.dirname(__file__), "arial.ttf")
    font = ImageFont.truetype(font_path, 20)  # 글꼴 및 크기 설정

    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        if score >= 0.85:  # 신뢰도 점수가 0.9 이상인 경우만 그리기
            box = [int(b) for b in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
            # 텍스트 박스 배경
            text = f"{class_mapping[label.item()]}: {score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_location = [box[0], box[1] - text_height]
            draw.rectangle([tuple(text_location), (text_location[0] + text_width, text_location[1] + text_height+2)], fill="yellow")
            # 텍스트 그리기
            draw.text((box[0], box[1] - text_height), text, fill="black", font=font)

    img_io = io.BytesIO()
    image.save(img_io, format='JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
