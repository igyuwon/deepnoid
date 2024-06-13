from flask import Flask, request, render_template, send_file
from PIL import Image, ImageDraw, ImageFont
import io
import torch
import torchvision.transforms as T
import os

app = Flask(__name__)

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

def get_model_instance_segmentation(num_classes):
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=91, weights_backbone=weights_backbone)
    return model

class_names = {
    1: 'Gun',
    2: 'Knife',
    3: 'Pliers',
    4: 'Scissors',
    5: 'Wrench'
}

num_classes = len(class_names) + 1
model = get_model_instance_segmentation(num_classes)

checkpoint = torch.load('C:/workspace/deepnoid/deepnoid/PBL2/pth/last_30epoch_model.pth', map_location=torch.device('cuda'))
model.load_state_dict(checkpoint)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.eval()

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
    transformed_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(transformed_image)
    
    print("Prediction:", prediction)  # Debug: print prediction to ensure it's working

    draw = ImageDraw.Draw(image)
    font_path = os.path.join(os.path.dirname(__file__), "fonts", "arial.ttf")
    font = ImageFont.truetype(font_path, 20)
    for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
        if score >= 0.3:  # Lowered threshold for testing
            box = [int(b) for b in box.tolist()]
            draw.rectangle(box, outline="red", width=3)
            class_name = class_names.get(label.item(), "Unknown")
            text = f"{class_name} {score:.2f}"
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_location = [box[0], box[1] - text_height]
            draw.rectangle([tuple(text_location), (text_location[0] + text_width, text_location[1] + text_height)], fill="red")
            draw.text((box[0], box[1] - text_height), text, fill="white", font=font)
    
    img_io = io.BytesIO()
    image.save(img_io, format='JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
