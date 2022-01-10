from re import DEBUG, sub
from flask import Flask, render_template, request, redirect, send_file, url_for,jsonify

from werkzeug.utils import secure_filename
import os
import cv2
import subprocess
import torch
from cnnlstm import CNNLSTM
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
import torch.nn.functional as F
from PIL import Image
import sys


app = Flask(__name__)
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)


print("----Loading Model-------")
checkpoint = torch.load("./ckpt.pth", map_location='cpu')
model = CNNLSTM(num_classes=2)
model.load_state_dict(checkpoint['state_dict'])
class_to_idx = {"goal" : 1, "no-goal" : 0}
mean =  [114.7748, 107.7354 , 99.4750]
std =   [38.7568578 , 37.88248729,40.02898126 ]
device = torch.device("cpu")
print(class_to_idx)
idx_to_class = {}
for name, label in class_to_idx.items():
    idx_to_class[label] = name
print("----Model Loaded-------")



# def resume_model(opt, model):
#     """ Resume model 
#     """
#     checkpoint = torch.load(opt.resume_path, map_location='cpu')
#     model.load_state_dict(checkpoint['state_dict'])


def predict(clip, model):

    norm_method = Normalize(mean,std)

    spatial_transform = Compose([
        Scale((224,224)),
        #Scale(int(opt.sample_size / opt.scale_in_test)),
        #CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(1), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs)
    scores, idx = torch.topk(outputs, k=1)
    mask = scores > 0.6
    preds = idx[mask]
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

######################Doesnt work########################################

@app.route("/detect", methods=['POST'])
def detect():
    if not request.method == "POST":
        return
    print(request.files)
    video = request.files['video']
    # print(request)
    # # video.save(os.path.join(uploads_dir, secure_filename(video.filename)))
    # print(video)
    return jsonify("hello")

    clip = []
    frame_count = 0
    while True:
        ret, img = video.read()
        if frame_count == 16:
            # print(len(clip))
            preds = predict(clip, model)
            draw = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            if preds.size(0) != 0:
                # print(idx_to_class[preds.item()])
                cv2.putText(draw, idx_to_class[preds.item(
                )], (100, 100), font, .5, (255, 255, 255), 1, cv2.LINE_AA)

            frame_count = 0
            clip = []

        #img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = Image.fromarray(img)
        clip.append(img)
        frame_count += 1
    jsonify(result=result, probability=pred_proba)

   
   

# @app.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         # Get the image from post request
#         img = base64_to_pil(request.json)

#         # Save the image to ./uploads
#         # img.save("./uploads/image.png")

#         # Make prediction
#         preds = model_predict(img, model)

#         # Process your result for human
#         pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
#         pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

#         result = str(pred_class[0][0][1])               # Convert to string
#         result = result.replace('_', ' ').capitalize()
        
#         # Serialize the result, you can add additional fields
#         return jsonify(result=result, probability=pred_proba)

#     return None


@app.route('/return-files', methods=['GET'])
def return_file():
    obj = request.args.get('obj')
    loc = os.path.join("runs/detect", obj)
    print(loc)
    try:
        return send_file(os.path.join("runs/detect", obj), attachment_filename=obj)
        # return send_from_directory(loc, obj)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True,port = 6969)