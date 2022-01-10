# Core Pkgs
# import ee
from torch.serialization import load
import streamlit as st
import cv2
import numpy as np
# import folium
import os
from PIL import Image
# from selenium import webdriver
import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
import datetime
from dateutil.relativedelta import relativedelta
import tempfile
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)


import numpy as np
import torch
import torch.nn.functional as F
from cnnlstm import CNNLSTM
import matplotlib.pyplot as plt
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'


st.set_page_config(layout="wide")
if torch.cuda.is_available() :
    device = "cuda"
else :
    device = "cpu"


   

def load_model():
    
    print("----Loading Model-------")
    checkpoint = torch.load("model.pth", map_location='cpu')
    model = CNNLSTM(num_classes=2)
    model.load_state_dict(checkpoint['state_dict'])
    class_to_idx = {"goal": 1, "no-goal": 0}
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    print("----Model Loaded-------")
    print(model.fc1.weight)
    model.eval()
    return model.to(device)

def predict(clip, model,thresh):
  
    mean = [114.7748, 107.7354, 99.4750]
    std = [1,1,1] 
    norm_method = Normalize(mean, std)
    spatial_transform = Compose([
        Scale((224, 224)),
        ToTensor(1), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    
   

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0).to(device)
    with torch.no_grad():
        # print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs,dim=1)
        print(outputs.cpu().numpy())
    if outputs[:,-1] > thresh :
       return 1
    else:
       return 0

def detect(video,model,thresh):
    fps = video.get(5)
    num_frames = 2 * (fps // 16  + 1)
    print("fps : " , fps)
    out = []
    outs = []
    clip = []

    frame_count = 0
    while True:
        ret, img = video.read()
        if frame_count == 16:
            out.append(predict(clip,model,thresh))
            frame_count = 0
            clip = []
        if len(out) == num_frames or not ret:
            pred = max(out)
            outs.append(pred)
            out = []
        if not ret :
            return  outs
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        clip.append(img)
        frame_count += 1

def plot(out):
    fig = plt.figure(figsize = (10, 5))
    if type(out) == int :
        out = [out]
    base = [str((2*(i+1))) + "sec" for i in range(len(out))]
    print(out)
    # creating the bar plot
    plt.bar(base, out, color ='maroon',
            width = 0.4)
    
    plt.xlabel("time (sec)")
    plt.ylabel("Shot or Not")
    ax = plt.gca()
    ax.set_ylim([0,1])
    plt.plot()
    plt.show()
    return fig
   


def main():
    """BasketBall Pointer Detection App"""
    st.title("MTX Hackolympics")

    activities = ["Inference", "About" , "Metrics"]
    choice = st.sidebar.selectbox("Select Activty", activities)

    if choice == 'Inference':
        st.subheader("Video Detection")

        options = ["Upload Video"]
        selection = st.selectbox("Select Option", options)

        if selection == 'Upload Video':
            video_file = st.file_uploader("Upload Video", type=['mp4', 'mov'])

            if video_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                vf = cv2.VideoCapture(tfile.name)
                model = load_model()
                out = detect(video=vf,model=model,thresh=0.5)
                print(out)
                fig = plot(out)

                st.text("Original Video")
                st.video(video_file)
                st.text("Inference Result :")                
                if 1 in out :
                    st.text("Goal !!")
                else :
                    st.text("No Goal !!")
                st.pyplot(fig)

    elif choice == 'About':
        st.subheader("About Green Cover Detection App")
    
    elif choice == "Metrics" :
        figs = os.listdir("./figs")
        names = ["loss-curve" , "confusion matrix" , "ROC_AUC curve" ,"Accuracy" , "F1" , "Precision" , "Recall"]
        for fig , name in zip(figs , names) :
            st.text(name)
            st.image(os.path.join("./figs",fig))


if __name__ == '__main__':
    main()
