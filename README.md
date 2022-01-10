# TURING_BOT
Implementation of CNN LSTM with Resnet backend for Video Classification

# Getting Started
## Prerequisites
* PyTorch (1.7.1)
* Python 3
* Streamlit
* Docker

## Run the app
```
git clone https://github.com/Obelus0/TURING_BOT.git
docker build -t basketball . 
docker run basketball
```



## Train
Once you have created the dataset, start training ->
```
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes <num_classes>
```

## Inference
```
python inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes <num_classes> --resume_path <path-to-model.pth> 
```


## Tensorboard Visualisation Loss
![alt text](https://github.com/Obelus0/TURING_BOT/blob/main/photos/0.png)

## ROC curve 
![alt text](https://github.com/Obelus0/TURING_BOT/blob/main/photos/2.png)

## Confusion Matrix
![alt text](https://github.com/Obelus0/TURING_BOT/blob/main/photos/1.png)


## Inference
```
python inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes <num_classes> --resume_path <path-to-model.pth> 
```


