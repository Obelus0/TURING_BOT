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

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--resume_path <path-to-model> 
```


## Tensorboard Visualisation Loss
![alt text](https://github.com/pranoyr/cnn-lstm/blob/master/images/Screenshot%202020-08-13%20at%205.54.36%20PM.png)

## ROC curve 
![alt text](https://github.com/pranoyr/cnn-lstm/blob/master/images/Screenshot%202020-08-13%20at%205.54.36%20PM.png)

## Confusion Matrix
![alt text](https://github.com/pranoyr/cnn-lstm/blob/master/images/Screenshot%202020-08-13%20at%205.54.36%20PM.png)


## Inference
```
python inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes <num_classes> --resume_path <path-to-model.pth> 
```

## References
* https://github.com/kenshohara/video-classification-3d-cnn-pytorch
* https://github.com/HHTseng/video-classification

## License
This project is licensed under the MIT License 

