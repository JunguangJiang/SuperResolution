# Super Resolution
## Usage
#### Training
```bash
python -m visdom.server # start a server to monitor the training process
python train2.py # start training
```

#### Processing a y4m video
```bash
python test_video.py 
```
LR videos are in data/test/input and HR videos are in data/test/output

## Environment
python3.6
pytorch
torchnet
tqdm