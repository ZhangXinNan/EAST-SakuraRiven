


# AlienWare
虚拟环境：(py36_pytorch12)

# 20191209 第一次训练
batch_size = 8
num_workds = 0

# 待办
1. 加载训练过的模型，继续fine-tune
2. 测试模型指标

# 评价
./pths/east_vgg16.pth
```bash
python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit_east_vgg16.zip

Calculated!{"precision": 0.8435782108945528, "recall": 0.8127106403466539, "hmean": 0.8278567925453654, "AP": 0}
```

./pths_zx/model_epoch_400.pth
```bash
python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit_zx_400.zip

Calculated!{"precision": 0.813079551000488, "recall": 0.8021184400577757, "hmean": 0.8075618031992244, "AP": 0}
```


./pths_zx/model_epoch_600.pth
```bash
python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit_zx_600.zip

Calculated!{"precision": 0.82275390625, "recall": 0.8112662493981705, "hmean": 0.816969696969697, "AP": 0}
```