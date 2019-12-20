


# AlienWare
虚拟环境：(py36_pytorch12)

# 20191209 第一次训练
batch_size = 8
num_workds = 0

# 待办
1. 加载训练过的模型，继续fine-tune
2. 测试模型指标
3. 修改为检测四边形
4. 将vgg16替换为resnet50/pvanet

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

# 20191220第二次训练
在pths/east_vgg16.pth上进行微调
训练70个epoch。保存在pths_zx_md

# 20191220 第一次训练 屏幕区域
10个epoch
pths_zx_md_screen

# 20191220 第二次训练屏幕区域
修改dataset_md_screen.py
1. 不进行shrink
2. 不计算框的损失



