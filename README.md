# Super Resolution
## Introduction
1. 服务器用户名jiangjunguang，密码jiangjunguang。代码放在/workspace/SuperResolution/src下。
2. 数据集已经全部转成图片放置于114服务器的/data/Youku_SR_video/image目录下。也可以在代码路径下通过YoukuDataset/image进行访问。
3. 已使用conda配置好python环境，使用方法：source activate Youku。
4. 代码在https://github.com/subeeshvasu/2018_subeesh_epsr_eccvw 和 https://github.com/thstkdgus35/EDSR-PyTorch的基础上进行了些许修改。后者的实现更加完整，但是由于时间比较早，和当前的pytorch版本不太兼容。

### Code Structure
- main.py程序的入口，option.py和template.py都是用于命令行参数解析的，无需阅读
- trainer.py 训练器
- data文件夹 Dataset子类，我们只需要使用youku.py(可能有bug)进行训练
- model文件夹 不同的模型实现（重点阅读）
- loss文件夹 来自https://github.com/thstkdgus35/EDSR-PyTorch EDSR对于GAN的实现 （尚未集成进来）
- preprocess 数据下载、格式转换。（无需阅读）

### Progress
- 当前可以将训练代码跑起来。
- 代码中尚未使用loss函数、也没有使用GAN。
- trainer尚未仔细阅读。

## Usage
#### Training
```bash
python main.py --help //查看所有的参数

//使用已经训练好的模型对../test下的图片进行测试,输出在experiment/test/results下
python main.py --data_test Demo --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/epsr1_model.pt --test_only --save_results

//数据集文件在YoukuDataset/image下,训练集名称为Youku,验证集名称为Youku,进行模型的训练
python main.py --data_train Youku --dir_data YoukuDataset/image --data_train Youku --data_test Youku --epochs 10 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/epsr1_model.pt
```

## Environment
* Python (Tested with 3.6)
* PyTorch >= 0.4.0
* numpy
* scipy
* matplotlib
* tqdm