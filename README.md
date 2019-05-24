# Super Resolution
## Introduction
1. 服务器用户名jiangjunguang，密码jiangjunguang。代码放在/workspace/SuperResolution/src下。
2. 数据集已经全部转成图片放置于114服务器的/data/Youku_SR_video/image目录下。也可以在代码路径下通过YoukuDataset/image进行访问。
3. 已使用conda配置好python环境，使用方法：source activate Youku。
4. 在https://github.com/thstkdgus35/EDSR-PyTorch的基础上进行了些许修改
5. https://github.com/icpm/super-resolution 包含了多种模型的实现。

### Code Structure
- main.py程序的入口，option.py和template.py都是用于命令行参数解析的，无需阅读
- trainer.py 训练器
- data文件夹 Dataset子类，我们只需要使用youku.py(可能有bug)进行训练
- model文件夹 不同的模型实现（重点阅读）
- loss文件夹 来自https://github.com/thstkdgus35/EDSR-PyTorch EDSR对于GAN的实现 （尚未集成进来）
- preprocess 数据下载、格式转换。（无需阅读）

### Further work
- 改进模型本身
- 研究如何使用VMAF指标，目前的验证阶段只会计算PSNR
- 如何进行针对比赛进行数据增广？
- 尝试不同的损失函数的组合
- 如何给定一个视频，自动给出超分辨率后的结果？（模型的使用）

## Usage
#### Training
```bash
python main.py --help //查看所有的参数

//使用已经训练好的模型对../test下的图片进行测试,输出在experiment/test/results下
python main.py --data_test Demo --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train ../experiment/epsr1_model.pt --test_only --save_results

//数据集文件在YoukuDataset/image下,训练集名称为Youku,验证集名称为Youku,进行模型完整的训练
python main.py --data_train Youku --dir_data YoukuDataset/image --data_train Youku --data_test Youku --epochs 10 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss 1*MSE --print_every 1 --load "" --test_every 10 --cuda cuda:7

//数据集文件在YoukuDataset/image下,训练集名称为Youku,验证集名称为Youku,进行代码的测试(YoukuDataset/small下的图片数量比较少，可以较快地测试程序的bug)
python main.py --data_train Youku --dir_data YoukuDataset/small --data_train Youku --data_test Youku --epochs 10 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss 1*MSE --print_every 1 --load "" --test_every 10 --cuda cuda:7
```
常用参数说明
- dir_data 数据集的文件夹位置
- epochs 训练次数
- loss 1*MSE+0.5*GAN 损失函数及其比重
- print_every 训练阶段打印损失函数的频率
- load 假如之前有训练记录，可以用load指定路径来恢复训练
- save 指定训练保存的路径，默认为test(由于根目录是experiment，那么保存在experiment/test下)
- test_every 每隔多少个batch进行一次测试
- cuda 指定cuda的版本

损失函数和PSNR的可视化结果均在save指定的文件夹下。

## Environment
* Python (Tested with 3.6)
* PyTorch >= 0.4.0
* numpy
* scipy
* matplotlib
* tqdm
