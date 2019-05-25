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
- loss文件夹 不同损失函数（包括GAN）的实现
- preprocess 数据下载、格式转换。（无需阅读）

### Further work
- 改进模型本身
- 研究如何使用VMAF指标，目前的验证阶段只会计算PSNR
- 如何进行针对比赛进行数据增广？
- 尝试不同的损失函数的组合
- 如何给定一个视频，自动给出超分辨率后的结果？（模型的使用）

## Usage
#### Training
使用方法见https://shimo.im/docs/omS9rb73rqQ3CH9x


## Environment
* Python (Tested with 3.6)
* PyTorch >= 0.4.0
* numpy
* scipy
* matplotlib
* tqdm
