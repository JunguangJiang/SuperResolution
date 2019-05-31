### Res scale
修改过的RGM mean+调整过的数据集+L1损失函数+0.0001初始学习速率+Leaky Relu
edsr_split1: 32 x 0.1
edsr_split2: 16 x 0.1 + 16 x 0.05
edsr_split4: 8 x 0.1 + 8 x 0.05 + 8 x 0.025 + 8 x 0.0125
```bash
# edsr_split1
nohup python main.py --data_train Youku --dir_data YoukuDataset/sample2 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss 1*MSE --print_every 100 --load "" --save "edsr_split1" --test_every 1000 --cuda cuda:1 --batch_size 16 --model EDSR --reset --lr 0.0001 --ms_factor 0.75 --ms_patience 10 > log/edsr_split1.txt 2>&1 &  

# edsr_split2
nohup python main.py --data_train Youku --dir_data YoukuDataset/sample2 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 16 --n_feats 256 --res_scale 0.1 --res_scale_factor 0.5 --n_dedsrblocks 2 --loss 1*MSE --print_every 100 --load "" --save "edsr_split2" --test_every 1000 --cuda cuda:3 --batch_size 16 --model DEDSR --reset --lr 0.0001 --ms_factor 0.75 --ms_patience 10 > log/edsr_split2.txt 2>&1 &

nohup python main.py --dir_data YoukuDataset/sample3 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 16 --n_feats 256 --res_scale 0.1 --res_scale_factor 0.5 --n_dedsrblocks 2 --loss 1*MSE --print_every 100 --load "edsr_split2" --save "edsr_split2" --test_every 1000 --cuda cuda:3 --batch_size 16 --model DEDSR --pre_train ../experiment/edsr_split2/model/model_best.pt --act "relu" --ms_factor 0.75 --ms_patience 10 > log/edsr_split2.txt 2>&1 &

# edsr_split3
nohup python main.py --data_train Youku --dir_data YoukuDataset/sample2 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 8 --n_feats 256 --res_scale 0.1 --res_scale_factor 0.5 --n_dedsrblocks 4 --loss 1*MSE --print_every 100 --load "" --save "edsr_split3" --test_every 1000 --cuda cuda:4 --batch_size 16 --model DEDSR --reset --lr 0.0001 --ms_factor 0.75 --ms_patience 10 > log/edsr_split3.txt 2>&1 &

nohup python main.py --dir_data YoukuDataset/sample3 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 8 --n_feats 256 --res_scale 0.1 --res_scale_factor 0.5 --n_dedsrblocks 4 --loss 1*MSE --print_every 100 --load "edsr_split3" --save "edsr_split3" --test_every 1000 --cuda cuda:4 --batch_size 16 --model DEDSR --pre_train ../experiment/edsr_split3/model/model_best.pt --act "relu" --ms_factor 0.75 --ms_patience 10 > log/edsr_split3.txt 2>&1 &

# edsr_split4
nohup python main.py --data_train Youku --dir_data YoukuDataset/sample2 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 8 --n_feats 256 --res_scale 0.1 --res_scale_factor 0.8 --n_dedsrblocks 4 --loss 1*MSE --print_every 100 --load "" --save "edsr_split4" --test_every 1000 --cuda cuda:5 --batch_size 16 --model DEDSR --reset --lr 0.0001 --ms_factor 0.75 --ms_patience 10 > log/edsr_split4.txt 2>&1 &
 
 nohup python main.py --dir_data YoukuDataset/sample3 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 8 --n_feats 256 --res_scale 0.1 --res_scale_factor 0.8 --n_dedsrblocks 4 --loss 1*MSE --print_every 100 --load "edsr_split4" --save "edsr_split4" --test_every 1000 --cuda cuda:1 --batch_size 16 --model DEDSR --pre_train ../experiment/edsr_split4/model/model_best.pt --act "relu" --ms_factor 0.75 --ms_patience 10 > log/edsr_split4.txt 2>&1 &
```

### Relu vs Leaky Relu
修改过的RGM mean+调整过的数据集+MSE损失函数+0.0001初始学习速率
edsr_relu = edsr_split1
edsr_leaky_relu:
```bash
#edsr_leaky_relu
nohup python main.py --data_train Youku --dir_data YoukuDataset/sample2 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss 1*MSE --print_every 100 --load "" --save "edsr_leaky_relu" --test_every 1000 --cuda cuda:5 --batch_size 16 --model EDSR --reset --lr 0.0001 --ms_factor 0.75 --ms_patience 10 --act "leaky_relu"> log/edsr_leaky_relu.txt 2>&1 &
```

### ESRGAN
在以MSE为loss上预训练过的模型作为初始模型
0.01*L1 + 0.005*RGAN + VGG22
```bash
nohup python main.py --dir_data YoukuDataset/sample3 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --res_scale_factor 1 --loss 0.01*L1+0.005*RGAN+1*VGG22 --print_every 100 --save "esrgan" --test_every 500 --cuda cuda:2 --batch_size 16 --model EDSR --pre_train ../experiment/esrgan/model/model_best.pt --act "relu" --ms_factor 0.75 --ms_patience 10 > log/esrgan.txt 2>&1 &
```

### baseline
```bash
# 恢复训练
nohup python main.py --dir_data YoukuDataset/sample3 --data_train Youku --data_test Youku --epochs 500 --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --res_scale_factor 1 --loss 1*MSE --print_every 100 --load "baseline_edsr_mse2" --save "baseline_edsr_mse2" --test_every 500 --cuda cuda:7 --batch_size 16 --model EDSR --pre_train ../experiment/baseline_edsr_mse2/model/model_best.pt --act "relu" --ms_factor 0.75 --ms_patience 10 > log/baseline_edsr_mse2.txt 2>&1 &

# 测试
nohup python videotester.py --dir_data YoukuDataset/test/input1 --data_test YoukuTest --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss 1*MSE --cuda cuda:5 --model EDSR --save_results --results_dir YoukuDataset/test/output --pre_train ../experiment/baseline_edsr_mse2/model/model_best.pt --test_only --save "baseline_edsr_mse_test2" --act "relu" > log/baseline_edsr_mse_test2.txt 2>&1 &

nohup python videotester.py --dir_data YoukuDataset/test/input2 --data_test YoukuTest --scale 4 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --loss 1*MSE --cuda cuda:2 --model EDSR --save_results --results_dir YoukuDataset/test/output --pre_train ../experiment/baseline_edsr_mse2/model/model_best.pt --test_only --save "baseline_edsr_mse_test2_2" --act "relu" > log/baseline_edsr_mse_test2_2.txt 2>&1 &
```