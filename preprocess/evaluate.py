"""
Eval the bmps produced by the Model
"""
from preprocess import preprocess


if __name__ == '__main__':
    preprocess.bmps_2_y4ms(source_dir="YoukuDataset/eval/bmp/output", target_dir="YoukuDataset/eval/y4m/output", ratio=0.1)
    preprocess.y4ms_2_yuvs(y4m_dir="YoukuDataset/eval/y4m/output", yuv_dir="YoukuDataset/eval/yuv/output")
    # TODO add VMAF evaluation

