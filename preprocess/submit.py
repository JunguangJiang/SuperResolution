"""
Convert all the bmp files to the final result.zip
"""
from preprocess import preprocess
import os

if __name__ == '__main__':
    preprocess.bmps_2_y4ms(source_dir="YoukuDataset/test/output", target_dir="YoukuDataset/test/y4m", ratio=1)
    preprocess.create_final_result("YoukuDataset/test/y4m", "YoukuDataset/test/result", 0.1)
    os.chroot("YoukuDataset/test/result")
    os.system("zip result.zip *.y4m")

