import os
import cv2
import numpy as np
import tqdm


def compute_rgb_mean(dirs):
    per_image_Rmean = []
    per_image_Gmean = []
    per_image_Bmean = []
    for dir in dirs:
        filenames = os.listdir(dir)
        for filename in tqdm.tqdm(filenames):
            img = cv2.imread(os.path.join(dir, filename), 1)
            per_image_Bmean.append(np.mean(img[:, :, 0]))
            per_image_Gmean.append(np.mean(img[:, :, 1]))
            per_image_Rmean.append(np.mean(img[:, :, 2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    return R_mean, G_mean, B_mean


if __name__ == '__main__':
    R, G, B = compute_rgb_mean(["YoukuDataset/image2/train/input", "YoukuDataset/image2/valid/input"])
    print("R={}, G={}, B={}".format(R, G, B))

