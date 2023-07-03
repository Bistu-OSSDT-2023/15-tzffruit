import shutil
import cv2
import os
import os.path as osp
import numpy as np
from tqdm import tqdm


# 实际的图片保存和读取的过程中存在中文，所以这里通过这两种方式来应对中文读取的情况。
# handle chinese path
def cv_imread(file_path, type=-1):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    if type == 0:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    return cv_img


def cv_imwrite(file_path, cv_img, is_gray=True):
    if len(cv_img.shape) == 3 and is_gray:
        cv_img = cv_img[:, :, 0]
    cv2.imencode(file_path[-4:], cv_img)[1].tofile(file_path)


def data_clean(src_folder, english_name):
    clean_folder = src_folder + "_cleaned"
    if os.path.isdir(clean_folder):
        print("保存目录已存在")
        shutil.rmtree(clean_folder)
    os.mkdir(clean_folder)
    # 数据清晰的过程主要是通过oepncv来进行读取，读取之后没有问题就可以进行保存
    # 数据清晰的过程中，一是为了保证数据是可以读取的，二是需要将原先的中文修改为英文，方便后续的程序读取。
    image_names = os.listdir(src_folder)
    with tqdm(total=len(image_names)) as pabr:
        for i, image_name in enumerate(image_names):
            image_path = osp.join(src_folder, image_name)
            try:
                img = cv_imread(image_path)
                img_channel = img.shape[-1]
                if img_channel == 3:
                    save_image_name = english_name + "_" + str(i) + ".jpg"
                    save_path = osp.join(clean_folder, save_image_name)
                    cv_imwrite(file_path=save_path, cv_img=img, is_gray=False)
            except:
                print("{}是坏图".format(image_name))
            pabr.update(1)


if __name__ == '__main__':
    data_clean(src_folder="D:/upppppppppp/cls/cls_torch_tem/data/向日葵", english_name="sunflowers")
