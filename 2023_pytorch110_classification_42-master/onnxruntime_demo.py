import numpy as np
import onnxruntime
from PIL import Image

class_names = {'0': '雏菊', '1': '蒲公英', '2': '玫瑰', '3': '向日葵', '4': '郁金香'}

# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差
# 预测图片
session = onnxruntime.InferenceSession(r"C:\Users\nongc\Desktop\ImageClassifier.onnx")


def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,
                    top_margin))
    # img.save('thumb.jpg')
    # 相同的预处理方法
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))

    return img


image_path = r"C:\Users\nongc\Desktop\百度云下载\2023_pytorch110_classification_42-master\2023_pytorch110_classification_42-master\flowers_5\roses\99383371_37a5ac12a3_n.jpg"  # '1':
img = process_image(image_path)
img = np.expand_dims(img, 0)

outputs = session.run([], {"modelInput": img.astype('float32')})
result_index = int(np.argmax(np.squeeze(outputs)))
result = class_names['%d' % result_index]  # 获得对应的名称

print(np.squeeze(outputs), '\n', img.shape)
print(f"预测种类为： {result} 对应索引为：{np.argmax(np.squeeze(outputs))}")
# print(np.min(outputs),np.argmin(np.squeeze(outputs)),np.max(outputs))
