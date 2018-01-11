from model.cnn_base import ConvNet
from sklearn.model_selection import train_test_split
import cv2
import random
from data.load import loader


def baseline():
    # 数据集结构
    dataset = {"train_images": None, "train_labels": None, "valid_images": None, "valid_labels": None}
    # 划分训练集验证集
    images = loader()
    (dataset["train_images"], dataset["valid_images"], dataset["train_labels"], dataset[
        "valid_labels"]) = train_test_split(images["images"],images["clazz"], test_size=0.3, random_state=42)
    # dataset["train_images"], dataset["train_labels"] = augmentation(dataset["train_images"], dataset["train_labels"])
    conv_net = ConvNet()
    conv_net.train(dataset=dataset, n_epoch=5, batch_size=128)


# 数据增强
def augmentation(images, labels):
    print("正在进行数据增强")

    # 水平翻转
    def _filp(image):
        return cv2.flip(image, 1)

    # 增加噪声
    def _gauss(image):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for k in range(image.shape[2]):
                    image[i][j][k] += random.gauss(0, 0.01)
        return image

    # 图像竖直方向裁剪

    return images, labels


if __name__ == "__main__":
    baseline()
