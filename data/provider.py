import os
import tensorflow as tf


class SourceDomainData:
    def read_txt_file(self):
        self.img_paths = []
        self.labels = []

        for line in open(self.txt_file, 'r'):
            items = line.split(' ')
            self.img_paths.append(items[0])
            self.labels.append(int(items[1]))

    def __init__(self, txt_file, batch_size, num_classes, image_size, buffer_scale=10):

        self.image_size = image_size
        self.batch_size = batch_size
        self.txt_file = txt_file
        #txt list file, stored as: imagename label_id
        self.num_classes = num_classes

        buffer_size = batch_size * buffer_scale

        # 读取图片
        self.read_txt_file()
        self.dataset_size = len(self.labels)
        print("num of train datas=", self.dataset_size)

        # 转换成Tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)

        # 创建数据集
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        print("data type=", type(data))
        data = data.map(self.parse_function)
        data = data.repeat(1000)
        data = data.shuffle(buffer_size=buffer_size)

        # 设置self data Batch
        self.data = data.batch(batch_size)
        print("self.data type=", type(self.data))

    def augment_dataset(self, image, size):

        distorted_image = tf.image.random_brightness(image,max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.5, upper=1.5)
        distorted_image = tf.image.random_crop(distorted_image, [self.image_size[0], self.image_size[1], 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        return float_image

    def parse_function(self, filename, label):

        label_ = tf.one_hot(label, self.num_classes)

        # img = tf.read_file(filename)
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = self.augment_dataset(img, self.image_size)

        return img, label_


class TargetDomainData:

    def read_txt_file(self):
        self.img_paths = []
        self.labels = []

        for line in open(self.txt_file, 'r'):
            try:
                items = line.split(',')
                if self.data_root_dir:
                    self.img_paths.append(os.path.join(self.data_root_dir, items[0]))
                else:
                    self.img_paths.append(items[0])
                self.labels.append([float(elem) for elem in items[1:]])
            except:
                print(line)
                raise ValueError

    def __init__(self, txt_file, batch_size, image_size, buffer_scale=10, is_train=True, data_root_dir=None):

        self.txt_file = txt_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.data_root_dir = data_root_dir
        buffer_size = batch_size * buffer_scale

        # 读取图片
        self.read_txt_file()
        self.dataset_size = len(self.labels)

        if is_train:
            print("num of train data=", self.dataset_size)
        else:
            print("num of valid data=", self.dataset_size)

        # 转换成Tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.float32)

        # 创建数据集
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        print("data type=", type(data))
        data = data.map(self.parse_function)
        # data = data.repeat(1000)
        data = data.shuffle(buffer_size=buffer_size)

        # 设置self data Batch
        self.data = data.batch(batch_size)
        print("self.data type=", type(self.data))

    def augment_dataset(self, image, size):

        distorted_image = tf.image.random_brightness(image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
        distorted_image = tf.image.random_crop(distorted_image, [self.image_size[0], self.image_size[1], 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        return distorted_image

    def parse_function(self, filename, label):

        label_ = tf.cast(label, tf.float32)

        # img = tf.read_file(filename)
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        if self.is_train:
            img = self.augment_dataset(img, self.image_size)

        # Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

        return img, label_

