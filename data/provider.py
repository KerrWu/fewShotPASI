import os
import tensorflow as tf


class SourceDomainData:

    def read_txt_file(self, txt_file):
        img_paths = []
        labels = []

        for line in open(txt_file, 'r'):
            try:
                items = line.strip().split(',')
                if self.data_root_dir:
                    img_paths.append(os.path.join(self.data_root_dir, items[0]))
                else:
                    img_paths.append(items[0])
                labels.append(int(items[1]))
            except:
                print(line)
                raise ValueError



        return img_paths, labels

    def __init__(self, pos_txt_file, neg_txt_file, batch_size, image_size, buffer_scale=10, is_train=True, data_root_dir=None):

        self.image_size = image_size
        self.batch_size = batch_size
        self.pos_txt_file = pos_txt_file
        self.neg_txt_file = neg_txt_file
        #txt list file, stored as: imagename label_id
        self.is_train = is_train
        self.data_root_dir = data_root_dir

        buffer_size = batch_size * buffer_scale

        # 读取图片
        self.pos_img_paths, self.pos_labels = self.read_txt_file(pos_txt_file)
        self.pos_labels = tf.convert_to_tensor(self.pos_labels, dtype=tf.int32)
        self.pos_dataset_size = len(self.pos_img_paths)
        print("num of train positive datas=", self.pos_dataset_size)

        self.neg_img_paths, self.neg_labels= self.read_txt_file(neg_txt_file)
        self.neg_labels = tf.convert_to_tensor(self.neg_labels, dtype=tf.int32)
        self.neg_dataset_size = len(self.neg_img_paths)
        print("num of train negative datas=", self.neg_dataset_size)

        # 转换成Tensor
        self.pos_img_paths = tf.convert_to_tensor(self.pos_img_paths, dtype=tf.string)
        self.neg_img_paths = tf.convert_to_tensor(self.neg_img_paths, dtype=tf.string)

        # 创建数据集
        pos_data = tf.data.Dataset.from_tensor_slices((self.pos_img_paths, self.pos_labels))
        print("data type=", type(pos_data))
        pos_data = pos_data.map(self.parse_function)
        pos_data = pos_data.repeat(-1)
        pos_data = pos_data.shuffle(buffer_size=buffer_size//2)
        pos_data = pos_data.batch(batch_size//2)

        neg_data = tf.data.Dataset.from_tensor_slices((self.neg_img_paths, self.neg_labels))
        print("data type=", type(neg_data))
        neg_data = neg_data.map(self.parse_function)
        neg_data = neg_data.repeat(-1)
        neg_data = neg_data.shuffle(buffer_size=buffer_size//2)
        neg_data = neg_data.batch(batch_size//2)

        self.data = neg_data.concatenate(pos_data)
        self.data = self.data.map(self.double_label)

    def double_label(self, filename_batch, label_batch):

        return filename_batch, label_batch, label_batch

    def augment_dataset(self, image, size):

        distorted_image = tf.image.random_brightness(image,max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.5, upper=1.5)
        distorted_image = tf.image.random_crop(distorted_image, [self.image_size[0], self.image_size[1], 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        return float_image

    def parse_function(self, filename, label):

        # img = tf.read_file(filename)
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = self.augment_dataset(img, self.image_size)

        return img, label


class TargetDomainData:

    def read_txt_file(self):
        self.img_paths = []
        self.labels = []

        for line in open(self.txt_file, 'r'):
            try:
                items = line.strip().split(',')
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
        data = data.repeat(-1)
        data = data.shuffle(buffer_size=buffer_size)

        # 设置self data Batch
        self.data = data.batch(batch_size)
        print("self.data type=", type(self.data))

    def augment_dataset(self, image, size):

        distorted_image = tf.image.random_brightness(image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5)
        # distorted_image = tf.image.random_crop(distorted_image, [self.image_size[0], self.image_size[1], 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        return distorted_image

    def parse_function(self, filename, label):
        print(filename)
        label_ = tf.cast(label, tf.float32)

        # img = tf.read_file(filename)
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize_images(img, [self.image_size[0], self.image_size[1]])
        if self.is_train:
            img = self.augment_dataset(img, self.image_size)

        # Subtract off the mean and divide by the variance of the pixels.
        img = tf.image.per_image_standardization(img)

        return img, label_



if __name__ == "__main__":
    pos_txt_file = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/source_data/pos.txt"
    neg_txt_file = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/source_data/neg.txt"
    batch_size = 16
    image_size = (224, 224)
    data_root_dir = "/media/wz209/a29353b7-1090-433f-b452-b4ce827adb17/wz/PASI/all_data/source_data"

    tmp = SourceDomainData(pos_txt_file, neg_txt_file, batch_size, image_size, data_root_dir=data_root_dir)
    print(tmp.data)