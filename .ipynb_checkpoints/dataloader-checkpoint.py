import numpy as np
import os
import tensorflow as tf
from preprocessing import vgg_preprocessing
class DataLoader:
    def normalize(data):
        data = data.astype(np.float32)
        if len(data.shape) == 3:
            data = np.expand_dims(data, -1)
        if data.shape[-1] == 1:
            data = np.tile(data, [1, 1, 1, 3])
        data = data / 255.0
        return data

    def get_dataset_from_folder(dataset_name,
                                dataset_root,
                                csv_file = "",
                                split = "train",
                                batch_size = 128,
                                image_size = 224,
                                no_workers = 4,
                                shuffle = False,
                                is_training=False):
        dataset_dir = os.path.join(dataset_root, dataset_name, split)
        def load_data_from_folder():
            classes = os.listdir(dataset_dir)
            classes.sort()
            image_paths = []
            labels = []
            for i, cls in enumerate(classes):
                cls_dir = os.path.join(dataset_dir, cls)
                image_names = os.listdir(cls_dir)
                labels.extend([i] * len(image_names))
                for name in image_names:
                    image_path = os.path.join(cls_dir, name)
                    image_paths.append(image_path)
                assert len(image_paths) == len(labels)
            return image_paths, labels

        def load_data_from_csv(csv_input):
            image_paths = []
            labels = []
            with open(csv_input, "r") as fi:
                for line in fi:
                    info = line.strip().split(",")
                    image_paths.append(os.path.join(dataset_root,info[1]))
                    labels.append(int(info[0]))
                assert len(image_paths) == len(labels)
            no_classes = max(labels) + 1
            return image_paths, labels, no_classes

        def preprocess(image_path, label):
            image = tf.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels = 3)
#             image = tf.image.resize_images(image, [image_size, image_size])
            image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training)
#             image = image * 1.0 / 127.5 - 1.0
            return image, label

        if (csv_file != ""):
            image_paths, labels, no_classes = load_data_from_csv(csv_file)
        else:
            image_paths, labels, no_classes = load_data_from_folder()

        no_samples = len(image_paths)
        image_paths = tf.convert_to_tensor(image_paths, dtype = tf.string)
        labels = tf.convert_to_tensor(labels, dtype = tf.int32)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        if shuffle == True:
            dataset = dataset.shuffle(no_samples)
        dataset = dataset.map(preprocess, no_workers)
        dataset = dataset.batch(batch_size = batch_size).repeat()
        images, labels = dataset.make_one_shot_iterator().get_next()

        return images, labels, no_samples, no_classes

