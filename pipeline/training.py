import math
import os


import main_parts.config as cfg
from parts.interpreter import keras_model_to_tflite
from pipeline.sequence import TubSequence
from pipeline.types import TubDataset
from main_parts.utils import get_model, normalize_image, train_test_split
import tensorflow as tf
import albumentations as A


class BatchSequence(object):
    def __init__(self,
                 model,
                 config,
                 records,
                 is_train):
        self.model = model
        self.config = config
        self.sequence = TubSequence(records)
        self.batch_size = 128
        self.is_train = is_train
        self.pipeline = self._create_pipeline()

    def __len__(self):
        return math.ceil(len(self.pipeline) / self.batch_size)

    def image_processor(self, img_arr):
        img_arr1 = img_arr
        if (self.is_train):
            transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.4), contrast_limit=(-0.4, 0.2), p=0.5),
                A.RandomGamma(gamma_limit=(40, 160), p=0.3),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
                A.OneOf([
                    A.MotionBlur(blur_limit=(13, 17)),
                    A.GaussianBlur(blur_limit=8,  sigma_limit=(0.5, 7))
                ], p=0.25),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.4),
                    src_radius=20,
                    src_color=(255, 255, 255),
                    angle_range=(0.25, 0.25),
                    num_flare_circles_range=(1, 2),
                    method="physics_based",
                    p=0.1)
                ])
            img_arr1 = transform(image=img_arr)["image"]
        return img_arr1

    def _create_pipeline(self):
        def get_x(record):
            out_dict = self.model.x_transform(record, self.image_processor)
            out_dict['img_in'] = normalize_image(out_dict['img_in'])
            return out_dict

        def get_y(record):
            y = self.model.y_transform(record)
            return y

        pipeline = self.sequence.build_pipeline(x_transform=get_x,
                                                y_transform=get_y)
        return pipeline

    def create_tf_data(self):
        dataset = tf.data.Dataset.from_generator(
            generator=lambda: self.pipeline,
            output_types=self.model.output_types(),
            output_shapes=self.model.output_shapes())
        return dataset.repeat().batch(self.batch_size)


def train(tub_paths):
    base_path = os.path.abspath("./models/mypilot")

    kl = get_model(True)
    kl.interpreter.summary()

    tubs = tub_paths.split(',')
    all_tub_paths = [os.path.expanduser(tub) for tub in tubs]
    dataset = TubDataset(config=cfg, tub_paths=all_tub_paths)
    training_records, validation_records \
        = train_test_split(dataset.get_records(), shuffle=True,
                           test_size=(1. - 0.8))
    print(f'Records # Training {len(training_records)}')
    print(f'Records # Validation {len(validation_records)}')
    dataset.close()

    training_pipe = BatchSequence(kl, cfg, training_records, is_train=True)
    validation_pipe = BatchSequence(kl, cfg, validation_records, is_train=False)
    tune = tf.data.experimental.AUTOTUNE
    dataset_train = training_pipe.create_tf_data().prefetch(tune)
    dataset_validate = validation_pipe.create_tf_data().prefetch(tune)

    train_size = len(training_pipe)
    val_size = len(validation_pipe)

    h5_model_path = f'{base_path}.h5'
    tf_lite_model_path = f'{base_path}.tflite'
    history = kl.train(model_path=h5_model_path,
                       train_data=dataset_train,
                       train_steps=train_size,
                       batch_size=128,
                       validation_data=dataset_validate,
                       validation_steps=val_size,
                       epochs=100,
                       verbose=True,
                       min_delta=.0005,
                       patience=5)

    keras_model_to_tflite(h5_model_path, tf_lite_model_path)

    return history
