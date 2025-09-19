import main_parts.config as cfg
import random

from PIL import Image
import numpy as np

def normalize_image(img_arr_uint):
    return img_arr_uint.astype(np.float64) * (1.0 / 255.0)

def load_image(filename):
    img_arr = load_image_sized(filename, cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)

    return img_arr


def load_image_sized(filename, image_width, image_height, image_depth):
    try:
        img = Image.open(filename)
        if img.height != image_height or img.width != image_width:
            img = img.resize((image_width, image_height))

        if image_depth == 1:
            img = img.convert('L')

        img_arr = np.asarray(img)

        if img.mode == 'L':
            h, w = img_arr.shape[:2]
            img_arr = img_arr.reshape(h, w, 1)

        return img_arr

    except Exception as e:
        print(f'failed to load image from {filename}: {e.message}')
        return None


def get_model(is_train):
    from parts.keras import KerasLinear
    from parts.interpreter import KerasInterpreter, TfLite

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    if is_train:
        interpreter = KerasInterpreter()
    else:
        interpreter = TfLite()

    kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
    return kl

def train_test_split(data_list,
                     shuffle = True,
                     test_size = 0.2):
    target_train_size = int(len(data_list) * (1. - test_size))

    if shuffle:
        train_data = []
        i_sample = 0
        while i_sample < target_train_size and len(data_list) > 1:
            i_choice = random.randint(0, len(data_list) - 1)
            train_data.append(data_list.pop(i_choice))
            i_sample += 1

        val_data = data_list

    else:
        train_data = data_list[:target_train_size]
        val_data = data_list[target_train_size:]

    return train_data, val_data
