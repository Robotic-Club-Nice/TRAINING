import datetime
from abc import ABC, abstractmethod

import numpy as np

from main_parts.utils import normalize_image
from pipeline.types import TubRecord
from parts.interpreter import KerasInterpreter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, Input, Convolution2D, Dropout, Flatten)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class KerasPilot(ABC):
    def __init__(self,
                 interpreter = KerasInterpreter(),
                 input_shape = (120, 160, 3)):
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        print(f'Created {self} with interpreter: {interpreter}')

    def load(self, model_path):
        print(f'Loading model {model_path}')
        self.interpreter.load(model_path)

    def load_weights(self, model_path, by_name = True):
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception(f"Unknown optimizer type: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    def get_input_shape(self, input_name):
        return self.interpreter.get_input_shape(input_name)

    def run(self, img_arr, *other_arr):
        norm_img_arr = normalize_image(img_arr)
        np_other_array = tuple(np.array(arr) for arr in other_arr)
        values = (norm_img_arr, ) + np_other_array
        input_dict = dict(zip(self.output_shapes()[0].keys(), values))
        return self.inference_from_dict(input_dict)

    def inference_from_dict(self, input_dict):
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(self, interpreter_out):
        pass

    def train(self,
              model_path,
              train_data,
              train_steps,
              batch_size,
              validation_data,
              validation_steps,
              epochs,
              verbose = 1,
              min_delta = .0005,
              patience = 5):
        assert isinstance(self.interpreter, KerasInterpreter)
        model = self.interpreter.model
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]

        tic = datetime.datetime.now()
        print('////////// Starting training //////////')
        history = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False)
        toc = datetime.datetime.now()
        print(f'////////// Finished training in: {toc - tic} //////////')

        return history.history

    def x_transform(self, record, img_processor):
        assert isinstance(record, TubRecord), "TubRecord required"
        img_arr = record.image(processor=img_processor)
        return {'img_in': img_arr}

    def y_transform(self, record):
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self):
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self):
        return {}

    def __str__(self):
        return type(self).__name__


class KerasLinear(KerasPilot):
    def __init__(self,
                 interpreter = KerasInterpreter(),
                 input_shape = (120, 160, 3),
                 num_outputs = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def y_transform(self, record):
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle = record.underlying['user/angle']
        throttle = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))

def core_cnn_layers(img_in, drop, l4_stride=1):
    x = img_in
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x

def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model
