import os
from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf
from tensorflow import keras


def keras_model_to_tflite(in_filename, out_filename):
    model = tf.keras.models.load_model(in_filename, compile=False)
    keras_to_tflite(model, out_filename)

def keras_to_tflite(model, out_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(out_filename, "wb").write(tflite_model)


class Interpreter(ABC):
    def __init__(self):
        self.input_keys = None
        self.output_keys = None
        self.shapes = None

    @abstractmethod
    def load(self, model_path):
        pass

    def load_weights(self, model_path, by_name=True):
        raise NotImplementedError('Requires implementation')

    def set_model(self, pilot):
        pass

    def set_optimizer(self, optimizer):
        pass

    def compile(self, **kwargs):
        raise NotImplementedError('Requires implementation')

    @abstractmethod
    def get_input_shape(self, input_name):
        pass

    def predict(self, img_arr, *other_arr):
        input_dict = dict(zip(self.input_keys, (img_arr, *other_arr)))
        return self.predict_from_dict(input_dict)

    def predict_from_dict(self, input_dict):
        pass

    def summary(self):
        pass

    def __str__(self):
        return type(self).__name__


class KerasInterpreter(Interpreter):

    def __init__(self):
        super().__init__()
        self.model = None

    def set_model(self, pilot):
        self.model = pilot.create_model()
        input_shape = self.model.input_shape
        if type(input_shape) is not list:
            input_shape = [input_shape]
        output_shape = self.model.output_shape
        if type(output_shape) is not list:
            output_shape = [output_shape]

        self.input_keys = self.model.input_names
        self.output_keys = self.model.output_names
        self.shapes = (dict(zip(self.input_keys, input_shape)),
                       dict(zip(self.output_keys, output_shape)))

    def set_optimizer(self, optimizer):
        self.model.optimizer = optimizer

    def get_input_shape(self, input_name):
        assert self.model, 'Model not set'
        return self.shapes[0][input_name]

    def compile(self, **kwargs):
        assert self.model, 'Model not set'
        self.model.compile(**kwargs)

    def predict_from_dict(self, input_dict):
        for k, v in input_dict.items():
            input_dict[k] = self.expand_and_convert(v)
        outputs = self.model(input_dict, training=False)
        if type(outputs) is list:
            output = [output.numpy().squeeze(axis=0) for output in outputs]
            return output
        else:
            return outputs.numpy().squeeze(axis=0)

    def load(self, model_path):
        print(f'Loading model {model_path}')
        self.model = keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path, by_name=True):
        assert self.model, 'Model not set'
        self.model.load_weights(model_path, by_name=by_name)

    def summary(self):
        return self.model.summary()

    @staticmethod
    def expand_and_convert(arr):
        arr_exp = np.expand_dims(arr, axis=0)
        return arr_exp


class TfLite(Interpreter):

    def __init__(self):
        super().__init__()
        self.interpreter = None
        self.runner = None
        self.signatures = None

    def load(self, model_path):
        assert os.path.splitext(model_path)[1] == '.tflite', \
            'TFlitePilot should load only .tflite files'
        print(f'Loading model {model_path}')
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.signatures = self.interpreter.get_signature_list()
        self.runner = self.interpreter.get_signature_runner()
        self.input_keys = self.signatures['serving_default']['inputs']
        self.output_keys = self.signatures['serving_default']['outputs']

    def compile(self, **kwargs):
        pass

    def predict_from_dict(self, input_dict):
        for k, v in input_dict.items():
            input_dict[k] = self.expand_and_convert(v)
        outputs = self.runner(**input_dict)
        ret = list(outputs[k][0] for k in self.output_keys)
        return ret if len(ret) > 1 else ret[0]

    def get_input_shape(self, input_name):
        assert self.interpreter is not None, "Need to load tflite model first"
        details = self.interpreter.get_input_details()
        for detail in details:
            if detail['name'] == f"serving_default_{input_name}:0":
                return detail['shape']
        raise RuntimeError(f'{input_name} not found in TFlite model')

    @staticmethod
    def expand_and_convert(arr):
        arr_exp = np.expand_dims(arr, axis=0).astype(np.float32)
        return arr_exp
