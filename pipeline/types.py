from copy import copy
import os
from parts.tub_v2 import Tub
from main_parts.utils import load_image


class TubRecord(object):
    def __init__(self, config, base_path, underlying):
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._image = None

    def __copy__(self):
        tubrec = TubRecord(self.config,
                           copy(self.base_path),
                           copy(self.underlying))
        tubrec._image = self._image
        return tubrec

    def image(self, processor=None, as_nparray=True):
        if self._image is None:
            _image = self._extract_image(as_nparray, processor)
        else:
            _image = self._image
            if processor:
                _image = processor(_image)
        return _image

    def _extract_image(self, as_nparray, processor):
        image_path = self.underlying['cam/image_array']
        full_path = os.path.join(self.base_path, 'images', image_path)
        _image = load_image(full_path)
        self._image = _image
        if processor:
            _image = processor(_image)
            self._image = _image
        return _image

    def __repr__(self):
        return repr(self.underlying)


class TubDataset(object):

    def __init__(self, config, tub_paths):
        self.config = config
        self.tub_paths = tub_paths
        self.tubs = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records = list()

    def get_records(self):
        if not self.records:
            print(f'Loading tubs from paths {self.tub_paths}')
            for tub in self.tubs:
                for underlying in tub:
                    record = TubRecord(self.config, tub.base_path, underlying)
                    self.records.append(record)
        return self.records

    def close(self):
        for tub in self.tubs:
            tub.close()
