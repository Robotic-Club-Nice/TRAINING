class SizedIterator():
    def __init__(self):
        pass


class TubSeqIterator(SizedIterator):
    def __init__(self, records):
        self.records = records or list()
        self.current_index = 0

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return TubSeqIterator(self.records)

    def __next__(self):
        if self.current_index >= len(self.records):
            raise StopIteration('No more records')

        record = self.records[self.current_index]
        self.current_index += 1
        return record

    next = __next__


class TfmIterator(SizedIterator):
    def __init__(self,
                 iterable,
                 x_transform,
                 y_transform):

        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iterator = BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform)

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        return BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform)

    def __next__(self):
        return next(self.iterator)


class TfmTupleIterator(SizedIterator):
    def __init__(self,
                 iterable,
                 x_transform,
                 y_transform):

        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iterator = BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform)

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        return BaseTfmIterator_(
            iterable=self.iterable,
            x_transform=self.x_transform,
            y_transform=self.y_transform)

    def __next__(self):
        return next(self.iterator)


class BaseTfmIterator_(SizedIterator):

    def __init__(self,
                 iterable,
                 x_transform,
                 y_transform):

        self.iterable = iterable
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.iterator = iter(self.iterable)

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        return BaseTfmIterator_(
            self.iterable, self.x_transform, self.y_transform)

    def __next__(self):
        record = next(self.iterator)
        if isinstance(record, tuple) and len(record) == 2:
            x, y = record
            return self.x_transform(x), self.y_transform(y)
        else:
            return self.x_transform(record), self.y_transform(record)


class TubSequence():
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        return TubSeqIterator(self.records)

    def __len__(self):
        return len(self.records)

    def build_pipeline(self,
                       x_transform,
                       y_transform):
        return TfmIterator(self,
                           x_transform=x_transform,
                           y_transform=y_transform)

    @classmethod
    def map_pipeline(
            cls,
            x_transform,
            y_transform,
            pipeline):
        return TfmTupleIterator(pipeline,
                                x_transform=x_transform,
                                y_transform=y_transform)

    @classmethod
    def map_pipeline_factory(
            cls,
            x_transform,
            y_transform,
            factory):

        pipeline = factory()
        return cls.map_pipeline(pipeline=pipeline,
                                x_transform=x_transform,
                                y_transform=y_transform)
