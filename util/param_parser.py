# some extra parameter parsers

import argparse

class DictParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(DictParser, self).__init__(*args, **kwargs)
        self.local_dict = {}

    def __call__(self, parser, namespace, values, option_string = None):

        for kv in values.split(','):
            k, v = kv.split('=')
            try:
                self.local_dict[k] = float(v)
            except:
                self.local_dict[k] = v
        setattr(namespace, self.dest, self.local_dict)

class ListParser(argparse.Action):

    def __init__(self, * args, **kwargs):

        super(ListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string = None):

        self.local_list = values.split(',')

        setattr(namespace, self.dest, self.local_list)

class IntListParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(IntListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string = None):

        self.local_list = list(map(int, values.split(',')))

        setattr(namespace, self.dest, self.local_list)

class FloatListParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(FloatListParser, self).__init__(*args, **kwargs)
        self.local_list = []

    def __call__(self, parser, namespace, values, option_string = None):

        self.local_list = list(map(float, values.split(',')))

        setattr(namespace, self.dest, self.local_list)

class BooleanParser(argparse.Action):

    def __init__(self, *args, **kwargs):

        super(BooleanParser, self).__init__(*args, **kwargs)
        self.values = None

    def __call__(self, parser, namespace, values, option_string = None):

        self.values = False if int(values) == 0 else True

        setattr(namespace, self.dest, self.values)
