import numpy as np


class Dataset:
    def __init__(self, features, classes, type):
        self._features = normalize(np.array(features))
        self._classes = np.array(classes)
        self._type = type


CATEGORIAL = 1
NUMERIC = 2

def load(filename, separator=',', has_header=True, type=NUMERIC):
    instances = []
    classes = []
    convert = float if type is NUMERIC else str

    with open('datasets/' + filename) as file:
        if has_header:
            file.readline()  # pula a primeira linha

        while True:
            line = file.readline()
            if not line: break

            values = line.split(separator)
            instances.append([convert(val) for val in values[:-1]])
            classes.append(values[-1])

    return Dataset(instances, classes, type)


def normalize(instances):
    pass  # TODO


def benchmark():
    return load('dadosBenchmark_validacaoAlgoritmoAD.csv', type=CATEGORIAL)

def ionosphere():
    return load('ionosphere.data', has_header=False)

def pima():
    return load('pima.tsv', separator='\t')

def wdbc():
    return load('wdbc.data', has_header=False)

def wine():
    return load('wine.data', has_header=False)
