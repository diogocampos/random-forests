import numpy as np


class Dataset:
    def __init__(self, features, classes, type):
        self._features = normalize(np.array(features))
        self._classes = np.array(classes)
        self._type = type

    def same_class_for_all_instances(self):
        # Se todas as instancias tem a mesma classe, retorna a classe.
        # Caso contrario, retorna None.
        pass  # TODO

    def most_frequent_class(self):
        # Retorna a classe mais frequente neste dataset.
        pass  # TODO

    def values_of(self, feature):
        # Retorna uma lista com os possiveis valores do feature dado.
        pass  # TODO

    def subset(self, feature, value):
        # Retorna o sub-Dataset das instancias cujo valor de feature == value.
        pass  # TODO

    def is_empty(self):
        # Retorna True se este dataset estiver vazio.
        pass  # TODO

    def value_for(self, instance, feature):
        # Retorna um valor discretizado para o feature da instancia dada.
        if self._type is NUMERIC:
            pass  # TODO
        else:
            return instance[feature]


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
    # instances: np.array 2D de features

    minimums = np.min(instances, axis=0)
    maximums = np.max(instances, axis=0)
    return (features - minimums) / (maximums - minimums)


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
