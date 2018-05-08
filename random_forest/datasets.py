import numpy as np


CATEGORIAL = 1
NUMERIC = 2

class Dataset:
    def __init__(self, instances, classes, type, parent=None):
        # instances: lista de listas de valores de features
        # classes: lista com as classes associadas a cada instancia
        # type: tipo do dataset, pode ser CATEGORIAL ou NUMERIC
        # parent: Dataset do qual este eh um subconjunto

        if type is NUMERIC and parent is None:
            self._instances = normalize_features(np.array(instances))
        else:
            self._instances = np.array(instances)

        self._classes = np.array(classes)
        self._type = type
        self._parent = parent

    def features(self):
        # Retorna uma lista de indices das colunas de features do dataset

        num_features = self._instances.shape[1]
        return range(num_features)

    def same_class_for_all_instances(self):
        # Se todas as instancias tem a mesma classe, retorna a classe.
        # Caso contrario, retorna None.

        _class = self._classes[0]
        same = np.all(self._classes == _class)
        return _class if same else None

    def most_frequent_class(self):
        # Retorna a classe mais frequente neste dataset.

        classes, counts = np.unique(self._classes, return_counts=True)
        most_frequent = classes[np.argmax(counts)]
        return most_frequent

    def values_of(self, feature):
        # feature: indice de uma coluna do dataset
        # Retorna uma lista com todos os possiveis valores do feature dado.
        # TODO: Adicionar suporte a features numericos.

        if self._parent is None:
            return np.unique(self._instances[:, feature])
        else:
            return self._parent.values_of(feature)

    def subset(self, feature, value):
        # feature: indice de uma coluna do dataset
        # value: valor do feature
        # Retorna o sub-Dataset das instancias cujo valor de feature == value.
        # TODO: Adicionar suporte a features numericos.

        indexes = np.where(self._instances[:, feature] == value)
        instances = self._instances[indexes]
        classes = self._classes[indexes]

        return Dataset(instances, classes, self._type, parent=self)

    def is_empty(self):
        # Retorna True se este dataset estiver vazio.

        return len(self._instances) == 0

    def value_for(self, instance, feature):
        # instance: lista com os valores de cada feature
        # feature: indice de uma coluna do dataset
        # Retorna um valor *discretizado* para o feature da instancia dada.
        # TODO: Adicionar suporte a features numericos.

        return instance[feature]


def load_benchmark_dataset():
    return load('dadosBenchmark_validacaoAlgoritmoAD.csv', separator=';')

def load_ionosphere_dataset():
    return load('ionosphere.data', has_header=False, type=NUMERIC)

def load_pima_dataset():
    return load('pima.tsv', separator='\t', type=NUMERIC)

def load_wdbc_dataset():
    return load('wdbc.data', has_header=False, type=NUMERIC)

def load_wine_dataset():
    return load('wine.data', has_header=False, type=NUMERIC)


def load(filename, separator=',', has_header=True, type=CATEGORIAL):
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


def normalize_features(instances):
    # instances: np.array 2D de features
    # Retorna um array de instancias normalizadas no intervalo [0, 1].

    minimums = np.min(instances, axis=0)
    maximums = np.max(instances, axis=0)
    return (instances - minimums) / (maximums - minimums)
