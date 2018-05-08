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

        self._entropy = None  # cache do resultado do calculo da entropia

        if type is NUMERIC:
            # pontos de corte de cada feature
            self._thresholds = np.mean(self._instances, axis=0)


    def size(self):
        # Retorna o numero de instancias deste dataset.

        return len(self._instances)


    def is_empty(self):
        # Retorna True se este dataset estiver vazio.

        return len(self._instances) == 0


    def features(self):
        # Retorna uma lista de indices das colunas de features do dataset

        num_features = self._instances.shape[1]
        return list(range(num_features))


    def values_of(self, feature):
        # feature: indice de uma coluna do dataset
        # Retorna uma lista com todos os possiveis valores do feature dado.

        if self._parent is not None:
            return self._parent.values_of(feature)

        if self._type is CATEGORIAL:
            return np.unique(self._instances[:, feature])
        else:
            # Para features numericos, os possiveis valores sao:
            #    False: o valor eh menor ou igual ao ponto de corte
            #    True: o valor eh maior que o ponto de corte
            return [False, True]


    def value_for(self, instance, feature):
        # instance: lista com os valores de cada feature
        # feature: indice de uma coluna do dataset
        # Retorna um valor *discretizado* para o feature da instancia dada.

        if self._type is CATEGORIAL:
            return instance[feature]
        else:
            return instance[feature] > self._thresholds[feature]


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


    def info_gain(self, feature):
        # feature: indice de uma coluna do dataset
        # Retorna o Ganho de Informacao do feature neste dataset.

        return self.entropy() - self.feature_entropy(feature)


    def entropy(self):
        # Retorna Info(D), a entropia deste dataset.

        if self._entropy is None:
            classes, counts = np.unique(self._classes, return_counts=True)
            probabilities = counts / len(self._classes)
            self._entropy = -np.sum(probabilities * np.log2(probabilities))

        return self._entropy


    def feature_entropy(self, feature):
        # feature: indice de uma coluna do dataset
        # Retorna InfoA(D), a entropia do feature neste dataset.

        values = self.values_of(feature)
        subsets = (self.subset(feature, val) for val in values)
        return sum((s.size() / self.size()) * s.entropy() for s in subsets)


    def subset(self, feature, value):
        # feature: indice de uma coluna do dataset
        # value: valor do feature
        # Retorna o sub-Dataset das instancias cujo valor de feature == value.

        values = self._instances[:, feature]  # extrai a coluna do dataset
        if self._type is NUMERIC:
            values = values > self._thresholds[feature]

        indexes = np.where(values == value)
        instances = self._instances[indexes]
        classes = self._classes[indexes]

        return Dataset(instances, classes, self._type, parent=self)


    def random_bootstrap(self):
        # Retorna um Dataset com uma amostra aleatoria das instancias

        size = len(self._instances)
        indexes = np.random.choice(size, size, replace=True)
        instances = self._instances[indexes]
        classes = self._classes[indexes]

        return Dataset(instances, classes, self._type, parent=self)


def load_benchmark_dataset():
    return load('dadosBenchmark_validacaoAlgoritmoAD.csv', separator=';')

def load_pima_dataset():
    return load('pima.tsv', separator='\t', type=NUMERIC)

def load_wine_dataset():
    return load('wine.data', has_header=False, type=NUMERIC)

def load_ionosphere_dataset():
    return load('ionosphere.data', has_header=False, type=NUMERIC)

def load_wdbc_dataset():
    return load('wdbc.data', has_header=False, type=NUMERIC)


def load(filename, separator=',', has_header=True, type=CATEGORIAL):
    instances = []
    classes = []
    convert = float if type is NUMERIC else str

    with open('datasets/' + filename) as file:
        if has_header:
            file.readline()  # descarta a primeira linha

        while True:
            line = file.readline().strip()
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
