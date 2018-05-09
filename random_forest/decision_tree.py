import math
import random


class DecisionTree:
    def __init__(self, dataset, root_node):
        self._dataset = dataset
        self._root = root_node

    def classify(self, instance):
        return self._root.classify(instance, self._dataset)

    def print_(self):
        return self._root.print_()


class DecisionNode:
    def __init__(self, gain, feature, children):
        # gain: valor do Ganho de Informação associado a esta divisao
        # feature: indice da coluna do dataset
        # children: dicionario de sub-arvores para cada valor do feature
        self._gain = gain
        self._feature = feature
        self._children = children

    def classify(self, instance, dataset):
        value = dataset.value_for(instance, self._feature)
        child = self._children[value]
        return child.classify(instance, dataset)

    def print_(self, level=0):
        indent = ''.join(level * ['    '])

        line = 'DecisionNode (gain = %f, feature = %d):'
        print(indent + line % (self._gain, self._feature))

        for value, child in self._children.items():
            print(indent + '    value = %r:' % value)
            child.print_(level=level+2)


class LeafNode:
    def __init__(self, _class):
        self._class = _class

    def classify(self, instance, dataset):
        return self._class

    def print_(self, level=0):
        indent = ''.join(level * ['    '])
        line = 'LeafNode (_class = %r)' % self._class
        print(indent + line)


def build_tree(training_data, randomize=True):
    # training_data: objeto da classe Dataset
    # randomize:
    #     True se cada divisao deve considerar apenas uma amostra dos features
    #     False se cada divisao deve considerar todos os features

    features = training_data.features()
    num_features = len(features)

    if randomize:
        m = round(math.sqrt(num_features))
    else:
        m = num_features

    root_node = build_recursive(training_data, features, m)
    return DecisionTree(training_data, root_node)


def build_recursive(training_data, features, m):
    # training_data: objeto da classe Dataset
    # features: lista de indices das colunas do dataset
    # m: numero de features a serem considerados a cada divisao

    same_class = training_data.same_class_for_all_instances()
    if same_class is not None:
        return LeafNode(same_class)

    if len(features) == 0:
        return LeafNode(training_data.most_frequent_class())

    gain, feature = select_feature(training_data, random_sample(m, features))

    remaining_features = features.copy()
    remaining_features.remove(feature)

    children = {}
    for value in training_data.values_of(feature):
        subset = training_data.subset(feature, value)
        if subset.is_empty():
            children[value] = LeafNode(training_data.most_frequent_class())
        else:
            children[value] = build_recursive(subset, remaining_features, m)

    return DecisionNode(gain, feature, children)


def random_sample(sample_size, items):
    # sample_size: tamanho da amostra desejada
    # items: uma lista de itens quaisquer
    # Retorna uma lista de itens selecionados aleatoriamente.

    if sample_size < len(items):
        return random.sample(items, sample_size)
    else:
        return items


def select_feature(dataset, features):
    # dataset: objeto da classe Dataset
    # features: lista de indices de colunas do dataset
    # Retorna (ganho, indice) do "melhor" feature do subconjunto dado.

    max_gain, feature = max((dataset.info_gain(f), f) for f in features)
    return max_gain, feature
