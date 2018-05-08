import random


class DecisionNode:
    def __init__(self, feature, children):
        self._feature = feature
        self._children = children

    def predict(self, instance, dataset):
        value = dataset.value_for(instance, self._feature)
        child = self._children[value]
        return child.predict(instance)


class LeafNode:
    def __init__(self, _class):
        self._class = _class

    def predict(self, instance, dataset):
        return self._class


def decision_tree(training_data, features, m):
    # training_data: objeto da classe Dataset
    # features: lista de indices das colunas do dataset
    # m: numero de features a serem considerados para selecao

    same_class = training_data.same_class_for_all_instances()
    if same_class:
        return LeafNode(same_class)

    if len(features) == 0:
        return LeafNode(training_data.most_frequent_class())

    feature = select_best_feature(training_data, random_sample(features, m))

    remaining_features = features.copy()
    remaining_features.remove(feature)

    children = {}
    for value in training_data.values_of(feature):
        subset = training_data.subset(feature, value)
        if subset.is_empty():
            children[value] = LeafNode(training_data.most_frequent_class())
        else:
            children[value] = decision_tree(subset, remaining_features, m)

    return DecisionNode(feature, children)


def random_sample(items, size):
    # Seleciona aleatoriamente `size` elementos de `items`.

    if len(items) <= size:
        return items
    else:
        return random.sample(features, size)


def select_best_feature(dataset, features):
    # Retorna o "melhor" feature do subconjunto dado.

    if len(features) == 1: return features[0]

    # TODO
