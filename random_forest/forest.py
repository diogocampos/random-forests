import numpy as np
from . import decision_tree


class Forest:
    def __init__(self, trees):
        # trees: lista com as arvores da floresta
        self._trees = trees

    def classify(self, instance):
        # instance: a instancia a ser classificada
        # Retorna a classe mais votada pelas arvores da floresta.

        votes = [tree.classify(instance) for tree in self._trees]
        classes, counts = np.unique(np.array(votes), return_counts=True)
        winner = classes[np.argmax(counts)]
        return winner


def random_forest(training_data, num_trees):
    # training_data: o Dataset a partir do qual a floresta sera gerada
    # num_trees: numero de arvores a serem produzidas
    # Retorna uma Forest com `num_trees` arvores

    trees = []
    for i in range(num_trees):
        bootstrap = training_data.random_bootstrap()
        tree = decision_tree.build_tree(bootstrap, randomize=True)
        trees.append(tree)

    return Forest(trees)


def cross_validation(dataset, num_trees, num_folds):
    # dataset: o Dataset completo a ser utilizado
    # num_trees: numero de arvores por floresta
    # num_folds: numero de folds do metodo k-fold cross-validation
    # Gera a sequencia das F1-measures obtidas (uma por fold).

    for training_data, test_data in dataset.random_folds(num_folds):
        forest = random_forest(training_data, num_trees)
        f1_score = test_data.evaluate_classifier(forest)
        yield f1_score
