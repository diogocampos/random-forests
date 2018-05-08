import numpy as np

import decision_tree


class Forest:
    def __init__(self, dataset, roots):
        # dataset: objeto Dataset que deu origem a esta floresta
        # roots: lista com a raiz de cada arvore da floresta
        self._dataset = dataset
        self._roots = roots

    def classify(self, instance):
        # instance: a instancia a ser classificada
        # Retorna a classe mais votada pelas arvores da floresta.

        votes = [root.classify(instance, self._dataset) for root in self._roots]
        classes, counts = np.unique(np.array(votes), return_counts=True)
        winner = classes[np.argmax(counts)]
        return winner


def random_forest(training_data, num_trees):
    # dataset: o Dataset de treinamento a partir do qual a floresta sera gerada
    # num_trees: numero de arvores a serem produzidas
    # Retorna uma Forest com `num_trees` arvores

    roots = []
    for i in range(num_trees):
        bootstrap = training_data.random_bootstrap()
        root = decision_tree.build_tree(bootstrap, randomize=True)
        roots.append(root)

    return Forest(training_data, roots)
