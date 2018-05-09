#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Random Forests

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


from random_forest import datasets, decision_tree

dataset = datasets.load_dataset('benchmark')
tree = decision_tree.build_tree(dataset, randomize=False)
tree.print_()
