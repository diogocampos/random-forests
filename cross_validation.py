#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Random Forests

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


from random_forest import datasets, forest


NUM_FOLDS = 10
NUM_TREES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

dataset = datasets.load_pima_dataset()

for num_trees in NUM_TREES:
    print(num_trees, end='', flush=True)

    for f1_score in forest.cross_validation(dataset, num_trees, NUM_FOLDS):
        print(',%s' % f1_score, end='', flush=True)

    print()
