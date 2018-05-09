#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Random Forests

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


from random_forest import datasets, forest


# Parametros
DATASET = datasets.load_pima_dataset()
NUM_FOLDS = 10
MAX_NUM_TREES = 50


def main():
    # Executa k-fold cross-validation com os parametros definidos acima.
    # Imprime os resultados em formato CSV: cada linha contem o valor de
    # `NTREE` seguido de `K` F1-scores resultantes de cada um dos `K` folds.

    for num_trees in range(5, MAX_NUM_TREES + 1, 5):
        print(num_trees, end='', flush=True)

        scores = forest.cross_validation(DATASET, num_trees, NUM_FOLDS)
        for f1_score in scores:
            print(',%s' % f1_score, end='', flush=True)

        print()


if __name__ == '__main__':
    main()
