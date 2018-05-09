#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Random Forests

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


import sys
from random_forest import datasets, forest


USAGE = '''
Uso:  $ %s DATASET
DATASET pode ser 'benchmark', 'pima', 'wine', 'ionosphere', ou 'wdbc'.
'''.strip()


# Parametros
FOLDS = 10
MAX_TREES = 50


def main(argv):
    # Executa k-fold cross-validation com os parametros definidos acima.
    # Imprime os resultados em formato CSV: cada linha contem o valor de
    # NTREE seguido de K F1-scores resultantes de cada um dos K folds.

    try:
        dataset_name = argv[1]
        dataset = datasets.load_dataset(dataset_name.lower())
    except IndexError:
        print(USAGE % argv[0], file=sys.stderr)
        sys.exit(1)


    for ntree in range(5, MAX_TREES + 1, 5):
        print(ntree, end='', flush=True)

        f1_scores = forest.cross_validation(dataset, ntree, FOLDS)
        for score in f1_scores:
            print(',%s' % score, end='', flush=True)

        print()


if __name__ == '__main__':
    main(sys.argv)
