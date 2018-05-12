#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Random Forests

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


import sys
from random_forest import datasets, forest


USAGE = """
Uso:  $ %s DATASET
DATASET pode ser 'pima', 'wine', 'ionosphere', ou 'wdbc'.
""".strip()


# Parametros
FOLDS = 10
MAX_TREES = 100


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

    header = ['Ntree'] + ['Fold%d' % (i+1) for i in range(FOLDS)] + ['Mean']
    print(','.join(header))

    for ntree in range(5, MAX_TREES + 1, 5):
        print('%s,' % ntree, end='', flush=True)

        f1_scores = []
        for score in forest.cross_validation(dataset, ntree, FOLDS):
            f1_scores.append(score)
            print('%s,' % score, end='', flush=True)

        mean = sum(f1_scores) / len(f1_scores)
        print(mean)


if __name__ == '__main__':
    try:
        main(sys.argv)
    except KeyboardInterrupt:
        sys.exit(1)
