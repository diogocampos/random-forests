#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Random-forests

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


import math
import random
import numpy as np

INPUT_FILE = 'datasets/pima.tsv'

class DecisionTree():
    tree = {}
    instances = []

    def __init__(self, instances):
        self.instances = instances

    def build(self):
        #remove class column
        features = self.instances[:, :-1]
        klasses = self.instances[:, -1]

        max_gain = 0
        best_feature = None
        for feature in features.T:
            gain = self.entropy_gain(feature, klasses)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature

        print(max_gain)

    #@TODO categorical features
    def entropy_gain(self, feature, klasses):
        klasses_set = set(klasses)
        info_d = 0
        total = len(self.instances)
        for klass in klasses_set:
            total_in_klass = sum([1 for instance in self.instances if instance[-1] == klass])
            info_d -= (total_in_klass/total) * math.log((total_in_klass/total), 2)

        #divide continuous features by their mean
        mean = np.mean(feature)
        partitions = [lambda a,b: a <= b, lambda a,b: a > b]
        feature_with_klass = np.dstack((feature, klasses))[0]
        info_feature = 0
        for partition_func in partitions:
            total_partition = sum([1 for instance in feature if partition_func(instance, mean)])
            info_feature_partition = (total_partition/total)
            info_sum = 0
            for klass in klasses_set:
                total_in_klass = sum([1 for instance in feature_with_klass if instance[1] == klass and partition_func(instance[0], mean)])
                if total_in_klass != 0 and total_partition != 0:
                    info_sum -= (total_in_klass/total_partition)*math.log(total_in_klass/total_partition, 2)

            info_feature_partition *= info_sum
            info_feature += info_feature_partition

        gain = info_d - info_feature
        return gain

class Node():
    left = None
    right = None
    feature = ""

    def __init__(self, feature, left, right):
        self.feature = feature
        self.left = left
        self.right = right

def main():
    data = np.array(load_csv(INPUT_FILE))
    tree  = DecisionTree(data)
    tree.build()


def load_csv(filename):
    data = []

    with open(filename) as file:
        file.readline()  # skip first line
        while True:
            line = file.readline()
            if not line: break

            # fields = line.split(',')
            fields = line.split('\t')
            # read all attributes as floats, except for the class
            instance = [float(val) for val in fields[0:-1]] + [int(fields[-1])]
            data.append(instance)

    return data


if __name__ == '__main__':
    main()
