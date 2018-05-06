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
from collections import Counter

# INPUT_FILE = 'datasets/pima.tsv'
INPUT_FILE = 'datasets/dadosBenchmark_validacaoAlgoritmoAD.csv'

class DecisionTree():
    root = None
    instances = []
    feature_names = []

    def __init__(self, instances, feature_names):
        self.instances = instances
        self.feature_names = feature_names

    def build(self):
        #remove class column
        feature_columns = self.instances[:, :-1]
        klasses = self.instances[:, -1]
        features = []
        for index, col in enumerate(feature_columns.T):
            feature = Feature(self.feature_names[index], col)
            features.append(feature)

        self.root = self.build_recursive(features, klasses, self.feature_names)

        return self.root

    def most_common(self, klasses):
        b = Counter(klasses)
        return b.most_common(1)[0][0]

    def get_best_feature(self, features, klasses):
        max_gain = 0
        best_feature = None
        for feature in features:
            gain = self.entropy_gain(feature.values, klasses)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature

        return best_feature

    def split_features(self, features, best_feature, klasses, direction):
        split_features = []
        split_klasses = []
        first_feature = True
        #get rows that match the best feature split
        for feature in features:
            values = []
            for i, value in enumerate(feature.values):
                if is_float(best_feature.values[i]) and direction == 'left':
                    split_func = lambda a: a <= np.mean(best_feature.values)
                elif is_float(best_feature.values[i]) and direction == 'right':
                    split_func = lambda a: a > np.mean(best_feature.values)
                elif not is_float(best_feature.values[i]) and direction == 'left':
                    subset = random.sample(list(feature.values), 2)
                    split_func = lambda a: a in subset
                elif not is_float(best_feature.values[i]) and direction == 'right':
                    subset = random.sample(list(feature.values), 2)
                    split_func = lambda a,b: a not in subset

                if split_func(best_feature.values[i]):
                    values.append(value)
                    if first_feature:
                        split_klasses.append(klasses[i])

            first_feature = False
            split_feature = Feature(feature.name, values)
            split_features.append(split_feature)

        return split_features, split_klasses, split_func

    def build_recursive(self, features, klasses, feature_names):
        node = Node()
        if len(set(klasses)) ==  1:
            node.is_leaf = True
            node.feature = klasses[0]
            return node
        if len(feature_names) == 0:
            node.is_leaf = True
            node.feature = self.most_common(klasses)
            return node
        else:
            considered_features = [feature for feature in features if feature.name in feature_names]
            best_feature = self.get_best_feature(considered_features, klasses)
            node.feature = best_feature.name
            #remove best feature from list
            feature_names = [feature for feature in feature_names if feature != best_feature.name]

            left_features, left_klasses, left_function = self.split_features(features, best_feature, klasses, 'left')
            if len(left_klasses) == 0:
                node.feature = self.most_common(klasses)
                node.is_leaf = True
                return node

            node.left_function = left_function
            node.left = self.build_recursive(left_features, left_klasses, feature_names)

            right_features, right_klasses, right_function = self.split_features(features, best_feature, klasses, 'right')
            if len(right_klasses) == 0:
                node.feature = self.most_common(klasses)
                node.is_leaf = True
                return node

            node.right_function = right_function
            node.right = self.build_recursive(right_features, right_klasses, feature_names)

            return node

    def entropy_gain(self, feature, klasses):
        klasses_set = set(klasses)
        info_d = 0
        total = len(self.instances)
        for klass in klasses_set:
            total_in_klass = sum([1 for instance in self.instances if instance[-1] == klass])
            info_d -= (total_in_klass/total) * math.log((total_in_klass/total), 2)

        if is_float(feature[0]):
            return self.continuous_gain(feature, klasses, info_d, total, klasses_set)
        else:
            return self.categorical_gain(feature, klasses, info_d, total, klasses_set)

    def categorical_gain(self, feature, klasses, info_d, total, klasses_set):
        partitions = [lambda a,b: a in b, lambda a,b: a not in b]
        feature_with_klass = np.dstack((feature, klasses))[0]
        info_feature = 0
        subset = random.sample(list(feature), 2)
        for partition_func in partitions:
            total_partition = sum([1 for instance in feature if partition_func(instance, subset)])
            info_feature_partition = (total_partition/total)
            info_sum = 0
            for klass in klasses_set:
                total_in_klass = sum([1 for instance in feature_with_klass if instance[1] == klass
                                      and partition_func(instance[0], subset)])
                if total_in_klass != 0 and total_partition != 0:
                    info_sum -= (total_in_klass/total_partition)*math.log(total_in_klass/total_partition, 2)

            info_feature_partition *= info_sum
            info_feature += info_feature_partition

        gain = info_d - info_feature
        return gain

    #divide continuous features by their mean
    def continuous_gain(self, feature, klasses, info_d, total, klasses_set):
        partitions = [lambda a,b: a <= b, lambda a,b: a > b]
        mean = np.mean(feature)
        feature_with_klass = np.dstack((feature, klasses))[0]
        info_feature = 0
        for partition_func in partitions:
            total_partition = sum([1 for instance in feature if partition_func(instance, mean)])
            info_feature_partition = (total_partition/total)
            info_sum = 0
            for klass in klasses_set:
                total_in_klass = sum([1 for instance in feature_with_klass if instance[1] == klass
                                      and partition_func(instance[0], mean)])
                if total_in_klass != 0 and total_partition != 0:
                    info_sum -= (total_in_klass/total_partition)*math.log(total_in_klass/total_partition, 2)

            info_feature_partition *= info_sum
            info_feature += info_feature_partition

        gain = info_d - info_feature
        return gain

    def predict(self, instance):
        node = self.root

        while not node.is_leaf:
            value = instance[self.feature_names.index(node.feature)]
            if node.left_function(value):
                node = node.left
            elif node.right_function(value):
                node = node.right

        return node.feature


class Feature():
    name = ""
    values = []
    def __init__(self, name, values):
        self.name = name
        self.values = values


class Node():
    is_leaf = False
    left = None
    right = None
    left_function = None
    right_function = None
    feature = ""

    def __init__(self, feature = "", left = None, right = None):
        self.feature = feature
        self.left = left
        self.right = right

def main():
    feature_names, data = load_csv(INPUT_FILE, ';')
    data = np.array(data)
    tree  = DecisionTree(data, feature_names)
    tree.build()
    right = 0
    wrong = 0
    for pred in data:
        p = pred[:-1]
        if tree.predict(p) == pred[-1]:
            right += 1
        else:
            wrong += 1

    print(right/(right+wrong))

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def load_csv(filename, separator= ','):

    data = []

    with open(filename) as file:
        feature_names = file.readline().split(separator)[:-1]  # skip first line
        while True:
            line = file.readline()
            if not line: break

            fields = line.split(separator)
            # read all attributes as floats, except for the class
            instance = [float(val) if is_float(val) else val for val in fields[0:-1]] + \
                       [int(fields[-1]) if is_float(fields[-1]) else fields[-1]]
            data.append(instance)

    return feature_names, data

if __name__ == '__main__':
    main()
