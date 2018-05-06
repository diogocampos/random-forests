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

        self.root = self.build_recursive(features, klasses)

        return self.root


    def get_best_feature(self, features, klasses):
        max_gain = 0
        best_feature = None
        for feature in features:
            gain = self.entropy_gain(feature.values, klasses)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature

        return best_feature

    def build_recursive(self, features, klasses):
        node = Node()
        if len(set(klasses)) ==  1:
            node.is_leaf = True
            node.feature = klasses[0]
            return node
        if len(features) == 0:
            #@TODO get most common class
            node.is_leaf = True
            node.feature = 1
            return node
        else:
            best_feature = self.get_best_feature(features, klasses)
            node.feature = best_feature.name
            #remove best feature from list
            features = [feature for feature in features if feature.name != best_feature.name]

            #left node
            split_features = []
            split_klasses = []
            first_feature = True
            for feature in features:
                values = []
                for i, value in enumerate(feature.values):
                    if best_feature.values[i] <= np.mean(best_feature.values):
                        values.append(value)
                        if first_feature:
                            split_klasses.append(klasses[i])

                first_feature = False
                split_feature = Feature(feature.name, values)
                split_features.append(split_feature)
            node.left_function = lambda a: a <=np.mean(best_feature.values)
            node.left = self.build_recursive(split_features, split_klasses)

            #right node @TODO DRY
            split_features2 = []
            split_klasses2 = []
            first_feature2 = True
            for feature in features:
                values = []
                for i, value in enumerate(feature.values):
                    if best_feature.values[i] > np.mean(best_feature.values):
                        values.append(value)
                        if first_feature2:
                            split_klasses2.append(klasses[i])

                first_feature2 = False
                split_feature = Feature(feature.name, values)
                split_features2.append(split_feature)

            node.right_function = lambda a: a > np.mean(best_feature.values)
            node.right = self.build_recursive(split_features2, split_klasses2)
            return node


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
    feature_names, data = load_csv(INPUT_FILE, '\t')
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

def load_csv(filename, separator= ','):
    data = []

    with open(filename) as file:
        feature_names = file.readline().split(separator)  # skip first line
        while True:
            line = file.readline()
            if not line: break

            fields = line.split(separator)
            # read all attributes as floats, except for the class
            instance = [float(val) for val in fields[0:-1]] + [int(fields[-1])]
            data.append(instance)

    return feature_names, data


if __name__ == '__main__':
    main()
