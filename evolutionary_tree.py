import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn import tree
from graphviz import Source

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

train = df[0:110]
test = df[111:150]
features = df.columns[:4]

y = pd.factorize(train['species'])[0]

random_forest = RandomForestClassifier()
random_forest.fit(train[features], y)

print()

print(tree.plot_tree(random_forest.estimators_[0]))