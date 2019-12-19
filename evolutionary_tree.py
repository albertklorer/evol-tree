import numpy as np
import pandas as pd
import math

class Node:
    """
        Represents a node in decision tree. 
        feature: feature node was split on when fitting training data 
        output: class with majority in tree
        children: children of node reperesented as dicitonary. key is value 
            of feature where the node was split, corresponding value stores child Node
        index: used to assign a unique index to each node
    """
    def __init__(self, feature, output):
        self.feature = feature
        self.output = output
        self.children = {}
        self.index = -1

    def add_child(self, feature_value, child):
        self.children[feature_value] = child

class DecisionTree:
    """
        Represents decision tree
        root: root node of decision tree
    """
    def __init__(self):
        self.root = None

    # helper function to count unique values of each class
    def count_unique_values(self, Y):
        counts = {}
        for val in Y:
            if val not in counts:
                counts[val] = 1
            else: 
                counts[val] = counts[val] + 1

        return counts

    # helper function to calculate entropy of a set
    def entropy(self, Y):
        counts = self.count_unique_values(Y)
        entropy = 0
        length = len(Y)
        for val in counts:
            p = counts[val] / length
            entropy += (-p)*math.log2(p)
        return entropy

    # helper function to find information gain of feature
    def gain_ratio(self, x, Y, feature):
        original_entropy = self.entropy(Y)
        info_f = 0  
        split_info = 0
        values = set(x[:,feature])
        df = pd.DataFrame(x)
        # add Y values to last column of dataframe
        df[df.shape[1]] = Y
        initial_size = df.shape[0] 
        for i in values:
            df1 = df[df[feature] == i]
            current_size = df1.shape[0]
            info_f += (current_size/initial_size)*self.entropy(df1[df1.shape[1]-1])
            split_info += (-current_size/initial_size)*math.log2(current_size/initial_size)

        # to handle the case when split info = 0 which leads to division by 0 error
        if split_info == 0 :
            return math.inf

        info_gain = original_entropy - info_f
        gain_ratio = info_gain / split_info
        return gain_ratio

    # returns root of decision tree built after fitting training data
    def decision_tree(self,X,Y,features,level,classes):
        # If the node consists of only 1 class
        if len(set(Y)) == 1:
            output = None
            for i in classes:
                if i in Y:
                    output = i
                    
            return Node(None,output)

        # If we have run out of features to split upon
        # In this case we will output the class with maximum count
        if len(features) == 0:
            freq_map = self.count_unique_values(Y)
            output = None
            max_count = -math.inf
            for i in classes:
                if i in freq_map :
                    if freq_map[i] > max_count :
                        output = i
                        max_count = freq_map[i]

            return Node(None,output)
        
        # Finding the best feature to split upon
        max_gain = -math.inf
        final_feature = None
        for f in features :

            current_gain = self.gain_ratio(X,Y,f)

            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = f

        freq_map = self.count_unique_values(Y)
        output = None
        max_count = -math.inf

        for i in classes:
            if i in freq_map:
                if freq_map[i] > max_count :
                    output = i
                    max_count = freq_map[i]
            
        unique_values = set(X[:,final_feature]) # unique_values represents the unique values of the feature selected
        df = pd.DataFrame(X)
        # Adding Y values as the last column in the dataframe
        df[df.shape[1]] = Y

        current_node = Node(final_feature,output)

        # Now removing the selected feature from the list as we do not want to split on one feature more than once(in a given root to leaf node path)
        index  = features.index(final_feature)
        features.remove(final_feature)
        for i in unique_values:
            # Creating a new dataframe with value of selected feature = i
            df1 = df[df[final_feature] == i]
            # Segregating the X and Y values and recursively calling on the splits
            node = self.decision_tree(df1.iloc[:,0:df1.shape[1]-1].values,df1.iloc[:,df1.shape[1]-1].values,features,level+1,classes)
            current_node.add_child(i,node)

        # Add the removed feature     
        features.insert(index,final_feature)

        return current_node

    # Fits tree to the given training data
    def fit(self,X,Y):
        features = [i for i in range(len(X[0]))]
        classes = set(Y)
        level = 0
        self.root = self.decision_tree(X,Y,features,level,classes)

    # returns prediciton based on predictor values
    def predict(self,X):
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.predict_for(X[i],self.root)
        return Y

    # predicts the class for a given testing point and returns the answer
    def predict_for(self,data,node):
        if len(node.children) == 0 :
            return node.output
        # represents the value of feature on which the split was made  
        val = data[node.feature]    
        if val not in node.children :
            return node.output
        
        # Recursively call on the splits
        return self.predict_for(data,node.children[val])

    # returns the mean accuracy of test set
    def score(self,X,Y):
        Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                count = count + 1
        return count/len(Y_pred)

class EvolutionaryForest():
    """
    Represents evolutionary forest
    """
    def __init__(self, population_size=4, mutation_rate=0.5, iterations=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.iterations = iterations
        self.population = []
        self.population_fitness = []
    
    # helper function to shuffle random subsample of X and Y at same indices
    def shuffle(self, X, Y):
        randomize = np.arange(len(Y))
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        subsample_length = np.random.randint(int(len(Y) / 2), len(Y))
        X = X[:subsample_length, :]
        Y = Y[:subsample_length]
        return X, Y

    # helper function to find fitness values of population
    def evaluate_fitness(self, X_test, Y_test):
        if len(self.population_fitness) < len(self.population):
            for i in range(len(self.population) - len(self.population_fitness)):
                self.population_fitness.append(self.population[i].score(X_test, Y_test))
        else:
            for i in range(len(self.population)):
                self.population_fitness[i] = self.population[i].score(X_test, Y_test)
    
    # helper function to return best parents of population
    def mating_pool(self, population_fitness):
        parents = []
        for i in range(int(self.population_size / 3)):
            max_fitness = 0
            max_index = -1
            for j in range(len(population_fitness)):
                if population_fitness[j] > max_fitness:
                    max_fitness = population_fitness[j]
                    max_index = j
            del population_fitness[max_index]
            parents.append(self.population[max_index])
        
        return parents

    # helper function to create children of parent population 
    def create_children(self, parents, children_size):
        children = []
        for i in range(children_size):
            parent_1 = parents[np.random.randint(len(parents))]
            parent_2 = parents[np.random.randint(len(parents))]

            # reroll parent_2 until it does not match parent_1
            while parent_1 is parent_2:
                parent_2 = parents[np.random.randint(len(parents))]

            # iterate through children nodes of first parent 
            for j in range(len(parent_1.root.children)):
                # roll based on mutation_rate value
                if np.random.uniform() < self.mutation_rate:
                    parent_1.root.children[j] = np.random.choice(parent_2.root.children)

            children.append(parent_1)

        self.population = children

    # generate a population based on training and testing data 
    def fit(self, X_train, Y_train, X_test, Y_test):
        # generate from scratch if population is 0 
        if len(self.population) is 0:
            for i in range(self.population_size):
                shuffle_X, shuffle_Y = self.shuffle(X_train, Y_train)
                decision_tree = DecisionTree()
                decision_tree.fit(shuffle_X, shuffle_Y)
                self.population.append(decision_tree)
            self.evaluate_fitness(X_test, Y_test)
        # repeat for specified number of iterations
        for i in range(self.iterations):
            mating_pool = self.mating_pool(self.population_fitness)
            self.create_children(mating_pool, len(self.population))
            self.evaluate_fitness(X_test, Y_test)
            print(i)
            print(self.population_fitness)
        

    

    
tree = DecisionTree()

iris = np.genfromtxt('iris.csv', delimiter=',')
np.random.shuffle(iris)
iris_train = iris[0:100]
iris_test = iris[101:]

x_train = iris_train[:, 0:4]
y_train = iris_train[:, 4]
x_test = iris_test[:, 0:4]
y_test = iris_test[:, 4]
tree.fit(x_train,y_train)
Y_pred = tree.predict(x_train)
evolutionary_tree = EvolutionaryForest()
evolutionary_tree.fit(x_train, y_train, x_test, y_test)
print(evolutionary_tree.population_fitness)
population = evolutionary_tree.create_children(evolutionary_tree.mating_pool(evolutionary_tree.population_fitness), evolutionary_tree.population_size)
for member in population: 
    print(member.score(x_test, y_test))

print(len(evolutionary_tree.population))

# for child in tree.root.children:
#     print(tree.root.children[child].feature)
# print("Predictions :",Y_pred)
# print()
# print("Score :",tree.score(x_test,y_test)) # Score on training data