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
        age: size of decision tree
    """
    def __init__(self):
        self.root = None
        self.age = 0

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
            print("Level",level)
            output = None
            for i in classes:
                if i in Y:
                    output = i
                    print("Count of",i,"=",len(Y))
                else :
                    print("Count of",i,"=",0)
            print("Current Entropy is =  0.0")

            print("Reached leaf Node")
            return Node(None,output)

        # If we have run out of features to split upon
        # In this case we will output the class with maximum count
        if len(features) == 0:
            print("Level",level)
            freq_map = self.count_unique_values(Y)
            output = None
            max_count = -math.inf
            for i in classes:
                if i not in freq_map:
                    print("Count of",i,"=",0)
                else :
                    if freq_map[i] > max_count :
                        output = i
                        max_count = freq_map[i]
                    print("Count of",i,"=",freq_map[i])


            print("Current Entropy  is =",self.entropy(Y))     

            print("Reached leaf Node")
            return Node(None,output)

        
        # Finding the best feature to split upon
        max_gain = -math.inf
        final_feature = None
        for f in features :

            current_gain = self.gain_ratio(X,Y,f)

            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = f

        print("Level",level)
        freq_map = self.count_unique_values(Y)
        output = None
        max_count = -math.inf

        for i in classes:
            if i not in freq_map:
                print("Count of",i,"=",0)
            else :
                if freq_map[i] > max_count :
                    output = i
                    max_count = freq_map[i]
                print("Count of",i,"=",freq_map[i])
   
        print("Current Entropy is =",self.entropy(Y))
        print("Splitting on feature  X[",final_feature,"] with gain ratio ",max_gain,sep="")
        print()


            
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

tree = DecisionTree()

iris = np.genfromtxt('iris.csv', delimiter=',')
np.random.shuffle(iris)
iris_train = iris[0:100]
iris_test = iris[101:]

x_train = iris_train[:, 0:3]
y_train = iris_train[:, 4]
x_test = iris_test[:, 0:3]
y_test = iris_test[:, 4]
tree.fit(x_train,y_train)
Y_pred = tree.predict(x_train)
print("Predictions :",Y_pred)
print()
print("Score :",tree.score(x_test,y_test)) # Score on training data