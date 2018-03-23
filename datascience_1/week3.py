from week2 import *

def probabilities(counts):
    count_0 = 0 if 0 not in counts else counts[0]  #could have no 0 values
    count_1 = 0 if 1 not in counts else counts[1]
    total = count_0 + count_1
    probs = (0,0) if total == 0 else (1.0*count_0/total, 1.0*count_1/total)  #build 2-tuple
    return probs

def gini(counts):
    (p0,p1) = probabilities(counts)
    sum_probs = p0**2 + p1**2
    gini = 1 - sum_probs
    return gini

def gig(starting_table, split_column, target_column):
    
    #split into two branches, i.e., two sub-tables
    true_table = starting_table.loc[starting_table[split_column] == 1]
    false_table = starting_table.loc[starting_table[split_column] == 0]
    
    #Now see how the target column is divided up in each sub-table (and the starting table)
    true_counts = true_table[target_column].value_counts()  # Note using true_table and not titanic_table
    false_counts = false_table[target_column].value_counts()  # Note using true_table and not titanic_table
    starting_counts = starting_table[target_column].value_counts() 
    
    #compute the gini impurity for the 3 tables
    starting_gini = gini(starting_counts)
    true_gini = gini(true_counts)
    false_gini = gini(false_counts)

    #compute the weights
    starting_size = len(starting_table.index)
    true_weight = 0.0 if starting_size == 0 else 1.0*len(true_table.index)/starting_size
    false_weight = 0.0 if starting_size == 0 else 1.0*len(false_table.index)/starting_size
    
    #wrap it up and put on a bow
    gig = starting_gini - (true_weight * true_gini + false_weight * false_gini)
    
    return gig


