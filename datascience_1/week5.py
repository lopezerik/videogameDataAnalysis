from week4 import *
import pandas as pd

def compute_training(slices, left_out):
    training_slices = []
    for i in range(len(slices)):
        if i == left_out:
            continue
        training_slices.append(slices[i])
    return pd.concat(training_slices)

def caser(table, tree, target):
    scratch_table = pd.DataFrame(columns=['prediction', 'actual'])
    scratch_table['prediction'] = table.apply(lambda row: tree_predictor(row, tree), axis=1)
    scratch_table['actual'] = table[target]  # just copy the target column
    cases = scratch_table.apply(lambda row: predictor_case(row, pred='prediction', target='actual'), axis=1)
    return cases.value_counts()

def k_fold(table, k, target, hypers, candidate_columns):
    result_columns = ['name', 'true_positive', 'false_positive', 'true_negative', 'false_negative', 'accuracy', 'f1', 'informedness']
    k_fold_results_table = pd.DataFrame(columns=result_columns)
    
    total_len = len(table.index)
    split_size = int(total_len/(1.0*k))
    slices = []

    #generate the slices
    for i in range(k-1):
        a_slice =  table[i*split_size:(i+1)*split_size]
        slices.append( a_slice )
    slices.append( table[(k-1)*split_size:] )
    
    #generate test results
    for i in range(k):
        test_table = slices[i]
        train_table = compute_training(slices, i)
        fold_tree = build_tree_iter(train_table, candidate_columns, target, hypers)  # train
        fold_cases = caser(test_table, fold_tree, target)  # test

        k_fold_results_table = k_fold_results_table.append(fold_cases,ignore_index=True)
        end = k_fold_results_table.last_valid_index()
        #while seems straightforward, believe below cause setting value of slice warnings
        #k_fold_results_table.accuracy.iloc[end] =  accuracy(fold_cases)
        #k_fold_results_table.name.iloc[end] =  'fold '+str(i+1)+' test'
        #k_fold_results_table.f1.iloc[end] =  f1(fold_cases)
        #k_fold_results_table.informedness.iloc[end] =  informedness(fold_cases)
        #below avoids warning messages
        k_fold_results_table.loc[end, 'accuracy'] =  accuracy(fold_cases)
        k_fold_results_table.loc[end, 'name'] =  'fold '+str(i+1)+' test'
        k_fold_results_table.loc[end, 'f1'] =  f1(fold_cases)
        k_fold_results_table.loc[end, 'informedness'] =  informedness(fold_cases)
        
    k_fold_results_table.__doc__ = str(hypers)  # adds comment to remind me of hyper params used
    return k_fold_results_table