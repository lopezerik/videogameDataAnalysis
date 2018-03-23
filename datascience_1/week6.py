from week5 import *
import random

def vote_taker(row, forest):
    votes = {0:0, 1:0}
    for tree in forest:
        prediction = tree_predictor(row, tree)
        votes[prediction] += 1
    winner = 1 if votes[1]>votes[0] else 0  #ties go to 0
    return winner

def seeder(seed):
    z = [seed]  # why embed in list? See https://stackoverflow.com/a/4851555
    def f():
        val = z[0]
        z[0] += 1
        return val
    return f

def forest_builder(table, column_choices, target, hypers):
    tree_n = 5 if 'total-trees' not in hypers else hypers['total-trees']
    m = int(len(column_choices)**.5) if 'm' not in hypers else hypers['m']
    k = hypers['max-depth'] if 'max-depth' in hypers else min(2, len(column_choices))
    gig_cutoff = hypers['gig-cutoff'] if 'gig-cutoff' in hypers else 0.0
    seed = 100 if 'starting-seed' not in hypers else hypers['starting-seed']
    new_seed = seeder(seed)

    #build a single tree - call it multiple times to build multiple trees
    def iterative_build(k):
        train = table.sample(frac=1.0, replace=True, random_state=new_seed())
        train = train.reset_index()
        left_out = table.loc[~table.index.isin(train['index'])]
        left_out = left_out.reset_index() # this gives us the old index in its own column
        oob_list = left_out['index'].tolist()  # list of row indices from original titanic table
        
        rcols = random.sample(column_choices, m)  # subspcace sampling
        columns_sorted = find_best_splitter(train, rcols, target)
        (best_column, gig_value) = columns_sorted[0]

        #Note I add _1 or _0 to make it more readable for debugging
        current_paths = [{'conjunction': [(best_column+'_1', build_pred(best_column, 1))],
                          'prediction': None,
                          'gig_score': gig_value},
                         {'conjunction': [(best_column+'_0', build_pred(best_column, 0))],
                          'prediction': None,
                          'gig_score': gig_value}
                        ]
        k -= 1  # we just built a level as seed so subtract 1 from k
        tree_paths = []  # add completed paths here

        while k>0:
            new_paths = []
            for path in current_paths:
                conjunct = path['conjunction']  # a list of (name, lambda)
                before_table = generate_table(train, conjunct)  #the subtable the current conjunct leads to
                rcols = random.sample(column_choices, m)  # subspace
                columns_sorted = find_best_splitter(before_table, rcols, target)
                (best_column, gig_value) = columns_sorted[0]
                if gig_value > gig_cutoff:
                    new_path_1 = {'conjunction': conjunct + [(best_column+'_1', build_pred(best_column, 1))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_1 ) #true
                    new_path_0 = {'conjunction': conjunct + [(best_column+'_0', build_pred(best_column, 0))],
                                'prediction': None,
                                 'gig_score': gig_value
                                 }
                    new_paths.append( new_path_0 ) #false
                else:
                    #not worth splitting so complete the path with a prediction
                    path['prediction'] = compute_prediction(before_table, target)
                    tree_paths.append(path)
            #end for loop

            current_paths = new_paths
            if current_paths != []:
                k -= 1
            else:
                break  # nothing left to extend so have copied all paths to tree_paths
        #end while loop

        #Generate predictions for all paths that have None
        for path in current_paths:
            conjunct = path['conjunction']
            before_table = generate_table(train, conjunct)
            path['prediction'] = compute_prediction(before_table, target)
            tree_paths.append(path)
        return (tree_paths, oob_list)
    
    forest = []
    for i in range(tree_n):
        (paths, oob) = iterative_build(k)
        forest.append({'paths': paths, 'weight': None, 'oob': oob})
        
    return forest