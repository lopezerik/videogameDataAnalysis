#from week2 import *
from week3 import *

def build_pred(column, branch):
	return lambda row: row[column] == branch


def find_best_splitter(table, choice_list, target):
	gig_scores = map(lambda col: (col, gig(table, col, target)), choice_list)
	gig_sorted = sorted(gig_scores, key=lambda item: item[1], reverse=True)
	return gig_sorted


def generate_table(table, conjunct):
	result_table = reduce(lambda accum, pair: accum.loc[pair[1]], conjunct, table)  # accum starts as table
	return result_table


def compute_prediction(table, target):
	counts = table[target].value_counts()  # counts looks like {0: v1, 1: v2}
	if 0 not in counts and 1 not in counts:
		raise LookupError('Prediction impossible - Empty tree on leaf')
	if 0 not in counts:
		prediction = 1
	elif 1 not in counts:
		prediction = 0
	elif counts[1] > counts[0]:  # ties go to 0 (negative)
		prediction = 1
	else:
		prediction = 0

	return prediction


def build_tree_iter(table, choices, target, hypers={} ):

	k = hypers['max-depth'] if 'max-depth' in hypers else min(4, len(choices))
	gig_cutoff = hypers['gig-cutoff'] if 'gig-cutoff' in hypers else 0.0
	
	def iterative_build(k):
		columns_sorted = find_best_splitter(table, choices, target)
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
				before_table = generate_table(table, conjunct)  #the subtable the current conjunct leads to
				columns_sorted = find_best_splitter(before_table, choices, target)
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
			before_table = generate_table(table, conjunct)
			path['prediction'] = compute_prediction(before_table, target)
			tree_paths.append(path)
		return tree_paths

	return {'paths': iterative_build(k), 'weight': None}

def tree_predictor(row, tree):
    
    #go through each path, one by one (could use a map instead of for loop?)
    for path in tree['paths']:
        conjuncts = path['conjunction']
        result = map(lambda tuple: tuple[1](row), conjuncts)
        if all(result):
            return path['prediction']
    raise LookupError('No true paths found for row: ' + str(row))
