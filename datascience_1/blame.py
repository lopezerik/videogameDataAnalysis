def right_and_wrong(row, tree):
    #go through each path, one by one (could use a map instead of for loop?)
    for i, path in enumerate(tree['paths']):
        conjuncts = path['conjunction']
        path_splits = map(lambda(tuple): tuple[0], conjuncts)
        result = map(lambda(tuple): tuple[1](row), conjuncts)  # potential to be parallelized
        if all(result):
            return (i, path_splits, path['prediction'] == row.Loan_Status)
    raise LookupError('No true paths found for row: ' + str(row))

#error for below: TypeError: ("<lambda>() got an unexpected keyword argument 'axis'", u'occurred at index 0')
def blame(table, tree):
    rw = table.apply(lambda row: right_and_wrong(row, tree), axis=1)
    new_dict = {}
    series = rw.value_counts()
    for key, count in series.iteritems():
        path = key[1]
        value = key[2]
        tpath = tuple(path)
        if tpath in new_dict:
            new_dict[tpath].append((value, count))
        else:
            new_dict[tpath] = [(value, count)]
    return new_dict
