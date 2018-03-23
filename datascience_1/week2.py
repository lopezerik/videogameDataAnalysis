def predictor_case(row, pred, target):
    actual = row[target]
    prediction = row[pred]
    if actual == 0 and prediction == 0:
        case = 'true_negative'
    elif actual == 1 and prediction == 1:
        case = 'true_positive'
    elif actual == 1 and prediction == 0:
        case = 'false_negative'
    else:
        case = 'false_positive'
    return case

def f1(cases):
    dict_cases = cases.to_dict()  # easier to work with dict than series
    #the heart of the matrix
    tp = 0 if 'true_positive' not in dict_cases else dict_cases['true_positive']  # use isin method if working with Series
    fn = 0 if 'false_negative' not in dict_cases else dict_cases['false_negative']
    tn = 0 if 'true_negative' not in dict_cases else dict_cases['true_negative']
    fp = 0 if 'false_positive' not in dict_cases else dict_cases['false_positive']
    total_pos = tp+fn
    total_pos_predict = tp+fp
    
    #other measures we can derive
    recall = 0.0 if total_pos == 0 else 1.0*tp/total_pos  # positive correct divided by total positive in the table
    precision = 0.0 if total_pos_predict == 0 else 1.0*tp/total_pos_predict # positive correct divided by all positive predictions made
    recall_div = 0.0 if recall == 0 else 1.0/recall
    precision_div = 0.0 if precision == 0 else 1.0/precision
    sum_f1 = recall_div + precision_div
    f1 = 0.0 if sum_f1 == 0 else 2.0/sum_f1
    return f1

def informedness(cases):
    dict_cases = cases.to_dict()  # easier to work with dict than series
    tp = 0 if 'true_positive' not in dict_cases else dict_cases['true_positive']
    fn = 0 if 'false_negative' not in dict_cases else dict_cases['false_negative']
    tn = 0 if 'true_negative' not in dict_cases else dict_cases['true_negative']
    fp = 0 if 'false_positive' not in dict_cases else dict_cases['false_positive']
    total_pos = tp+fn
    total_neg = tn+fp

    recall = 0.0 if total_pos == 0 else 1.0*tp/total_pos  # positive correct divided by total positive in the table
    specificty = 0.0 if total_neg == 0 else 1.0*tn/total_neg # negative correct divided by total negative in the table
    J = (recall + specificty) - 1
    return J

def accuracy(cases):
    dict_cases = cases.to_dict()
    tp = 0 if 'true_positive' not in dict_cases else dict_cases['true_positive']
    fn = 0 if 'false_negative' not in dict_cases else dict_cases['false_negative']
    tn = 0 if 'true_negative' not in dict_cases else dict_cases['true_negative']
    fp = 0 if 'false_positive' not in dict_cases else dict_cases['false_positive']

    return 1.0*(tp + tn)/(tp+tn+fp+fn)  #assumes at least one case exists