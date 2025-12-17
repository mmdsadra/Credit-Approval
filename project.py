import pandas as pd
import numpy as np

input_limit = 100
missing_data = '?'
have_header = False
character_missing_data = "missing"  # For now I have used most frequent character for missing values
result_column = 1
epoches = 10000
learning_rate = 0.001

def get_col(df, arg, limit):
    return df[arg][:limit].to_numpy()

def init():
    try:
        if (have_header):
            df = pd.read_csv("crx.xls")
        else:
            df = pd.read_csv("crx.xls", header = None)
    except Exception as e:
        print("file not found\n")
        print(e+"\n")
    return df

def get_features(df, n, limit):
    datas = []
    datas = df.iloc[:limit, :n]
    return np.array(datas)

def get_labels(df, n,limit):
    datas = []
    for i in range(result_column):
        datas.append(get_col(df, n-i, limit))
    return np.array(datas)

def solve_missing_data(features):
    n = features.shape[1]
    for i in range(n):
        zero_count = (features[:,i] == missing_data).sum()
        if (zero_count > 0):
            index_list = np.where((features[:,i] == missing_data))[0]
            # TODO: I should change 0 index, It might be zero itself
            if (isinstance(features[0, i], (int, float))):
                avg = np.mean(features[:, i])
                for j in index_list:
                    features[j, i] = avg
            else:
                values, counts = np.unique(features[:, i], return_counts=True)
                most_frequent = values[np.argmax(counts)]
                for j in index_list:
                    features[j, i] = most_frequent   
    return features 

def features_convert_to_number(features):
    m, n = features.shape
    for i in range (n):
        data = features[:, i]
        if (isinstance(data[0], (int, float))):
            continue
        unique_data = np.unique(data)
        len_unique_data = len(unique_data)
        if (len_unique_data == 2):
            for j in range (m):
                if (data[j] == unique_data[0]):
                    data[j] = 0
                else:
                    data[j] = 1
        elif (len_unique_data > 2):
            num_arr = np.arange(1, len_unique_data+1, dtype=float)
            min = np.min(num_arr)
            max = np.max(num_arr)
            distance = max - min
            for j in range(len_unique_data):
                num_arr[j] -= min
                num_arr[j] /= distance
            for j in range (m):
                idx = np.where(unique_data == data[j])[0][0]
                data[j] = num_arr[idx]

        features[:, i] = data
    features = features.astype(float)
    return features

def labels_convert_to_number(lables):
    n = lables.shape[1]
    for i in range (n):
        if (lables[0][i] == '+'):
            lables[0][i] = 1
        else:
            lables[0][i] = 0
    lables = lables.astype(float).flatten()
    return lables

def init_weights_bias(ntehta):
    weights = np.zeros(ntehta)
    bias = 0
    return weights, bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logreg_compute_gradiant(features, lables, thetas, bias):
    m, n = features.shape
    z = np.dot(features, thetas) + bias 
    predictions = sigmoid(z)
    dw = (1/m) * np.dot(features.T, (predictions - lables))
    db = (1/m) * np.sum(predictions - lables)
    thetas = thetas - learning_rate * dw
    bias = bias - learning_rate * db
    return thetas, bias

def logreg_compute_cost(features, lables, thetas, bias):
    m = len(lables)
    z = np.dot(features, thetas) + bias
    prediction = sigmoid(z)
    epsilon = 1e-15
    prediction = np.clip(prediction, epsilon, 1 - epsilon)
    cost = -np.mean(lables * np.log(prediction) + (1-lables) * np.log(1-prediction))
    return cost

def logistic_regression(features, lables, nrows, thetas, bias):
    ncols = len(features)
    cost_history = []
    for epoch in range (epoches):
        thetas, bias = logreg_compute_gradiant(features, lables, thetas, bias)
        cost = logreg_compute_cost(features, lables, thetas, bias)
        cost_history.append(cost)
    return thetas, bias, cost_history
        
def get_test_inputs(df, n, limit):
    datas = []
    datas = df.iloc[limit:, :n]
    datas = np.array(datas)
    return datas

def get_test_results(df, n, limit):
    datas = []
    datas = df.iloc[limit:, n]
    datas = np.array(datas)
    return datas

def tresults_convert_to_number(test_results):
    n = len(test_results)
    for j in range (n):
        if (test_results[j] == '+'):
            test_results[j] = 1
        else:
            test_results[j] = 0
    test_results = np.array(test_results)
    return test_results

def predict(x, y, thetas, bias):
    n, m = x.shape
    ylen = len(y)
    z = np.dot(x, thetas) + bias
    predictions = sigmoid(z)
    plen = len(predictions)
    for i in range(plen):
        if (predictions[i] >= 0.5):
            predictions[i] = 1
        else:
            predictions[i] = 0
    success = 0
    for i in range (plen):
        if (predictions[i] == y[i]):
            success += 1
    return (success / ylen)

def main():
    df = init()
    ninput_data = df.shape[0] - input_limit
    if (ninput_data < 0):
        ninput_data = df.shape[0]
        ninput_test = 0
    else:
        ninput_test = input_limit
    nfeaurescols = df.shape[1] - result_column
    features = get_features(df, nfeaurescols, ninput_data)
    labels = get_labels(df, df.shape[1]-1, ninput_data)
    features = solve_missing_data(features)
    features = features_convert_to_number(features)
    labels = labels_convert_to_number(labels)
    theta_arr, bias = init_weights_bias(nfeaurescols)
    theta_arr, bias, cost_history = logistic_regression(features, labels, ninput_data, theta_arr, bias)
    if (ninput_test):
        test_inputs = get_test_inputs(df, nfeaurescols, ninput_data)
        test_results = get_test_results(df, nfeaurescols, ninput_data)
        test_results = tresults_convert_to_number(test_results)
        test_inputs = solve_missing_data(test_inputs)
        test_inputs = features_convert_to_number(test_inputs)
        sucess_percent = predict(test_inputs, test_results, theta_arr, bias)
        print(sucess_percent)

if __name__ == "__main__":
    main()
