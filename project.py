import pandas as pd
import numpy as np

input_limit = 100
missing_data = '?'
have_header = False
character_missing_data = "missing"
result_column = 1
epoches = 10000
learning_rate = 0.01

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
    # for i in range (n):
    #     datas.append(get_col(df, i, limit))
    datas = df.iloc[:limit, :n]
    return np.array(datas)

def get_labels(df, n,limit):
    datas = []
    for i in range(result_column):
        datas.append(get_col(df, n-i, limit))
    return np.array(datas)

def solve_missing_data(features, n):
    for i in range(n):
        zero_count = (features[i] == missing_data).sum()
        if (zero_count > 0):
            index_list = (features[i] == missing_data).tolist()
            # TODO: I should change 0 index, It might be zero itself
            if (isinstance(features[i][0], (int, float))):
                avg = np.mean(features[i])
                for j in index_list:
                    features[i][j] = avg
            else:
                for j in index_list:
                    features[i][j] = character_missing_data    

def init_weights_bias(ntehta):
    # weights = []
    # # TODO: I don't know how to initialize better
    # for i in range (ntehta):
    #     weights.append(np.random.rand())
    # bias = np.random.uniform(0, 100)
    weights = np.zeros(ntehta)
    bias = 0
    return weights, bias

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradiant(features, lables, thetas, bias):
    m, n = features.shape
    z = 0
    z += np.dot(features, thetas) + bias 
    print(z)
    return thetas, bias

def compute_cost():
    pass

def logistic_regression(features, lables, nrows, thetas, bias):
    ncols = len(features)
    for i in range (epoches):
        thetas, bias = compute_gradiant(features, lables, thetas, bias)
        cost = compute_cost()
        
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
    solve_missing_data(features, nfeaurescols)        
    theta_arr, bias = init_weights_bias(nfeaurescols)
    theta_arr, bias = logistic_regression(features, labels, ninput_data, theta_arr, bias)


if __name__ == "__main__":
    main()