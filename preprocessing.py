import numpy as np
from scipy.io import arff
import pandas as pd
import sys
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.special import erf

def import_data(dataset, freq='30Min'):
    if ".npz" in dataset:
        data = np.load('/cw/dtaijupiter/NoCsBack/dtai/luca/Datasets/'+dataset)
        X = data['X']
        X, indices = np.unique(X, axis=0, return_index=True)
        y = data['y']
        y = y[indices]
        y[y==0] = -1
        name = dataset.split('.')[0]
        if np.shape(X)[0] > 1000:
            X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=331, shuffle=True, stratify=y)
    elif ".arff" in dataset:
        data = arff.loadarff('/cw/dtaijupiter/NoCsBack/dtai/luca/Datasets/'+dataset)
        df = pd.DataFrame(data[0])
        df = df.drop_duplicates()
        df['outlier'] = [string.decode("utf-8") for string in df['outlier'].values]
        y = np.asarray([1 if string == 'yes' else -1 for string in df['outlier'].values])
        X = df[df.columns[:-2]].values
        name = dataset.split('_')[0]
        if np.shape(X)[0] > 1000:
            X, _, y, _ = train_test_split(X, y, train_size=1000, random_state=331, shuffle=True, stratify=y)
    elif "T" in dataset:
        path = '/cw/dtaijupiter/NoCsBack/dtai/luca/EDP_dataset/'

        T01_2016 = pd.read_csv(path+'T01_2016dataset.csv', low_memory=False).dropna()
        T06_2016 = pd.read_csv(path+'T06_2016dataset.csv', low_memory=False).dropna()
        T07_2016 = pd.read_csv(path+'T07_2016dataset.csv', low_memory=False).dropna()
        T11_2016 = pd.read_csv(path+'T11_2016dataset.csv', low_memory=False).dropna()

        T01_2017 = pd.read_csv(path+'T01_2017dataset.csv', low_memory=False).dropna()
        T06_2017 = pd.read_csv(path+'T06_2017dataset.csv', low_memory=False).dropna()
        T07_2017 = pd.read_csv(path+'T07_2017dataset.csv', low_memory=False).dropna()
        T11_2017 = pd.read_csv(path+'T11_2017dataset.csv', low_memory=False).dropna()

        T01 = pd.concat([T01_2016,T01_2017]).reset_index(drop=True)
        T06 = pd.concat([T06_2016,T06_2017]).reset_index(drop=True)
        T07 = pd.concat([T07_2016,T07_2017]).reset_index(drop=True)
        T11 = pd.concat([T11_2016,T11_2017]).reset_index(drop=True)
        idx07 = np.where(T07["Gen_Bear2_Temp_Avg"] == 205)[0]
        T07.loc[idx07,"Label"] = 1

        for i in range(6):
            col = T06.columns.values[i]
            idx06 = np.where(T06[col] == 205)[0]
            T06.loc[idx06,"Label"] = 1

        tid = dataset
        if tid == "T01":
            anomalies = pd.DataFrame(T01[T01["Label"] > 0]).reset_index(drop=True)
            normals = pd.DataFrame(T01[T01["Label"] == 0]).reset_index(drop=True)
        elif tid == "T06":
            anomalies = pd.DataFrame(T06[T06["Label"] > 0]).reset_index(drop=True)
            normals = pd.DataFrame(T06[T06["Label"] == 0]).reset_index(drop=True)
        elif tid == "T07":
            anomalies = pd.DataFrame(T07[T07["Label"] > 0]).reset_index(drop=True)
            normals = pd.DataFrame(T07[T07["Label"] == 0]).reset_index(drop=True)
        else:
            anomalies = pd.DataFrame(T11[T11["Label"] > 0]).reset_index(drop=True)
            normals = pd.DataFrame(T11[T11["Label"] == 0]).reset_index(drop=True)


        sampled_idx = np.random.randint(0,len(normals),1000-len(anomalies))
        sampled_normals = pd.DataFrame(normals.iloc[sampled_idx]).reset_index(drop=True)
        T = pd.concat([anomalies,sampled_normals]).reset_index(drop=True)

        X = np.array(T.loc[:,T.columns.values[2:len(T.columns.values)-4]])
        y = np.array(T.loc[:,"Label"])
        y[y==0] = -1
        name = tid
    else:
        sys.path.insert(0, '/cw/dtaijupiter/NoCsBack/dtai/luca/poc_wind_turbines')
        from config import handle_imports
        CONFIG = handle_imports()
        from pipeline import load_and_label_blade_icing_data, preprocess
        S = load_and_label_blade_icing_data(int(dataset), CONFIG['blade_icing_data'])
        S = preprocess(S)
        S = S.groupby(pd.Grouper(freq=freq)).mean(numeric_only=True).dropna().reset_index(drop=True)
        S = S.drop_duplicates()
        y = np.asarray([np.trunc(float(string)) for string in S['label'].values])
        X = S[S.columns[:-1]].values[y!=0]
        name = dataset
    return X, y[y!=0], name, S.columns.values

def get_train_split_seed():
    return 13

def normalize_scores(scores, method='unify'):
    if method == "linear":
        return (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() != scores.min() else np.zeros(len(X_test))
    elif method == 'unify':
        mu = np.mean(scores)
        sigma = np.std(scores)
        pre_erf_score = (scores - mu) / (sigma * np.sqrt(2))
        erf_score = erf(pre_erf_score)
        return np.nan_to_num(erf_score.clip(0, 1).ravel(), nan=0)
    else:
        return scores
    
def re100(scores,Y):
    
    n_anomalies = len(Y[Y==1])
    
    lowest_anomaly_score = scores[Y==1].min()
    
    scores = np.sort(scores)[::-1]
    idx = np.where(scores==lowest_anomaly_score)[0][0]
    
    return idx/n_anomalies