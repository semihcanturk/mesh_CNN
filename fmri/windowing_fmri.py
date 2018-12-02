import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from fmri import process_fmri as process


def encode(C, X, H, Gp, Gn):
    """
    encodes
    :param C: data labels
    :param X: data to be windowed
    :param H: window size
    :param Gp: start point guard
    :param Gn: end point guard
    :return:
    """
    _, m, _ = C.shape
    Np, p, T = X.shape
    N = T - H + 1
    num_examples = Np * N
    
    y = np.zeros([Np, N])
    C_temp = np.zeros(T)
    
    for i in range(Np):
        for j in range(m):
            temp_idx = [idx for idx, e in enumerate(C[i, j, :]) if e == 1]
            cue_idx1 = [idx - Gn for idx in temp_idx]
            cue_idx2 = [idx + Gp for idx in temp_idx]
            cue_idx = list(zip(cue_idx1, cue_idx2))
            
            for idx in cue_idx:
                C_temp[slice(*idx)] = j + 1
                
        y[i, :] = C_temp[0 : N]
    
    X_windowed = np.zeros([Np, N, p, H])
    
    for t in range(N):
        X_windowed[:, t, :, :] = X[:, :, t : t + H]
        
    y = np.reshape(y, (num_examples))
    X_windowed = np.reshape(X_windowed, (num_examples, p, H))
        
    return [X_windowed, y]


def craft(X):
    """
    feature crafting for signal
    :param X: fmri signal data
    :return: crafted features
    """
    num_examples, p, T = X.shape
    X_f = np.empty(shape=(num_examples, p, 0))

    # energy
    X_e = np.sum(np.power(X, 2), axis=2)
    X_e = np.reshape(X_e, (num_examples, p, 1))

    # min
    X_min = np.min(X, axis=2)
    X_min = np.reshape(X_min, (num_examples, p, 1))

    # max
    X_max = np.max(X, axis=2)
    X_max = np.reshape(X_max, (num_examples, p, 1))

    # avg
    X_avg = np.mean(X, axis=2)
    X_avg = np.reshape(X_avg, (num_examples, p, 1))

    # standard deviation
    X_std = np.std(X, axis=2)
    X_std = np.reshape(X_std, (num_examples, p, 1))

    # fft
    X_ft = np.fft.fft(X, axis=2)
    X_ft = np.abs(X_ft)

    # differences
    X_fd = np.diff(X, axis=2)
    X_sd = np.diff(X_fd, axis=2) 

    # zero crossings
    X_sgn = np.sign(X) > 0
    X_sgn_d = np.abs(np.diff(X_sgn, axis=2))
    X_num_zc = np.sum(X_sgn_d, axis=2)
    X_num_zc = np.reshape(X_num_zc, (num_examples, p, 1))

    features = [X, X_e, X_min, X_max, X_avg, X_std, X_ft, X_fd, X_sd, X_num_zc]
    for ft in features:
        X_f = np.append(X_f, ft, axis=2)
    
    _, _, d = X_f.shape
    
    X_f = np.reshape(X_f, (num_examples, p * d))
    
    return X_f


def apply_pca(X, n):
    scaler = preprocessing.StandardScaler().fit(X)
    X_std = scaler.transform(X) 
    
    pca = PCA(n_components=n)
    pca.fit(X_std)
    score = pca.explained_variance_ratio_
    score = sum(score)

    X_pca = pca.transform(X_std)
    
    return [X_pca, score]


def try_model(X, y, model):
    """
    run a ML model with the data and labels
    :param X: data
    :param y: labels
    :param model: desired ML model
    :return: resulting confusion matrix
    """
    seed = 21
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    
    stimuli = ['none', 'LF', 'LH', 'RF', 'RH', 'T']

    C_mat = confusion_matrix(y_test, y_hat) 
    C = pd.DataFrame(C_mat, index=stimuli, columns=stimuli)
    
    print(C)
    print()
    
    return C_mat


def main():
    """
    example data processing and training pipeline
    :return:
    """
    H, Gp, Gn = 15, 4, 4
    num_components = 300
    seed = 21

    C, _, X_bar = process.get_dataset('./all_subjects/', p=148, session='MOTOR_LR')
    
    X, y = encode(C, X_bar, H, Gp, Gn)
    X_f = craft(X)

    X_pca, pct_explained = apply_pca(X_f, num_components)

    model_lr = LogisticRegression(random_state=seed, class_weight='balanced')
    
    try_model(X_pca, y, model_lr)

if __name__ == '__main__':
    main()
