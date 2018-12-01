import numpy as np
import scipy.io as sio


def get_cues(MOTOR):
    C = MOTOR['ev_idx'][0,0]
    return C[1:, :]


def get_bold(MOTOR):
    ts = MOTOR['ts'][0,0]
    X = np.matrix(ts)
    X = X.transpose()
    return X


def get_vitals(MOTOR):
    resp = MOTOR['resp'][0,0][0]
    heart = MOTOR['heart'][0,0][0]
    V = np.matrix([resp, heart])
    V = V.transpose() 
    return V


def clean_bold(X, v):
    A_1 = np.linalg.inv(v.transpose() * v)
    A_2 = A_1 * v.transpose() 
    A_hat = A_2 * X
    X_hat = v * A_hat
    X_bar = X - X_hat
    return X_bar


def get_dataset_single(file, session, p=148, T=284):
    filenames = file

    Np = 1
    m = 5

    mis_matched = 0

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])
    X_bar = np.zeros([Np, p, T])

    ds = sio.loadmat(file).get('ds')
    MOTOR = ds[0, 0][session]

    C_i = get_cues(MOTOR)
    X_i = get_bold(MOTOR)
    X_bar_i = X_i
    X_bar = X_bar_i.transpose()

    return [C_i, X_i, X_bar]


def get_delabeled_dataset(filedir, session, p=148, T=284):
    with open(filedir + 'filenames.txt', 'r') as f:
        filenames = [s.strip() for s in f.readlines()]

    Np = len(filenames)
    m = 5

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])

    for i, s in enumerate(filenames):
        file = filedir + s
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0, 0][session]
        X_i = get_bold(MOTOR)
        X[i, :, :] = X_i.transpose()

    return [C, None, X]


def get_dataset(filedir, session, p=32492, T=284):
    with open(filedir + 'filenames.txt', 'r') as f:
        filenames = [ s.strip() for s in f.readlines() ]

    Np = len(filenames) 
    m = 5 

    mis_matched = 0

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])
    X_bar = np.zeros([Np, p, T])

    for i, s in enumerate(filenames):
        file = filedir + s 
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0,0][session]

        C_i = get_cues(MOTOR)
        X_i = get_bold(MOTOR)

        X_i = X_i[:, :32492]

        X_bar_i = X_i

        if X_i.shape[1] == 32492:
            C[i, :, :] = C_i
            X[i, :, :] = X_i.transpose()
            X_bar[i, :, :] = X_bar_i.transpose()
        else: 
            mis_matched += 1
   
    if mis_matched > 0:
        print('num mismatched: {}'.format(mis_matched))

    return [C, X, X_bar]


def get_delabeled_dataset(filedir, session, p=148, T=284):
    with open(filedir + 'filenames.txt', 'r') as f:
        filenames = [ s.strip() for s in f.readlines() ]

    Np = len(filenames) 
    m = 5 

    C = np.zeros([Np, m, T])
    X = np.zeros([Np, p, T])

    for i, s in enumerate(filenames):
        file = filedir + s 
        ds = sio.loadmat(file).get('ds')
        MOTOR = ds[0,0][session]
        X_i = get_bold(MOTOR)
        X[i, :, :] = X_i.transpose()
    
    return [C, None, X]


def get_delabeled_vitals(MOTOR):
    resp = MOTOR['resp'][0,0]
    heart = MOTOR['heart'][0,0]
    print('resp shape:' + str(resp.shape))
    V = np.matrix([resp[0,:], heart[0,:]])
    V = V.transpose() 
    return V


def main():
    C, X, X_bar = get_dataset('./all_subjects/', session='MOTOR_LR')

    Np, p, T = X.shape
    _, m, _ = C.shape

    if True:
        np.savetxt('C.dat', np.reshape(C, (Np, m*T)), delimiter=',')
        np.savetxt('X.dat', np.reshape(X, (Np, p*T)), delimiter=',')
        np.savetxt('X_bar.dat', np.reshape(X_bar, (Np, p*T)), delimiter=',')

if __name__ == '__main__':
    main()
