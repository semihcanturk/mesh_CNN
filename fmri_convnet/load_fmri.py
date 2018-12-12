import os
import numpy as np
import nibabel as nib
from numpy import genfromtxt
from mesh import mesh_traversal
from fmri import windowing_fmri
from fmri import process_fmri as dp
import matplotlib.pyplot as plt


def load_all_examples():
    """
    Loads all fMRI patient files from directory. The files can be found in the open source Human Connectome Project,
    in the "patientid_dense_tasks.mat" format. a "filenames.txt" that includes the files to be parsed in each line
    should also be created. If loading a single file is desired, see load_single_example()
    :return: parsed fmri files in train/test data and labels
    """
    H, Gp, Gn = 15, 4, 4

    C, _, X_bar = dp.get_dataset('../fmri/all_subjects/', session='MOTOR_LR')
    X, y = windowing_fmri.encode(C, X_bar, H, Gp, Gn)

    n_data = len(y)
    train_test_split = 0.8

    y = y.astype(int)

    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    y = y[s]

    n_train = int(n_data * train_test_split)

    train_data = X[:n_train]
    test_data = X[n_train:]
    train_labels = y[:n_train]
    test_labels = y[n_train:]

    return train_data, train_labels, test_data, test_labels


def load_single_example():
    """
    loads and parses data from a single example. The specified data is available from the open source
    Human Connectome Project.
    :return: parsed fmri file in train/test data and labels
    """
    H, Gp, Gn = 15, 4, 4
    C, _, X_bar = dp.get_dataset_single('./load_data/601127_aparc_tasks_aparc.mat', p=148, session='MOTOR_LR')

    img = nib.load('./load_data/tfMRI_MOTOR_LR_Atlas.dtseries.nii')
    a = np.array(img.dataobj)
    a = a[:, :32492]
    a_T = a.transpose()
    a_swap = np.swapaxes(a, 0, 1)

    # for plotting of the time series if desired

    # data_92 = a_swap[92]
    # data_93 = a_swap[93]
    # data_94 = a_swap[94]
    #
    # plt.figure(figsize=(20, 5))
    #
    # plt.subplot(3, 1, 1)
    # plt.plot(data_92, linewidth=5)
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(data_93, linewidth=5)
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(data_94, linewidth=5)
    #
    # plt.show()
    # plt.savefig('data.png')


    C = C.reshape(1, C.shape[0], C.shape[1])
    a_swap = a_swap.reshape(1, a_swap.shape[0], a_swap.shape[1])

    X, y = windowing_fmri.encode(C, a_swap, H, Gp, Gn)

    n_data = len(y)
    train_test_split = 0.8

    y = y.astype(int)
    #data = np.concatenate((X, y), axis=1)

    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    y = y[s]

    n_train = int(n_data * train_test_split)

    train_data = X[:n_train]
    test_data = X[n_train:]
    train_labels = y[:n_train]
    test_labels = y[n_train:]

    return train_data, train_labels, test_data, test_labels


def load():
    """
    Loads 3D surface data and creates its adjacency matrix.
    :return: training & test data, adjacency matrix of the mesh, the list of mesh coordinates and faces
    """
    try:
        coords_gii = genfromtxt('coords.csv', delimiter=',')
        faces_gii = genfromtxt('faces.csv', delimiter=',')
    except:
        # try:
        #     file_gii = './load_data/601127.L.inflated.32k_fs_LR.surf.gii'
        #     file_gii = os.path.join(file_gii)
        # except:
        file_gii = '/Users/semo/PycharmProjects/Conv_CNN/fmri_convnet/load_data/601127.L.inflated.32k_fs_LR.surf.gii'
        img = nib.load(file_gii)

        # these are the spatial coordinates
        coords_gii = img.darrays[0].data

        # these are the mesh connections
        faces_gii = img.darrays[1].data

        np.savetxt("coords.csv", coords_gii, delimiter=",")
        np.savetxt("faces.csv", faces_gii, delimiter=",")

    faces_gii = faces_gii.astype(int)
    adj_mtx, _, _ = mesh_traversal.create_adj_mtx(coords_gii, faces_gii, is_sparse=True)

    ###############

    train_data, train_labels, test_data, test_labels = load_all_examples()

    return train_data, train_labels, test_data, test_labels, adj_mtx, coords_gii, faces_gii
