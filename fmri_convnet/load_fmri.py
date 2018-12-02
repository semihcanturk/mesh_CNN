import os
import numpy as np
import pandas as pd
import nibabel as nib
#import openmesh as om
from numpy import genfromtxt
from mesh import mesh_traversal
from fmri import multiclass
from fmri import data_preproc as dp
import matplotlib.pyplot as plt


def load_all_examples():
    H, Gp, Gn = 15, 4, 4

    C, _, X_bar = dp.get_dataset('../fmri/all_subjects/', session='MOTOR_LR')
    X, y = multiclass.encode(C, X_bar, H, Gp, Gn)

    n_data = len(y)
    train_test_split = 0.8

    #X = X[:, :, 0]

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
    H, Gp, Gn = 15, 4, 4
    C, _, X_bar = dp.get_dataset_single('./load_data/601127_aparc_tasks_aparc.mat', p=148, session='MOTOR_LR')

    img = nib.load('./load_data/tfMRI_MOTOR_LR_Atlas.dtseries.nii')
    a = np.array(img.dataobj)
    a = a[:, :32492]
    a_T = a.transpose()
    a_swap = np.swapaxes(a, 0, 1)

    data_92 = a_swap[92]
    data_93 = a_swap[93]
    data_94 = a_swap[94]

    plt.figure(figsize=(20, 5))

    plt.subplot(3, 1, 1)
    plt.plot(data_92, linewidth=5)

    plt.subplot(3, 1, 2)
    plt.plot(data_93, linewidth=5)

    plt.subplot(3, 1, 3)
    plt.plot(data_94, linewidth=5)

    plt.show()
    plt.savefig('data.png')


    C = C.reshape(1, C.shape[0], C.shape[1])
    a_swap = a_swap.reshape(1, a_swap.shape[0], a_swap.shape[1])

    X, y = multiclass.encode(C, a_swap, H, Gp, Gn)

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
    try:
        coords_gii = genfromtxt('coords.csv', delimiter=',')
        faces_gii = genfromtxt('faces.csv', delimiter=',')
    except:
        file_gii = './load_data/601127.L.inflated.32k_fs_LR.surf.gii'
        file_gii = os.path.join(file_gii)
        img = nib.load(file_gii)

        # these are the spatial coordinates
        coords_gii = img.darrays[0].data

        # these are the mesh connections
        faces_gii = img.darrays[1].data

        np.savetxt("coords.csv", coords_gii, delimiter=",")
        np.savetxt("faces.csv", faces_gii, delimiter=",")

    faces_gii = faces_gii.astype(int)
    #mesh = om.TriMesh()

    #verts = []
    #for i in coords_gii:
    #    verts.append(mesh.add_vertex(i))

    #faces = []
    #for i in faces_gii:
    #    faces.append(mesh.add_face(verts[i[0]], verts[i[1]], verts[i[2]]))

    #om.write_mesh('mesh.off', mesh)
    #om.read_trimesh('mesh.off')

    adj_mtx, _, _ = mesh_traversal.create_adj_mtx(coords_gii, faces_gii, is_sparse=True)

    ###############

    train_data, train_labels, test_data, test_labels = load_all_examples()

    return train_data, train_labels, test_data, test_labels, adj_mtx, coords_gii, faces_gii