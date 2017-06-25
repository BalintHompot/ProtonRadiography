import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

#np.seterr(over='ignore')
dir = r"C:\Users\Bálint\Desktop\Bálint\program\Python\Proton\data" ##type the folder with the data here
listdir = os.listdir(dir)
print(listdir)
for file in listdir:

    # ////////////////////////opening h5 file and converting to numpy array
    l1means = []
    l2means = []
    err = []
    X = []
    Y = []
    for run in range(1,2):

        hdf5_file_name = file[:-8]
        hdf5_file_name_back = hdf5_file_name
        hdf5_file_name = hdf5_file_name + "H02_" + str(run) + ".h5"
        hdf5_file_name_back = hdf5_file_name_back + "H08_" + str(run) + ".h5"
        print(dir + "\\" + hdf5_file_name)
        print(dir + "\\" + hdf5_file_name_back)
        #////the whole thing above is just to find the matching files
        front    = h5py.File(dir + "\\" + hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
        back    = h5py.File(dir + "\\" + hdf5_file_name_back, 'r')   # 'r' means that hdf5 file is open in read-only mode

        frontdataTemp = []
        backdataTemp = []

        for frame in front:
            frontdataTemp.append(np.array(front[frame]['Data']))
            backdataTemp.append(np.array(back[frame]['Data']))
        #X.append(frontdataTemp)
        #Y.append(backdataTemp)
    X = (np.array(frontdataTemp)/10000)
    Y = (np.array(backdataTemp)/10000)

    print(X)

    #///////////////////////////////////////////////////////////////////


    print("training network")

    np.random.seed(1)

    # randomly initialize our weights with mean 0
    syn0 = 2 * np.random.random((65536, 100)) - 1
    syn1 = 2 * np.random.random((100, 65536)) - 1

    #print(syn0)
    #print(syn1)
    for j in range(20):

        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1value = np.dot(l0, syn0)
        l1 = nonlin(l1value)
        l2value = np.dot(l1, syn1)
        l2 = nonlin(l2value)

        #print(l0, l1, l2)
        # how much did we miss the target value?
        l2_error = Y - l2value

        if (j % 1) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))


        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error * nonlin(l2value, deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1value, deriv=True)

        syn1 += l1.T.dot(l2_error)
        syn0 += l0.T.dot(l1_error)
        print("iteration completed")


        err.append(np.mean(np.abs(Y-l2value)))
        print(str(err))
        #plt.plot(l2[1])
        #plt.show()




    print("first layer error is :")
    print(l1_error)
    print("second layer error is :")
    print(l2_error)
    print("final predicted output: ")
    print(l2)



    front.close()
    back.close()
    listdir.remove(hdf5_file_name)
    listdir.remove(hdf5_file_name_back)
    plt.plot(err)
    plt.show()
