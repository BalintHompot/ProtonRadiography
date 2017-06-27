import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib as ml

def meanCustom(x):
    sum = 0.
    for pos in range(65536):
        sum += x[pos]
    sum /= 65536
    return sum
def visualizeRelevantCells(w):
    plotcellsx = [256]
    plotcellsy = [256]
    for cell in range(361):
        plotcellsx.append(w[cell][0]/256)
        plotcellsy.append(w[cell][0]%256)
    plt.scatter(plotcellsx, plotcellsy)
    plt.axis([0, 256, 0, 256])
    plt.show()

def visualizeRelevantCells2(w):
    plotcells = [[0 for x in range(256)] for y in range(256)]
    for cell in range(361):
        #print(str(w[cell][0]//256) + "   " + str(w[cell][0]%256))
        if(w[cell][0]/256>=0 and w[cell][0]%256>=0):
            plotcells[int(w[cell][0]/256)][int(w[cell][0]%256)] = w[cell][1]
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(plotcells)
    ax.set_aspect('equal')

    cax = fig.add_axes([0, 256, 0, 256])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(ax = ax,cax = cax,orientation='vertical')
    plt.show()
def initializeWeights(syn0):
    counter = 0
    while (counter < 65536):
        pos = []
        for hor in range (-9, 10):
            for ver in range (-9, 10):
                position = ((counter/256) - ver) * 256 + (counter%256 + hor)
                pair = [position, np.random.random()]
                pos.append(pair)
        #plt.plot(pos)
        #plt.show()
        #if(counter%1==0):
            #visualizeRelevantCells(pos)
        syn0.append(pos)
        counter+=1
    np.save('weightsArr', syn0)
    return syn0

def limitedDotProduct(input, w):
    output = []

    for position in range(65536):
        product = 0.
        for pair in range(361):
            inputNode = int(w[position][pair][0])
            if inputNode > 0 and inputNode < 65536:
                #print(str(w[position][pair][1]), str(inputNode))
                product += w[position][pair][1] * input[inputNode]
        output.append(product)
    return output

def updateWeights(error, w):
    for position in range(65536):
        for pair in range(361):
            w[position][pair][1] += w[position][pair][1]*error[position]
            #if(w[position][pair][1]>1):
                #w[position][pair][1] =1
            #if (w[position][pair][1] < -1):
               # w[position][pair][1] = -1

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

#np.seterr(over='ignore')
dir = r"C:\Users\Bálint\Desktop\Bálint\program\Python\Proton\data" ##type the folder with the data here
listdir = os.listdir(dir)
print(listdir)
for file in listdir:

    # ////////////////////////opening h5 file and converting to numpy arra
    medata = []
    X = []
    y = []
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
        for dataset in range(400):
            for number in range(65536):
                if frontdataTemp[dataset][number]>2000:
                    frontdataTemp[dataset][number] = 0
                if backdataTemp[dataset][number]>2000:
                    backdataTemp[dataset][number] = 0
    X = (np.array(frontdataTemp)/10000)
    y = (np.array(backdataTemp)/10000)

    print(X)
    print(y)

    #///////////////////////////////////////////////////////////////////
    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)

    # initialize weights randomly with mean 0
    #syn0 = 2*np.random.random((65536,221)) - 1
    syn0 = []
    #syn0 = np.array(initializeWeights(syn0))
    syn0 = np.load("weightsArr.npy")
    #for pos in range(0,65536,2000):
        #visualizeRelevantCells2(syn0[pos])
    print(syn0)
    print(np.shape(syn0))
    for dataset in range(400):
        for iter in range(2):

        # forward propagation
            l0 = X[dataset]
            l1 = limitedDotProduct(l0, syn0)

            # how much did we miss?
            l1_error = y[dataset] - l1

            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            #l1_delta = limitedDotProduct(l1_error, syn0)

            # update weights
            #syn0 += np.dot(l0.T,l1_delta)

            #print(syn0)
            updateWeights(l1_error, syn0)
            #ijovisualizeRelevantCells2(syn0[40000])
            print(l1_error)
            print(np.max(y[dataset]))
            print(np.max(l1))
            print(np.max(l1_error))
            me = np.mean(abs(l1_error), dtype=np.float64)
            print("mean error is : " + str(me))
            medata.append(me)
            #print(meanCustom(l1_error))
            #print(np.shape(l1))
            np.save('weightsArr', syn0)
            #np.savetxt('weightsTXT', syn0,'%f %f %.5f')

    medatasave=np.array(medata)
    np.save('meanErrors', medatasave)
