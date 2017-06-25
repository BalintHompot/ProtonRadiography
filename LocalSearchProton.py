import h5py
import pylab
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

class correlation:
    def __init__(self, timegap, energyloss, front, back):
        self.timegap = timegap
        self.energyloss = energyloss
        self.accuracy = self.evaluation(front, back)

    def evaluation(self, front, back):
        counter = 0
        acc = 0
        for frame in front:
            frontdata = np.array(front[frame]['Data'])
            backdata = np.array(back[frame]['Data'])
            for position in range(0,65536 - self.timegap):
                if frontdata[position] != 0:
                    acc = acc + abs(frontdata[position] -self.energyloss - backdata[position + self.timegap])
                    counter = counter+1
                #print("frame is " + str(frame) + "position is " + str(position) + "acc is " + str(acc))
            #print("frame done")
        return acc/counter



dir = r"C:\Users\Bálint\Desktop\Bálint\program\Python\Proton\data"
listdir = os.listdir(dir)
print(listdir)
for file in listdir:
    for run in range(1, 4):



        hdf5_file_name = file[:-8]
        hdf5_file_name_back = hdf5_file_name
        hdf5_file_name = hdf5_file_name + "H02_" + str(run) + ".h5"
        hdf5_file_name_back = hdf5_file_name_back + "H08_" + str(run) + ".h5"
        print(dir + "\\" + hdf5_file_name)
        print(dir + "\\" + hdf5_file_name_back)
        front    = h5py.File(dir + "\\" + hdf5_file_name, 'r')   # 'r' means that hdf5 file is open in read-only mode
        back    = h5py.File(dir + "\\" + hdf5_file_name_back, 'r')   # 'r' means that hdf5 file is open in read-only mode

        timegap = 100
        energyloss = 150
        bestCorr = 99999
        lastEnergyChange = 0
        lastTimeChange = 0
        for generation in range(1, 3):
           currentbest = bestCorr
           for timechange in range(-1,2):
               for energychange in range(-1,2):
                   if not energychange == timechange == 0 and not (energychange == -lastEnergyChange and timechange == -lastTimeChange):
                       newCorr = correlation(timegap + timechange, energyloss + energychange, front, back).accuracy
                       print(str(newCorr), str(bestCorr))
                       if newCorr < bestCorr:
                           bestCorr = newCorr
                           timegap = timegap + timechange
                           energyloss = energyloss + energychange
                           lastEnergyChange = energychange
                           lastTimeChange = timechange
           print("time gap is " +str(timegap) + " energy loss is " + str(energyloss))
           if bestCorr == currentbest:
                break


        front.close()
        back.close()
        listdir.remove(hdf5_file_name)
        listdir.remove(hdf5_file_name_back)


        #for group in front:
            #print(group, "----------->", f[group])
            #for dataset in f[group]:
                #print("     ", dataset)
            #coor = np.array((f[group]['Height'], f[group]['Width']))
            #data = np.array(front[group]['Data'])
            #data = np.reshape(data,(-1, 256))
            #print(data)


            #fig = plt.figure(figsize=(6, 3.2))
            #ax = fig.add_subplot(111)
            #plt.imshow(data)


            #ax.set_title('colorMap' + group)
            #cax = fig.add_axes([0, 255, 0, 255])
            #cax.get_xaxis().set_visible(False)
            #cax.get_yaxis().set_visible(False)
           # cax.patch.set_alpha(0)
            #cax.set_frame_on(False)
            #plt.colorbar()
           # plt.show()

        ##pylab.show()
                #position.append((f[group]['Height'], f[group]['Width']))