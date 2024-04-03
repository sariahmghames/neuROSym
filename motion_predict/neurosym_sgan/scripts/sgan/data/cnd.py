import os
import math
import itertools
import numpy as np
import glob
import json
import copy


qtc_connections = {}
qtc_proba = {}

def get_qtcstate_proba():
    for key, value in qtc_connections.items():
        qtc_proba[key] = 1/len(value)


if __name__ == '__main__':

    curr_dir = os.getcwd()
    txt_file =  curr_dir + "/qtcc1_labels.txt"
    qtc = []

    with open(txt_file) as f:
        labels = [line.strip() for line in f.readlines()] # removes newline \n character at end of each effective line 
        for x in labels:
            arr2 = []
            arr3 = []
            qtc_str = x.split(' ')[1]
            arr1 = qtc_str.split(',')
            [arr2.append(i1.replace("[", '')) for i1 in arr1]
            [arr3.append(i2.replace("]", '')) for i2 in arr2]
            qtc.append([eval(i) for i in arr3])
        #print("qtcAB=",qtc)
        for l in range(len(qtc)):
            qtc_connections[str(l)] = []
        for ind, item in enumerate(qtc):
            qtc_copy = copy.deepcopy(qtc)
            qtc_copy.remove(item)

            for el in qtc_copy:
                if np.all(np.absolute(np.subtract(np.asarray(item),np.asarray(el))) != 2):
                    qtc_connections[str(ind)].append(qtc.index(el)) 

        get_qtcstate_proba()
        print("qtc_proba=", qtc_proba)


        with open("cnd_labels.txt", "w") as f:
            for ind, x in enumerate(labels):
                qtc_str = x.split(' ')[1]
                f.write(str(float("{:.4f}".format(qtc_proba[str(ind)]))))
                f.write(' ')
                f.write(qtc_str)
                f.write('\n')














