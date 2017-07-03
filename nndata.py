import nn
import img_manip as im

import numpy as np

import json
import cPickle
import os
import random
import gzip




class NNData(object):
    def __init__(self):
        self.dm = DataManip()

    def make_input_layer_data(self, input_layer_list, rsts,
                              filename="input_layer_data.pkl"):
        ild = zip(input_layer_list, rsts)
        self.dm.pkl_w(ild, filename)

    def load_input_layer_data(self, filename):
        ild = self.dm.pkl_r(filename)
        return ild

    def one_dim_to_input_layer(self, one_dim_data):
        il = []
        for i in one_dim_data:
            il.append([i])
        return il

    def json_save(self, filename,
                  sizes, weights, biases,
                  output_layer_map):
        f = open(filename, 'wb')
        data = {"sizes":sizes,
                "weights":weights.tolist,
                "biases":biases.tolist,
                "output_rsts":output_layer_map.values}
        json.dump(data, f)
        f.close()

    def json_load(self, filename):
        f = open(filename, 'rb')
        data = json.load(f)
        f.close()

        nn = nn.NN(data["sizes"], data["output_rsts"])
        nn.weights = np.array(data["weights"])
        nn.biases = np.array(data["biases"])

        return nn


class DataManip(object):
    def img_name_list(self, path):
        inl = [os.path.join(path, img_name) for img_name in os.listdir(path) if os.path.isfile(os.path.join(path, img_name))]
        return inl

    def extend_data_with_rdm_labels(self, img_l, rdm_ratio=0.0, shuffle=False):
        #If the img has a size too big make img_name_list coorperated with rdm_labels might be
        #a good idea
        rdm_l = [1.0 for i in xrange(int(rdm_ratio*len(img_l)))]
        extended_img_l = img_l+rdm_l
        if shuffle == True:
            random.shuffle(extended_img_l)
        return extended_img_l

    def extend_data_with_rdm_imgs(self, img_l, img_shape, color_size=3, rst=None,
                                        rdm_ratio=0.0, shuffle=True):
        #This is extended with the actually data, rather than labels.
        rdm_l = [1.0 for i in xrange(int(rdm_ratio*len(img_l)))]
        extended_img_l = img_l+rdm_l
        random.shuffle(extended_img_l)
        for i in xrange(len(extended_img_l)):
            if extended_img_l[i] == 1.0:
                extended_img_l[i] = (im.gen_rdm_img(img_shape, color_size), rst)
        if shuffle == True:
            random.shuffle(extended_img_l)
        return extended_img_l


    def pkl_r(self, filename):
        f = gzip.open(filename, 'rb')
        data = cPickle.load(f)
        f.close()
        return data

    def pkl_w(self, data, filename, dest_dir='./'):
        if os.path.isdir(dest_dir) != True:
            os.mkdir(dest_dir)
            f = gzip.open(os.path.join(dest_dir,filename), 'wb')
            cPickle.dump(data, f)
            f.close()

    def zip_rst(self, imgarr_l, rstarr):
        return zip(imgarr_l, rstarr)


    #The concept of training by patterns
    def mkDifArr(self, ipt_arr, abs_dif=False):
        #Return an array of the differences between each pair of
        #ajacent elements in the given array, and the last item of this array
        #would be the difference between ipt_arr[-1]-ipt_arr[0]
        difarr = []
        ipt_arr_cpy = ipt_arr
        if abs_dif==False:
            for i in xrange(len(ipt_arr_cpy)-1):
                difarr.append(nminus(ipt_arr_cpy[i+1],
                                     ipt_arr_cpy[i]))
        else:
            for i in xrange(len(ipt_arr_cpy)-1):
                difarr.append(abs(nminus(ipt_arr_cpy[i+1],
                                  ipt_arr_cpy[i])))
        return difarr

    #The concept of training by time
    def arr_subtraction(self, arr1, arr2, abs_dif=False):
        #^---------^
        #Maybe we won't need this beacause numpy has it, so check it later

        #arr1 and arr2 should have the same length, since they are expected to be
        #from the different inputs of the same type of neural network
        if abd_dif == False:
            arr_dif = [nminus(e1,e2) for e1 in arr1 for e2 in arr2]
        else:
            arr_dif = [abs(nminus(e1,e2)) for e1 in arr1 for e2 in arr2]

        return arr_dif





def nminus(a,b):
    #Will neurons in the input layer be a number for sure or it might be,
    #for instance, a list of rgb values? I'm not sure so let's put this func here

    #If a and b are lists, they should have the same length, or the type would differ
    rst = []
    if type(a) == list and type(b) == list:
        for i in a:
            for j in b:
                rst.append(nminus(i,j))
    else:
        rst = a-b
        return rst
    return rst


def list_split(l, idx):
    left = []
    right = []
    for i in xrange(idx):
        left.append(l[i])
    for i in xrange(len(l)-idx):
        right.append(l[-i])
    return (left,right)

