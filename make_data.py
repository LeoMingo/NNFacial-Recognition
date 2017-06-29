import img_manip as im
from nndata import DataManip
from nndata import NNData

import numpy as np


dm = DataManip()
nnd = NNData()


img_meng_list = dm.img_name_list('../data_t')


#^---------------^
#Add processing messages later
def make_img_data(path='./data', filename='img_data.pkl', comp=1, dest='./'):
    imlarr = []
    if comp == 1:
        imlarr = [im.img_to_array(img_name)
                    for img_name in img_meng_list]
    else:
        for img_name in img_meng_list:
            IM = im.ImgManip(im.img_to_array(img_name))
            imlarr.append(IM.compress_img_pixels(comp_factor=comp))
    dm.pkl_w(imlarr, filename, dest_dir=dest)


def read_img_data(filename='./img_data.pkl'):
    data = dm.pkl_r(filename)
    return data


def main():
    make_img_data(path='../NNFacial-data', filename='meng_data-Comp20.pkl', comp=20)
    meng_img_data = read_img_data('meng_data-Comp20.pkl')

    color_size = len(meng_img_data[0][0][0])
    input_layer_list = []
    for i in meng_img_data:
        iarr = np.reshape(i, (im.matrix_size_prod(i.shape)*color_size, 1))
        il = iarr.tolist()
        input_layer_list.append(il)

    #Make Meng NN input layer data
    rst = ["Meng" for i in xrange(len(input_layer_list))]
    nnd.make_input_layer_data(input_layer_list, rst, filename="meng_input_layer_data-Comp20.pkl")
    #Make randomly generated img input layer data
    img_shape = meng_img_data[0].shape
    rdm_img_data = [im.gen_rdm_img(img_shape)
                        for i in xrange(len(input_layer_list))]
    rdm_input_layer_list = []
    for i in rdm_img_data:
        iarr = np.reshape(i, (im.matrix_size_prod(i.shape)*color_size, 1))
        il = iarr.tolist()
        rdm_input_layer_list.append(il)
    rdm_rst = ["Others" for i in xrange(len(rdm_input_layer_list))]
    nnd.make_input_layer_data(input_layer_list, rdm_rst, filename="rdm_input_layer_data-Comp20.pkl")


if __name__ == "__main__":
    main()

































