from nndata import DataManip
from nndata import NNData
import img_manip as im
import make_data as md

import random



def load_data():
#####################################
#Making training data and test data
#####################################
"""
###This section has been obsolete and it used to make training data
    dm = DataManip()
    nnd = NNData()

    if os.path.isfile("./ImgsMeng.pkl") == False:
        md.make_img_data(path="./data/MengData",
                         filename="ImgsMeng.pkl")

    imgs_Meng_fullQ = md.read_img_data(filename="ImgsMeng.pkl")
    for img in imgs_Meng_fullQ:
        imgobj = im.ImgManip(img)
        img_Meng_comp.append(imgobj.compress_img_pixels(comp_factor=10))
    rsts_Meng = [rst_Meng for i in xrange(len(imgs_Meng))]

    #normal compressed img data
    img_Meng_comp = []
    data_Meng_comp = dm.zip_rst(imgs_Meng_comp, rsts_Meng)

    rdm_data_comp_shape = img_Meng_comp[0].shape
    rdm_data_comp = dm.extend_data_with_rdm_imgs(img_Meng_comp,
                                                rdm_data_comp_shape,
                                                rst=rst_Others,
                                                rdm_ratio=1.0)
    train_data, test_data = list_split(rdm_data_comp, int(0.9*len(rdm_data_comp)))


    #compressed img difference
    imgs_Meng_comp_dif = [dm.mkDifArr(img_arr) for img_arr in imgs_Meng_comp]
    data_Meng_comp_dif = dm.zip_rst(imgs_Meng_dif, rsts_Meng)
    dif_len = len(imgs_Meng_comp_dif[0])
    rdm_data_comp_dif = dm.extend_data_with_rdm_imgs(data_Meng_comp_dif,
                                                    (dif_len, 1),
                                                    rst=rst_Others,
                                                     rdm_ratio=1.0)
"""

    dm = DataManip()
    nnd = NNData()

    output_rsts = []
    meng_data = load_input_layer_data("meng_data-Comp20.pkl")
    output_rsts.append(meng_data[0][1])
    rdm_data = load_input_layer_data("rdm_data-Comp20.pkl")
    output_rsts.append(rdm_data[0][1])
    data = meng_data + rdm_data

    meng_data = None
    rdm_data= None

    for i in range(5):
    #Maybe I'm a little paranoid,
    #but just make it more times
        random.shuffle(data)
    train_data_num = int(0.9*len(data))
    train_data = [data[i] for i in xrange(train_data_num)]
    test_data_num = len(data)-train_data_num
    test_data = [data[-i-1] for i in xrange(test_data_num)]


    return train_data, test_data, output_rsts







