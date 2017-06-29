from nn import NN
from img_manip import matrix_size_prod as msp
from data_prep import load_data

#imgs = [utils.img_to_array(img_name) for img_name in img_names]
"""
initial_rst = [[0],[0]]
rst[0][0] = 1.0, rst indicates Meng Meng
rst[1][0] = 1.0, rst indicates others
"""




def main():

    train_data, test_data, output_rsts = load_data()

#######################################
#Training
#######################################
    input_layer_size = len(train_data[0][0])
    net = NN((input_layer_size, 30, 2), output_rsts)

    net.SGD(train_data, mini_batch_size=10, epochs=30, eta=3.0, test_data=test_data)




if __name__ == "__main__":
    main()





