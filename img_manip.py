"""
reduce pixels
amplify pixels

change color scales
--rotate by degrees
(others)

remove img green screen
composite images

generate images with random noises
insert images with random noises
"""

from math import floor
from random import randint
import random

import numpy as np
from scipy import misc


def matrix_size_prod(mat_shape):
    return mat_shape[0]*mat_shape[1]

def remainder(n,d):
    return int(n-(floor(n/d)*d))

def oddnum(num):
    if remainder(num,2) == 0:
        return False
    else:
        return True

def avg_arrs(arrays):
    #Calculate the average array of a given list of arrays,
    #which should have the same size
    arr_size = len(arrays[0])
    avg_arr = [0 for i in xrange(arr_size)]
    for i in xrange(arr_size):
        for a in arrays:
            avg_arr[i] += a[i]
        avg_arr[i] /= float(len(arrays))
    return avg_arr

class ImgManip(object):
    def __init__(self, img):
        self.img = img
        self.color_size = len(self.img[0][0])

    def compress_img_pixels(self, comp_factor=1, symmetrical_deletion=True):
        img_cpy = self.img.tolist()
        #Reshape the original image by cutting of the edges of the remainders
        #of rows and columns being divided by comp_factor,
        #set symmetrical_deletion to True to delete from
        #top, bottom, left, right 4 directions, otherwise only from
        #right and bottom directions
        row = len(img_cpy)
        col = len(img_cpy[0])
        nr = int(floor(row/comp_factor))
        nc = int(floor(col/comp_factor))
        rnr_rem=remainder(row,nr)
        cnc_rem=remainder(col,nc)

        if symmetrical_deletion==True:
            nr_half_rem = int(rnr_rem/2)
            nc_half_rem = int(cnc_rem/2)
            for r in range(nr_half_rem):
                img_cpy.pop(0)
                img_cpy.pop(-1)
            if oddnum(rnr_rem):
                img_cpy.pop(-1)

            for c in range(nc_half_rem):
                for r in xrange(len(img_cpy)):
                    img_cpy[r].pop(0)
                    img_cpy[r].pop(-1)
            if oddnum(cnc_rem):
                for r in xrange(len(img_cpy)):
                    img_cpy[r].pop(-1)
        else:
            for r in xrange(rnr_rem):
                img_cpy.pop(-1)
            for c in xrange(cnc_rem):
                for r in xrange(len(img_cpy)):
                    img_cpy[r].pop(-1)
        #Return a list of colors
        new_img_arr = [[] for r in xrange(nr)]
        color_l = []
        for r in xrange(nr):
            for c in xrange(nc):
                color_l = []
                for cfr in xrange(comp_factor):
                    for cfc in xrange(comp_factor):
                        color_l.append(
                                img_cpy[r*comp_factor+cfr][c*comp_factor+cfc])
                new_img_arr[r].append(avg_arrs(color_l))


        img_arr = np.array(new_img_arr)
        comp_img = np.reshape(img_arr, (nr,nc,self.color_size))
        return comp_img

    def rm_green_screen(self):
        img_cpy = self.img
        #This only suits for my data after hours of testing
        for row in xrange(img_cpy.shape[0]):
            for col in xrange(img_cpy.shape[1]):
                r,g,b = img_cpy[row][col]
                if (r < 120) and (g > 130):
                    img_cpy[row][col] = [255,255,255]
	return img_cpy

    def insert_noises(self, noise_ratio=0.0):
        #For now this will only insert generated rgb tuples
        msp = matrix_size_prod(self.img)
        noise_pix_num = msp*noise_ratio
        img_cpy = np.reshape(self.img, (msp, self.color_size))
        rdm_idx_l = [0.0 for i in xrange(msp)]
        for i in xrange(noise_pix_num):
            rdm_idx_l[i] = [1.0]
        random.shuffle(rdm_idx_l)

        if self.color_size == 3:    #RGB
            for i in xrange(msp):
                if rdm_idx_l[i] == 1.0:
                    img_cpy[i] = gen_rgb_tuple()
        #else if self.color_size == 4:  #RGBA
        #else:
        #   ...
        return img_cpy

    def left_rotate(self, angle):
        img_cpy = self.img
        misc.imrotate(angle, img_cpy)
        return img_cpy
    def right_rotate(self, angle):
        img_cpy = self.img
        misc.imrotate(360-angle, img_cpy)
        return img_cpy

    def displace_pixels(self, direction='up', displacement_num=1):
        img_cpy = self.img
        if direction=='up':
            for r in xrange(displacement_num):
                temp = img_cpy.pop(0)
                img_cpy.append(temp)
                temp = None
        elif direction=='down':
            for r in xrange(displacement_num):
                temp = img_cpy.pop(-1)
                #^--------------^
                img_cpy.preppend(temp)
                temp = None
        elif direction=='left':
            for c in xrange(displacement_num):
                for r in xrange(len(img_cpy)):
                    temp = img_cpy[r].pop(0)
                    img_cpy[r].append(temp)
                    temp = None
        elif direction=='right':
            for c in xrange(displacement_num):
                for r in xrange(len(img_cpy)):
                    temp = img_cpy[r].pop(-1)
                    img_cpy[r].preappend(temp)
                    temp = None
        return img_cpy

#    def composite_onto(self, background_img, center=(0,0)):
        #Paste the image onto the background image
        #Blablabla





def gen_rgb_tuple():
    rdmrgb = [random.randint(0,255) for i in xrange(3)]
    return rdmrgb

def gen_rdm_img(img_shape, color_size=3):
    img = [gen_rgb_tuple() for i in xrange(matrix_size_prod(img_shape))]
    img = np.array(img)
    img_arr = np.reshape(img, img_shape)
    return img_arr


def img_to_array(img_name):
    arr = misc.imread(img_name)
    return arr

def array_to_img(name, arr):
    misc.imsave(name, arr)






