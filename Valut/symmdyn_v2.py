#! usr/bin/python 
# -- DOB: 02 May 2015
# -- Edit: v2 - 8 atoms
import numpy as np 

def inv_symmdyn(Dyn): 

    para = np.array([ Dyn[1], Dyn[2], Dyn[3], Dyn[4], Dyn[7], Dyn[26], Dyn[27], Dyn[28], Dyn[31], Dyn[51], Dyn[52], Dyn[55], Dyn[60], Dyn[61], Dyn[62], Dyn[63], Dyn[102], Dyn[103] ]) 

    return para 
#-----------------------------#

def symmdyn(para): 

    temp = np.zeros(24) 

    temp[0] = -para[0]-para[1]-para[2]-para[3]-para[2]-para[3]-para[4]
    temp[6] = -para[0]-para[5]-para[6]-para[7]-para[6]-para[7]-para[8] 
    temp[11] = -para[1]-para[5]-para[9]-para[10]-para[9]-para[10]-para[11]    
              # 56 57 59 60 61 62 63 
              # 3  27 51 60 61 62 63   
    temp[15] = -para[2]-para[6]-para[9]-para[12]-para[13]-para[14]-para[15]
              # 96 97 98 99 101 102 103 
              # 4  28 52 60 63  102 103
    temp[20] = -para[3]-para[7]-para[10]-para[12]-para[14]-para[16]-para[17]
              # 112 113 114 115 117 118 119
              # 7   31  62  62  55  103 103
    temp[23] = -para[4]-para[8]-para[14]-para[14]-para[11]-para[17]-para[17]
    temp[1] = para[0] 
    temp[2] = para[1] 
    temp[3] = para[2] 
    temp[4] = para[3] 
    temp[5] = para[4] 
    temp[7] = para[5] 
    temp[8] = para[6] 
    temp[9] = para[7] 
    temp[10] = para[8] 
    temp[12] = para[9] 
    temp[13] = para[10] 
    temp[14] = para[11] 
    temp[16] = para[12] 
    temp[17] = para[13] 
    temp[18] = para[14] 
    temp[19] = para[15] 
    temp[21] = para[16] 
    temp[22] = para[17] 

    dyn = np.zeros(192) 

    dyn[np.array([ 0,  8, 16])] = temp[0] 
    dyn[np.array([ 1,  9, 17, 24, 32, 40])] = temp[1] 
    dyn[np.array([  2,  11,  21,  48,  80, 136])] = temp[2] 
    dyn[np.array([  3,   5,  10,  13,  18,  19,  56,  64,  72,  88, 120, 128])] = temp[3] 
    dyn[np.array([  4,   6,  12,  15,  22,  23,  96, 104, 144, 160, 176, 184])] = temp[4] 
    dyn[np.array([  7,  14,  20, 112, 152, 168])] = temp[5] 
    dyn[np.array([25, 33, 41])] = temp[6] 
    dyn[np.array([ 26,  35,  45,  49,  81, 137])] = temp[7] 
    dyn[np.array([ 27,  29,  34,  37,  42,  43,  57,  65,  73,  89, 121, 129])] = temp[8] 
    dyn[np.array([ 28,  30,  36,  39,  46,  47,  97, 105, 145, 161, 177, 185])] = temp[9] 
    dyn[np.array([ 31,  38,  44, 113, 153, 169])] = temp[10] 
    dyn[np.array([ 50,  83, 141])] = temp[11] 
    dyn[np.array([ 51,  53,  59,  69,  74,  82,  85,  93, 122, 131, 138, 139])] = temp[12] 
    dyn[np.array([ 52,  54,  84,  87,  98, 107, 142, 143, 146, 165, 179, 189])] = temp[13] 
    dyn[np.array([ 55,  86, 117, 140, 155, 170])] = temp[14] 
    dyn[np.array([ 58,  66,  75,  91, 125, 133])] = temp[15] 
    dyn[np.array([ 60,  70,  76,  95,  99, 106, 126, 135, 149, 162, 181, 187])] = temp[16] 
    dyn[np.array([ 61,  67,  77,  90, 123, 130])] = temp[17] 
    dyn[np.array([ 62,  68,  79,  92, 114, 115, 127, 134, 154, 157, 171, 173])] = temp[18] 
    dyn[np.array([ 63,  71,  78,  94, 101, 109, 124, 132, 147, 163, 178, 186])] = temp[19] 
    dyn[np.array([100, 108, 150, 166, 183, 191])] = temp[20] 
    dyn[np.array([102, 111, 148, 167, 180, 190])] = temp[21] 
    dyn[np.array([103, 110, 118, 119, 151, 156, 159, 164, 172, 174, 182, 188])] = temp[22] 
    dyn[np.array([116, 158, 175])] = temp[23] 

    return dyn 
#-------------------------#
