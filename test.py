#!/usr/local/bin/python

from robustness import *
from struct_formula import *
from load_data import *
import numpy as np
import time 
from target_robust import *
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
 
def runn_program():
	name = ['anadataiot1.mat', 'anadataiot2.mat','anadataiot3.mat','anadataiot4.mat','anadataiot5.mat','anadataiot6.mat']
	sigsets, time1, namelist, label = load_data(name[0])
	print(label)
	vocab = generate_vocab(10,3,0,1,0,1)
	params = (namelist,sigsets,time1,label)
	print(vocab)


	Lable = tf.constant(label)
	tokeni =[[2,3,4,5,6,7],[2,3,6,7,8,4]]
	token = tf.Variable(tokeni)
	target_robust = TARGET_ROBUST(token,vocab,params)

	target_robust.get_tree()
	robustn = target_robust.robust()
	print(robustn)



 
	

if __name__ == '__main__':
	runn_program()
