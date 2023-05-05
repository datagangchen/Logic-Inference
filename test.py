#!/usr/local/bin/python
from SSLdataloader import Gen_Data_loader, Dis_dataloader, load_vocab, save_vocab
from robustness import *
from struct_formula import *
from load_data import *
import numpy as np
import time 
#from target_robust import TARGET_ROBUST
from target  import TARGET, robust_loss
import tensorflow.compat.v1 as tf
#from target_lstm import TARGET_LSTM
from tensorflow.python.ops import tensor_array_ops, control_flow_ops



BATCH_SIZE = 10
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('length', 10, 'The length of toy data')
flags.DEFINE_string('model', "", 'Model NAME')

def runn_program():

	random.seed(11)
	np.random.seed(11)
	config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.5
	#sess = tf.compat.v1.Session(config=config)
	#sess.run(tf.global_variables_initializer())
	eval_file = 'save/eval_file.txt'
	vocab_file = 'save/vocab.txt'
	#name = ['anadataiot1.mat', 'anadataiot2.mat','anadataiot3.mat','anadataiot4.mat','anadataiot5.mat','anadataiot6.mat']
	#sigsets, time1, namelist, label = load_data(name[0])
	vocab = generate_vocab(1000,3,0,1,0,1)
	save_vocab(vocab_file,vocab)
	vocab = load_vocab(vocab_file)

	#params = (namelist,sigsets,time1,label)



	#target  = TARGET(BATCH_SIZE, 10, vocab)  # The oracle model



	#data_loader = Gen_Data_loader(BATCH_SIZE,FLAGS.length) # For testing	
	#data_loader.create_batches(eval_file)
	#batch = data_loader.next_batch()
	#embeds = sess.run(target.embeds,{target.x:batch})
	#loss  = robust_loss(embeds,params)
	data_loader = Dis_dataloader(BATCH_SIZE,FLAGS.length,vocab, params)
	#data_loader.load_train_data(eval_file)
	labels = 0 #data_loader.labels

 


	



	
	
 

 
	print(vocab)



 
	

if __name__ == '__main__':
	runn_program()
