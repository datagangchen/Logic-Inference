import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
 
from struct_formula import *

class TARGET_ROBUST(object):
    def __init__(self, sequence_token, vocab, params):
        self.sequence_token = sequence_token   #tf variable the sequence of words batch_size x sequence length
        self.name = params[0]
        self.signal = params[1]
        self.time = params[2]
        self.label = params[3]
        self.vocab = vocab   #matrix
        self.tree =[]

    def robust(self):
        robustness =[]
        for tree in self.tree:
            rewards = reward(tree, self.name, self.signal, self.time)
            re = [a*b for a,b in zip(rewards,self.label)]
            robustness.append(min(re))
        return robustness
    def get_tree(self):
        tree =[]
        formulas = Formula([],[],self.name,width= 2) 
        sess = tf.Session() 
        init = tf.global_variables_initializer()
        sess.run(init)
        with sess.as_default():
            sentence_batch = self.sequence_token.eval() #get the token value from tensorflow variables

        for sentence_idx in sentence_batch:
            sentence =[]
            for idx in sentence_idx:
                sentence.append(self.vocab[idx])   # transform idx word to encoded word
            tree.append(formulas.sentence2tree(sentence))     # decode to tree
        self.tree = tree   



  