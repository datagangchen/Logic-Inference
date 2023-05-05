import numpy as np
from struct_formula import Formula, reward, robust

class Gen_Data_loader():
    def __init__(self, batch_size,length):
        self.batch_size = batch_size
        self.token_stream = []
        self.length = length
    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.length:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size,length,vocab,params):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.length = length
        self.name =params[0]
        self.signal = params[1]
        self.time = params[2]
        self.label = params[3]
        self.vocab=vocab
  
    def load_train_data(self, data_file):
        # Load data
        examples = []
        labels =[]
        formulas = Formula([],[],self.name,width= 2)   
        with open(data_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                sentence =[]

                for word in parse_line:
                    sentence.append(self.vocab[word])

                tree= formulas.sentence2tree(sentence)    # decode to tree
                rewards = reward(tree, self.name,  self.signal,  self.time)
                re = [a*b for a,b in zip(rewards,self.label)]
                labels.append(min(re))
                examples.append(parse_line)

        self.sentences = np.array(examples)

        #Generate labels


        self.labels = np.array(labels)


        #Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

def load_vocab(data_file):
    vocab =[]
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [float(x) for x in line]    
            vocab.append(parse_line)

    return vocab

def save_vocab(data_file,vocab):
    with open(data_file,'r+') as f:
        f.truncate(0)
        f.close()
    with open(data_file,'w') as f:
        for poem in vocab:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            f.write(buffer)
  





