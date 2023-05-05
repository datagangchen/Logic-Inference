#!/usr/local/bin/python3

from binary_tree import *
from fractions import Fraction
import numpy as np
import random
from robustness import *
import multiprocessing
class Formula:
    def __init__(self, struc =None, data =None, name= None, low_time =0.0, up_time = 1.0, low_pre =0.0, up_pre=1.0, width = None):

        self.struc = struc
        self.data = data
        time = []
        for index in range(len(data)):
            if data[index]['Bound'] is not None:
                time = np.append(time, data[index]['Bound'])

        scale = []
        for index in range(len(data)):
            if data[index]['Value'][1] in ['>','>=','<','<=']:
                scale = np.append(scale,data[index]['Value'][2])

        name_value= []
        for index in range(len(data)):
            if data[index]['Value'][0] in name:
                name_value = np.append(name_value, 1+name.index(data[index]['Value'][0]))

        dir =[]
        comp = ['>', '>=', '<', '<=']
        for index in range(len(data)):
            if data[index]['Value'][1] in comp:
                dir = np.append(dir, comp.index(data[index]['Value'][1])+1)


        self.time = time
        self.scale = scale
        self.name_value = name_value
        self.dir =dir
        self.name = name
        self.width = width
        self.up_time = up_time
        self.up_pre = up_pre
        self.low_time = low_time
        self.low_pre = low_pre


    def get_action_tree(self,vector):
        #vector has 11 element [0-4 ] struc for tree, [5-6] time param, [7-9] predicate if applicable, [10] method
        method = vector[-1]
        if vector[2]>0:
            if vector[0] == 3:
                cargo = {'Value': 'alw', 'Bound':[vector[5], vector[6]+vector[5]]}
                name1 = self.name[int(vector[7])]
                dir =['>', '>=', '<', '<=']
                right = Tree({'Value':[name1, dir[int(vector[8])], vector[9] ], 'Bound': None})
                tree = Tree(cargo)
                tree.right = right
            elif vector[0] ==4:
                cargo = {'Value': 'ev', 'Bound':[vector[5], vector[6]+vector[5]]}
                name1 = self.name[int(vector[7])]
                dir =['>', '>=', '<', '<=']
                right = Tree({'Value':[name1, dir[int(vector[8])], vector[9]], 'Bound': None})
                tree = Tree(cargo)
                tree.right = right
            else:
                print('Invalid vector')
                return None
        elif vector[2] == 0:
            if vector[0] == 3:
                cargo = {'Value': 'alw', 'Bound':[vector[5], vector[6]]}
                tree = Tree(cargo)
            elif vector[0] ==4:
                cargo = {'Value': 'ev', 'Bound':[vector[5], vector[6]]}
                tree = Tree(cargo)
            else:
                print('Invalid vector')
                return None
        else:
            print('Invalid vector')
            return None

        return tree, method

    def combine_formula(self, tree_pre, tree_post, method, intervel = None):
        if method ==1:
            cargo ={'Value': 'and', 'Bound': None}
            tree = Tree(cargo)
            tree.left = tree_pre
            tree.right = tree_post
            return tree
        elif method == 2:
            cargo ={'Value': 'or', 'Bound': None}
            tree = Tree(cargo)
            tree.left = tree_pre
            tree.right = tree_post
            return tree
        elif method == 5:
            cargo ={'Value': 'Until', 'Bound': intervel}
            tree = Tree(cargo)
            tree.left = tree_pre
            tree.right = tree_post
            return tree    
        elif method == 3 or method == 4:
            tree = tree_pre
            tree.right = tree_post
            return tree
            
        elif method == 0:
            return tree_pre
        # initialize
        else:
            print('Invalid method')

    def get_newstate_tree(self,action, method):
        return self.combine_formula(action,self.get_tree(),method)

    def get_tree(self):
        return DecodeSuccinct(self.struc, self.data)
    def update_state(self, tree):
        struc=[]
        data =[]
        EncodeSuccint(tree,struc,data)
        self.__init__(struc,data,self.name,width = self.width)

    def state_vector(self):
        zero = np.zeros(2*self.width - len(self.time))
        time = np.concatenate((self.time,  zero), axis=0)
        zero = np.zeros(self.width - len(self.scale))
        scale =np.concatenate((self.scale,zero), axis=0)
        zero = np.zeros(self.width - len(self.name_value))
        name_value = np.concatenate((self.name_value,zero),axis =0)
        zero = np.zeros(self.width-len(self.dir))
        dir =np.concatenate((self.dir, zero), axis =0)     
        zero_struc =np.zeros(5*self.width - len(self.struc))
        struct =np.concatenate((self.struc,zero_struc), axis=0)
        vector = np.concatenate((struct,time,scale,name_value,dir), axis =0)
        return vector
    def model(self, act_vector):
        tree1, method = self.get_action_tree(act_vector)
        tree2 =self.get_tree()
        tree = self.combine_formula(tree1,tree2,method)
        self.update_state(tree)
        return self.state_vector()
    def vector_tree(self,vector):
        struct = vector[0:5*self.width]
        tim   = vector[5*self.width:7*self.width]
        sca  = vector[7*self.width:8*self.width]
        name_value   = vector[8*self.width:9*self.width]
        dirc   = vector[9*self.width:-1] 
        idx_stru = np.nonzero(struct)[0]
        struc = struct[0:idx_stru[-1]+3]
        idx_time = np.nonzero(tim)[0]
        timee = tim[0:idx_time[-1]+1]
        idx_sca = np.nonzero(sca)[0]
        scale = sca[0:idx_sca[-1]+1]
        idx_name = np.nonzero(name_value)[0]
        name = name_value[0:idx_name[-1]+1]
        idx_dir = np.nonzero(dirc)[0]
        dir = dirc[0:idx_dir[-1]+1]
        data =[]
        comp = ['>', '>=', '<', '<=']
        for _ in range(len(struc)):
            b = struc[0]
            struc.pop(0)
            if b>=1:
                if b == 1:
                    data.append({'Value':'and','Bound':None})
                elif b ==2:
                    data.append({'Value':'or', 'Bound':None})
                elif b ==3:
                    data.append({'Value':'alw','Bound':[timee[0],timee[1]]})
                    timee.pop(0)
                    timee.pop(0)
                elif b ==4:
                    data.append({'Value':'ev','Bound':[timee[0],timee[1]]})
                    timee.pop(0)
                    timee.pop(0)
                elif b ==5:
                    data.append({'Value':[self.name[int(name[0]-1)], comp[int(dir[0]-1)], scale[0]], 'Bound':None})
                    dir.pop(0)
                    scale.pop(0)
                    name.pop(0)
        struc = struct[0:idx_stru[-1]+3]
        tree = DecodeSuccinct(struc,data)
        return tree

    def random_intervel(self,low,high):
        intervel =[]
        intervel.append(round(random.uniform(low,high),2))
        intervel.append(round(random.uniform(low,high),2))
        intervel.sort()
        return intervel

    def random_tree(self,Length):
        if Length == 1: 
            operator = random.randint(3,4)

            if operator == 3:
                cargo ={'Value':'alw', 'Bound':self.random_intervel(self.low_time,self.up_time)}
            else:
                cargo ={'Value':'ev', 'Bound':self.random_intervel(self.low_time,self.up_time)}

            root = Tree(cargo)
            root.left = None

            name_idx = random.randint(0,len(self.name)-1)
            direc =['<=', '>', '>=','<']
            op_idx = random.randint(0,3)
            root.right = Tree({'Value':[self.name[name_idx], direc[int(op_idx)], round(random.uniform(self.low_pre,self.up_pre),3)], 'Bound': None})
            return root 

        if Length == 2:
            operator = random.randint(3,4)
            if operator == 3:
                cargo ={'Value':'alw', 'Bound':self.random_intervel(self.low_time,self.up_time)}
            else:
                cargo ={'Value':'ev', 'Bound':self.random_intervel(self.low_time,self.up_time)}

            root  = Tree(cargo)
            root.left = None
            root.right = self.random_tree(Length-1)
            
            return root

        if Length >= 3:
            operator = random.randint(1,2)
            if operator == 1:
                cargo ={'Value':'and', 'Bound':None}
            else:
                cargo ={'Value':'or', 'Bound':None}
            root = Tree(cargo)
            root.left  = self.random_tree(Length-2)
            root.right = self.random_tree(Length-2)
            return root

    def update_agenda(self, agenda, chart, action):
        chart.append(action)
        cargo ={'Value':'alw', 'Bound':self.random_intervel(self.low_time,self.up_time)}
        root = Tree(cargo)
        root.left = None
        root.right = None
        agenda.append(self.combine_formula(root, action,3))
        cargo ={'Value':'ev', 'Bound':self.random_intervel(self.low_time,self.up_time)}
        root = Tree(cargo)
        agenda.append(self.combine_formula(root, action,3))
        for tree in chart:
            if tree is not None:
                agenda.append(self.combine_formula(tree,action,2))
                agenda.append(self.combine_formula(tree,action,1))

    def agenda_vector(self,agenda):
        state =[]
        for i in range(len(agenda)):
            self.update_state(agenda[i])
            state.append([np.array(self.state_vector())])
        return state 

    def init_agenda(self, agenda,chart, N):  # N is the number of formula in agenda
        chart =[]
        while N>0:
            agenda.append(self.random_tree(1))
            N -=1

    def word2tree(self,word):

        if word[1] == 1:
            cargo = {'Value': 'alw', 'Bound':[word[5], word[6]]}
            name1 = self.name[int(word[2])]
            dir =['>=', '<']
            right = Tree({'Value':[name1, dir[int(word[3])-1], word[4] ], 'Bound': None})
            tree = Tree(cargo)
            tree.right = right
  
        elif word[1] ==2:
            cargo = {'Value': 'ev', 'Bound':[word[5], word[6]]}
            name1 = self.name[int(word[2])]
            dir =['>=', '<']
            right = Tree({'Value':[name1, dir[int(word[3])-1], word[4] ], 'Bound': None})
            tree = Tree(cargo)
            tree.right = right
        
        elif word[1] ==3:
            cargo = {'Value': 'alw', 'Bound':[word[5], word[6]]}
            name1 = self.name[int(word[2])]
            dir =['>=', '<']
            right = Tree({'Value':[name1, dir[int(word[3])-1], word[4] ], 'Bound': None})
            tree = Tree(cargo)
            cargo1 = {'Value': 'ev', 'Bound':[word[7], word[8]]}
            right1 = Tree(cargo1)
            right1.right = right
            tree.right = right1

        elif word[1]==4:

            cargo = {'Value': 'ev', 'Bound':[word[5], word[6]]}
            name1 = self.name[int(word[2])]
            dir =['>=', '<']
            right = Tree({'Value':[name1, dir[int(word[3])-1], word[4] ], 'Bound': None})
            tree = Tree(cargo)
            cargo1 = {'Value': 'alw', 'Bound':[word[7], word[8]]}
            right1 = Tree(cargo1)
            right1.right = right
            tree.right = right1

        else:
            print('Invalid word')


        return tree, word[0] 

    def sentence2tree(self, sentence):
        flag = True
        for word in sentence:
            if flag:
                pre_tree, method = self.word2tree(word)
                flag = False
            else:
                post_tree, method  = self.word2tree(word)
                pre_tree = self.combine_formula(pre_tree,post_tree,method)

        return  pre_tree
        

def Init_state(name,signal,time, Num):
    formulas = Formula([],[],name,width= Num)  
    formulas.up_time = max(time)
    formulas.up_pre = 1  # np.max(signal)
    formulas.low_pre = 0 #np.min(signal)
    Num = random.randint(Num-1, Num)
    tree = formulas.random_tree(Num)
    formulas.update_state(tree)
    return formulas.state_vector(), formulas





def robust(param):
    tree = param[0]
    name = param[1]
    signal = param[2]
    time = param[3]
    system = STL_Sys(name, signal, time)
    Robust = Robustness(sys.argv)
    Robust.SetTree(tree)
    return Robust.Eval(system)[0]

def reward(tree, name, signalset, time):
    ls = len(signalset)
    robustness=[]
    for index in range(ls):
        pp =(tree,name, signalset[index],time)
        robustness.append(robust(pp))
    return robustness

def poolreward(tree,name,signalset,time):
    ls = np.size(signalset,0)
    params=[]
    for index in range(ls):
        pp = (tree, name, signalset[index], time)
        params.append(pp)

    pool = multiprocessing.Pool()
    results = pool.map(robust, params)
    return results





































