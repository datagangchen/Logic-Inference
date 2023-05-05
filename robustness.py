import sys
from time import time
import numpy as np
# Robustness calculation
class STL_Sys:
    def __init__(self,name,signal,time):
        self.name=name
        self.signal=signal
        self.time=time

# The program will first read the formula and state file, then print out the raw formula
# and state. Unnecessary space and '\n' will be deleted afterward. A list is created from
# the input string and printed out. A compressed list is then used to build the tree and 
# show to the end-user. 

class Robustness:
	def __init__(self, argv):
		self.START_TIME = time()
		#self.option = option
		if len(argv) == 1:
			self.formula_filename = "example_formula.txt"
			self.state_filename   = "example_state.txt"
		elif len(argv) == 2:
			self.formula_filename = argv[1]
			self.state_filename   = "example_state.txt"
		elif len(argv) == 3:
			self.formula_filename = argv[1]
			self.state_filename   = argv[2]
		else:
			sys.exit("\033[1;31;47m Error: Too many input arguments! \033[0m")

	def SetTree(self, tree):
		self.tree = tree
	
	def find_nearest(self,a, a0):
		idx = np.abs(a - a0).argmin()
		return a.flat[idx]




	def GetTimeValues(self,system, interval):
		start_t = self.find_nearest(system.time, interval[0])
		end_t  = self.find_nearest(system.time, interval[-1])
		interval = np.array([start_t, end_t])
		if interval[0] == interval[-1]:
			time_values = np.array([interval[0]])
			return time_values
		ind_ti = np.nonzero(system.time >= interval[0])[0][0]
		
		# first time instant
		if system.time[ind_ti] == interval[0]:
			time_values = system.time[ind_ti]
			ind_ti = ind_ti + 1
		else:
			time_values = np.array([interval[0]])
		# Last time instant
		if interval[-1] == float('inf'):
			time_values = np.append(time_values, system.time[ind_ti:-1])
		else:
			ind_tf = np.nonzero(system.time >= interval[-1])[0][0]


			if not ind_tf:
				print('over time')
				time_values = np.append(time_values, system.time[ind_ti:-1])
				time_values = np.append(time_values,interval[-1])
			elif system.time[ind_tf] == interval[-1]:
				time_values = np.append(time_values, system.time[ind_ti:ind_tf])
			else:
				time_values = np.append(time_values, system.time[ind_ti: ind_tf - 1])
				time_values = np.append(time_values, interval[-1])

		#print(type(time_values))

		return time_values

	def Eval(self, system, interval=np.array([])):
		tree = self.tree
		if tree is None: return 0

		if len(interval) == 0: interval = np.array([0,0])

		if tree.cargo['Value'] == 'ev':
			phi_interval = tree.cargo['Bound']
			phi_interval = np.amax(np.array([np.array([phi_interval[0], phi_interval[-1]]),np.array([0, 0])]),axis= 0)
			phi_interval[0] = np.min(phi_interval)
			next_interval = phi_interval + np.array([interval[0],interval[-1]])
			self.tree = tree.right
			val_array, time_values = self.Eval(system, next_interval)
			if phi_interval[-1] != float('inf'):
				time_values = np.append(time_values, time_values[-1] + phi_interval[-1])
				val_array = np.append(val_array, val_array[-1])

			value_arr = np.empty([0])
			time_arr = np.empty([0])
			start_t = self.find_nearest(time_values, phi_interval[0]+interval[0])
			end_t = self.find_nearest(time_values,phi_interval[0]+interval[-1])
		
			find_interval= np.where(np.logical_and(time_values >= start_t, time_values <= end_t))[0]

			for index in range(len(time_values[find_interval])):
				start_t = self.find_nearest(time_values, time_values[find_interval[index]])
				end_t  = self.find_nearest(time_values, time_values[find_interval[index]]+phi_interval[-1]-phi_interval[0])
				find_phi = np.where(np.logical_and(time_values >= start_t, time_values <= end_t))[0]
				value_arr = np.append(value_arr, np.max(val_array[find_phi]))
				time_arr = np.append(time_arr, time_values[find_interval[index]]-phi_interval[0])

			return value_arr, time_arr

		elif tree.cargo['Value'] == 'alw':
			phi_interval = tree.cargo['Bound']
			phi_interval = np.amax(np.array([np.array([phi_interval[0], phi_interval[-1]]),np.array([0, 0])]),axis= 0)
			phi_interval[0] = np.min(phi_interval)
			next_interval = phi_interval + np.array([interval[0],interval[-1]])
			self.tree = tree.right
			val_array, time_values = self.Eval(system, next_interval)
			

			if phi_interval[-1] != float('inf'):
				time_values = np.append(time_values, time_values[-1] + phi_interval[-1])
				val_array = np.append(val_array, val_array[-1])
			value_arr = np.empty([0])
			time_arr = np.empty([0])

			start_t = self.find_nearest(time_values, phi_interval[0]+interval[0])
			end_t = self.find_nearest(time_values,phi_interval[0]+interval[-1])
			find_interval= np.where(np.logical_and(time_values >= start_t, time_values <= end_t))[0]

			for index in range(len(time_values[find_interval])):
				start_t = self.find_nearest(time_values, time_values[find_interval[index]])
				end_t  = self.find_nearest(time_values, time_values[find_interval[index]]+phi_interval[-1]-phi_interval[0])
				find_phi = np.where(np.logical_and(time_values >= start_t, time_values <= end_t))[0]
				value_arr = np.append(value_arr, np.min(val_array[find_phi]))
				time_arr = np.append(time_arr, time_values[find_interval[index]]-phi_interval[0])
			return value_arr, time_arr

		elif tree.cargo['Value'] == 'not':
			self.tree = tree.right
			val_array, time_values = self.Eval(system, interval)
			return -val_array, time_values

		elif tree.cargo['Value'] == 'and':
			self.tree = tree.left
			val_array1, time_values1 = self.Eval(system, interval)
			self.tree = tree.right
			val_array2, time_values2 = self.Eval(system, interval)
			# check data coherence
			if len(val_array1) != len(time_values1) or len(val_array2) != len(time_values2):
				print('RobustAnd: lengths of time steps and signal are different.')

			start_time = np.max(np.array([time_values1[0], time_values2[0]]))
			end_time   = np.min(np.array([time_values1[-1], time_values2[-1]]))
			start_t = self.find_nearest(time_values1, start_time)
			end_t  = self.find_nearest(time_values1, end_time)
			index_and = np.where(np.logical_and(time_values1 >= start_t, time_values1 <= end_t))[0]
			time_values = time_values1[index_and]
			val_array = np.amin(np.array([val_array1[index_and],val_array2[index_and]]),axis= 0)
			return val_array, time_values

		elif tree.cargo['Value'] == 'or':
			self.tree = tree.left
			val_array1, time_values1 = self.Eval(system, interval)
			self.tree = tree.right
			val_array2, time_values2 = self.Eval(system, interval)
			# check data coherence
			if len(val_array1) != len(time_values1) or len(val_array2) != len(time_values2):
				print('RobustAnd: lengths of time steps and signal are different.')

			start_time = np.max(np.array([time_values1[0], time_values2[0]]))
			end_time   = np.min(np.array([time_values1[-1], time_values2[-1]]))
			start_t = self.find_nearest(time_values1, start_time)
			end_t  = self.find_nearest(time_values1, end_time)
			index_and = np.where(np.logical_and(time_values1 >= start_t, time_values1 <= end_t))[0]
			time_values = time_values1[index_and]
			val_array = np.amax(np.array([val_array1[index_and],val_array2[index_and]]),axis= 0)
			return val_array, time_values

		elif tree.cargo['Value'] == "until":
			unt_interval = tree.cargo['Bound']
			unt_interval = np.amax(np.array([np.array([unt_interval[0], unt_interval[-1]]),np.array([0, 0])]),axis= 0)
			unt_interval[0] = np.min(unt_interval)
			interval1 = np.array([interval[0], unt_interval[-1]+interval[1]])
			interval2 = unt_interval + np.array([interval[0],interval[-1]])
			self.tree = tree.left
			value_arr1, time_values1 = self.Eval(system,interval1)
			self.tree = tree.right
			value_arr2, time_values2 = self.Eval(system, interval2)
			if unt_interval[-1] != float('inf'):
				time_values1 = np.append(time_values1, time_values1[-1] + unt_interval[-1])
				value_arr1  = np.append(value_arr1,value_arr1[-1])
				time_values2 = np.append(time_values2, time_values2[-1] + unt_interval[-1])
				value_arr2 = np.append(value_arr2, value_arr2[-1])

			value_arr = np.empty([0])
			value_arr_t = np.empty([0])

			find_interval = np.where(np.logical_and(system.time >= interval[0], system.time <= interval[-1]))[0]

			for index in range(len(system.time[find_interval])):
				phi_interval = np.array([system.time[find_interval[index]], system.time[find_interval[index]] + \
										 unt_interval[-1]])
				find_interval_u = np.where(np.logical_and(system.time >= phi_interval[0]+unt_interval[0], system.time <= phi_interval[-1]))[0]
				for index_u in range(len(system.time[find_interval_u])):
					find_phi_1 = np.where(np.logical_and(time_values1 >=system.time[find_interval[index]], time_values1 <= \
									system.time[find_interval_u[index_u]]-system.time[find_interval_u[0]]))[0]
					find_phi_2 = np.where(np.logical_and(time_values2 >= system.time[find_interval_u[index_u]], time_values2 <=\
														 phi_interval[-1]))[0]

					value_arr_t = np.append(value_arr_t, np.amin(np.append(value_arr2[find_phi_2], min(value_arr1[find_phi_1]))))
				value_arr =np.append(value_arr, np.amax(value_arr_t,axis =0))
				value_arr_t =np.empty([0])
			return value_arr, system.time[find_interval]

		elif tree.cargo['Value'][1] in ['<', '<=']:
			ind_name = system.name.index(tree.cargo['Value'][0])
			signal = system.signal[ind_name]
			time_values = self.GetTimeValues(system, interval)
			start_t = self.find_nearest(system.time, time_values[0])
			end_t  = self.find_nearest(system.time, time_values[-1])
			id_duration =   np.where(np.logical_and(system.time >= start_t, system.time <= end_t))[0]
			val_array =  tree.cargo['Value'][2] - signal[id_duration]

			return val_array, time_values

		elif tree.cargo['Value'][1] in ['>=', '>']:
			ind_name = system.name.index(tree.cargo['Value'][0])
			signal = system.signal[ind_name]
			time_values = self.GetTimeValues(system,interval)
			start_t = self.find_nearest(system.time, time_values[0])
			end_t  = self.find_nearest(system.time, time_values[-1])
			id_duration =   np.where(np.logical_and(system.time >= start_t, system.time <= end_t))[0]
			val_array = signal[id_duration] - tree.cargo['Value'][2]
			return val_array, time_values
		else:
			sys.exit("\033[1;31;47m\t Error: Unrecognized character to evaluate!\t\033 [0m")

	def Eval_Robust(self, system, interval=np.array([])):
		robustness, interval = self.Eval(system, interval)
		return robustness
