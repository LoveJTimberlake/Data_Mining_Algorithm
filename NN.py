# coding=utf-8


import numpy as np 
import math 
import scipy.special

class NeuralNetwork :

	def __init__(self,inputnode_num, hiddennode_num, outputnode_num,learingrate=:0.01)
		self.inputnode_num = inputnode_num
		self.hiddennode_num = hiddennode_num
		self.outputnode_num = outputnode_num

		self.wih = np.random.normal(0.0,pow(self.hiddennode_num,-0.5),(self.hiddennode_num,self.inputnode_num))
		self.who = np.random.normal(0.0, pow(self.outputnode_num,-0.5), (self.outputnode_num, self.hiddennode_num) )

		self.lr = learingrate
		self.activation_func = lambda x : scipy.special.expit(x)

	def train(self,input_list,targets_list):
		
		inputs = np.array(input_list, ndmin = 2).T
		targets = np.array(targets_list, ndmin = 2).T
		
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		final_inputs = np.dot(self.who,hidden_inputs)
		final_outputs = self.activation_func(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = np.dot(self.who.T, output_errors)

		self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outpts))
		self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))



	def predict(self,input_list):
		inputs = np.array(input_list,ndmin = 2).T

		hidden_inputs - np.dot(self.wih, inputs)
		hidden_outpts = self.activation_func(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)

		return final_outputs











