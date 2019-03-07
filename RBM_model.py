# coding=utf-8

import  tensorflow as tf 
import model_helper

class RBM:
	#用于过滤推荐系统的RBM

	def __init__(self,FLAGS):
		self.FLAGS = FLAGS
		self.weight_initializer = model_helper._get_weight_init()
		self.bias_initializer = model_helper._get_bias_init()
		self.init_parameter()


	def init_parameter(self):
		with tf.variable_scope('Network_parameter'): 	#在这个域下变量共享  
			self.W = get_variable('Weights', shape = (self.FLAGS.num_v, self.FLAGS.num_h), initializer = self.weight_initializer)
			self.bh = get_variable('hidden_bias',shape = (self.FLAGS.num_h), initializer = self.bias_initializer)
			self.bv = get_variable('visible_bias', shape = (self.FLAGS.num_v), initializer = self.bias_initializer)
			#get_variable会给变量加一个前缀（变量域名）括号第一个参数是变量在该变量域中的名称

	def _sample_h(self,v):

        #传入观测到的数据然后计算一个隐藏节点的激活输出 返回激活概率与伯努利采样的隐藏节点
        with tf.name_scope('sampling_hidden_units'):		
        	a = tf.nn.bias_add(tf.matmul(v,self.W), self.bh)  #matmul(mat1,mat2) 计算两个矩阵的乘积 
        	#tf.nn.bias_add(a,b) 表示b为bias（必须是一维的）支持广播，即可将一维bias加到高维数据上
        	p_h_v = tf.nn.sigmoid(a)  #a is 从v->h的input, p_h_v is activated概率值
        	h_ = self._bernouille_sampling(p_h_v, shape = [self.FLAGS.batch_size, int(p_h_v.shape[-1])])

        	return p_h_v, h_

    def _sample_v(self,h):
        #传入隐藏节点序列并计算一个可见节点的状态，返回激活概率与伯努利采样的可见节点
        with tf.name_scope('sampling_visible_units'):
        	a = tf.nn.bias_add(tf.matmul(h,tf.transpose(self.W, [1,0])),self.bv)
        	p_v_h = tf.nn.sigmoid(a)  #从h->v的输入经函数激活后的概率值
        	v_ = self._bernouille_sampling(p_v_h, shape = [self.FLAGS.batch_size, int(p_v_h.shape[-1])])

        	return p_v_h, v_

    def optimize(self,v):
    	#Gibbs sampling and gradients calculation and update parameters

    	with tf.name_scope('optimization'):
    		v0,vk,ph0,phk, _ = self._gibbs_sampling(v)
    		dW , db_h, db_v = self._compute_gradients(v0,vk,ph0,phk)
    		update_op = self._update_parameter(dW,db_h,db_v)

    	with tf.name_scope('accuracy'):
    		mask = tf.where(tf.less(v0,0.0), x = tf.zeros_like(v0), y = tf.ones_like(v0))
    		bool_mask = tf.cast(tf.where(tf.less(v0,0.0), x = tf.zeros_like(v0), y = tf.ones_like(v0)), dtype = tf.bool)
    		acc = tf.where(bool_mask, x = tf.abs(tf.subtract(v0,vk)), y = tf.zeros_like(v0))
    		n_values = tf.reduct_sum(mask)
    		acc = tf.subtract(1.0, tf.div(tf.reduce_sum(acc),n_values))

    	return update_op, acc









































