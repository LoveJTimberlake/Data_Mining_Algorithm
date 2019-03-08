# coding=utf-8

import  tensorflow as tf 

def get_bias_init():
	return tf.zeros_initializer()

def get_weight)init():
	return tf.random_normal_initializer(mean = 0.0, stddev = 0.5)

class RBM:
	#用于过滤推荐系统的RBM

	def __init__(self,FLAGS):
		self.FLAGS = FLAGS
		self.weight_initializer = get_weight_init()
		self.bias_initializer = get_bias_init()
		self.init_parameter()


	def init_parameter(self):
		with tf.variable_scope('Network_parameter'): 	#在这个域下变量共享  
			self.W = get_variable('Weights', shape = (self.FLAGS.num_v, self.FLAGS.num_h), initializer = self.weight_initializer)
			self.bh = get_variable('hidden_bias',shape = (self.FLAGS.num_h), initializer = self.bias_initializer)
			self.bv = get_variable('visible_bias', shape = (self.FLAGS.num_v), initializer = self.bias_initializer)
			#get_variable用来初始化变量，会给变量加一个前缀（变量域名）括号第一个参数是变量在该变量域中的名称

	def _sample_h(self,v):

        #传入观测到的数据然后计算一个隐藏节点的激活输出 返回激活概率与伯努利采样的隐藏节点
        with tf.name_scope('sampling_hidden_units'):		
        	a = tf.nn.bias_add(tf.matmul(v,self.W), self.bh)  #matmul(mat1,mat2) 计算两个矩阵的乘积 v.shape = [K,num_v]
        	#tf.nn.bias_add(a,b) 表示b为bias（必须是一维的）支持广播，即可将一维bias加到高维数据上  最后结果为K*num_h
        	p_h_v = tf.nn.sigmoid(a)  #a is 从v->h的input, p_h_v is activated概率值（shape = [K*num_h]） sigmoid()对矩阵每个元素进行一次激活
        	h_ = self._bernouille_sampling(p_h_v, shape = [self.FLAGS.batch_size, int(p_h_v.shape[-1])])

        	return p_h_v, h_

    def _sample_v(self,h): #h是一个矩阵
        #传入隐藏节点序列并计算一个可见节点的状态，返回激活概率与伯努利采样的可见节点
        with tf.name_scope('sampling_visible_units'):
        	a = tf.nn.bias_add(tf.matmul(h,tf.transpose(self.W, [1,0])),self.bv) 	#transpose(mat,perm) perm(列表)是表示如何转置 第i个数字表示新的矩阵第i个轴对应原来矩阵第perm[i]个轴
        	p_v_h = tf.nn.sigmoid(a)  #从h->v的输入经函数激活后的概率值
        	v_ = self._bernouille_sampling(p_v_h, shape = [self.FLAGS.batch_size, int(p_v_h.shape[-1])])

        	return p_v_h, v_

    def optimize(self,v):
    	#Gibbs sampling and gradients calculation and update parameters

    	with tf.name_scope('optimization'):
    		v0,vk,ph0,phk, _ = self._gibbs_sampling(v)
    		dW , db_h, db_v = self._compute_gradients(v0,vk,ph0,phk)  #目标函数对这三个学习参数的导数
    		update_op = self._update_parameter(dW,db_h,db_v)

    	with tf.name_scope('accuracy'):		#计算迭代一次更新参数后新模型的准确率 用来决定是否收敛
    		mask = tf.where(tf.less(v0,0.0), x = tf.zeros_like(v0), y = tf.ones_like(v0))
    		bool_mask = tf.cast(tf.where(tf.less(v0,0.0), x = tf.zeros_like(v0), y = tf.ones_like(v0)), dtype = tf.bool)
    		acc = tf.where(bool_mask, x = tf.abs(tf.subtract(v0,vk)), y = tf.zeros_like(v0))
    		n_values = tf.reduct_sum(mask)
    		acc = tf.subtract(1.0, tf.div(tf.reduce_sum(acc),n_values))

    	return update_op, acc

    def inference(self,v):  #预测   从v -> h' -> v' 进行一次预测  返回预测结果

    	#正在训练的样本被用来激活潜在节点计算得p_h_v  然后p_v_h是作为预测 给评过分/未评过分的电影

    	p_h_v = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(v,self.W),self.bh))  #计算从可见层到隐藏层的输入再经激活  p_h_v.shape = [K,num_h]
    	h_ = self._bernouille_sampling(p_h_v,shape = [1,int(p_h_v.shape[-1])])

    	p_v_h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(h_, tf.transpose(self.W, [1,0])), self.bv))
    	v_ = self._bernouille_sampling(p_v_h, shape = [1,int(p_v_h.shape[-1])])  #一次反向预测

    	return v_

    def _update_parameter(self,dW,db_h,db_v):   #dx表示x变量的变化量
    	alpha = self.FLAGS.learning_rate

    	update_op = [tf.assign(self.W, alpha * tf.add(self.W,dW)),
    				 tf.assign(self.bh, alpha * tf.add(self.bh,db_h)),
    				 tf.assign(self.bv, alpha * tf.add(self.bv,db_v))]

    	return update_op

   	def _compute_gradients(self,v0,vk,ph0,phk):  #四个参数由gibbs sampling而得

   		def condition(i,v0,vk,ph0,phk,dW,db_h,db_v):  #用作循环判断条件  只循环k次
   			r = tf.less(i,k)  #i与k对比大小结果的Bool标量（i,k长度为1）
   			return r[0]

   		def body(i,v0,vk,ph0,phk,dW,dbh,dbv):
   			#第i次循环就更新第i个参数 总共更新Batch_size次
   			v0_ = v0[i]
   			ph0_ = ph0[i] 

   			vk = vk[i] 
   			phk_ = phk[i]

   			#将它们进行维度变换 适合计算
   			ph0_ = tf.reshape(ph0_,[1,self.FLAGS.num_h])
   			v0_ = tf.reshape(v0_,[self.FLAGS.num_v,1])
   			phk_ = tf.reshape(phk_,[1,self.FLAGS.num_h])
   			vk_ = tf.reshape(vk_,[self.FLAGS.num_v,1])
   			#按照公式计算梯度
   			dw_ = tf.subtract(tf.multiply(ph0_,v0_),tf.multiply(phk_,vk_))
   			dbh_ = tf.subtract(ph0_,phk_) #返回相减后的张量
   			dbv_ = tf.subtract(v0_, vk_)

   			dbh_ = tf.reshape(dbh_,[self.FLAGS.num_h])
   			dbv_ = tf.reshape(dbv_,[self.FLAGS.num_v])

   			return [i+1,v0,vk,ph0,phk,tf.add(dW,dw_),tf.add(dbh,dbh_),tf.add(dbv,dbv_)]  #循环中返回的数值会作为下次循环该函数的参数

   		#初始化优化循环的参数
   		i = 0  #counter for loop body
   		k = tf.constant([self.FLAGS.batch_size]) #constant([])创建常量列表 

   		dW = tf.zeros((self.FLAGS.num_v,self.FLAGS.num_h))  # num_v * num_h 的权重矩阵
   		dbh = tf.zeros((self.FLAGS.num_h))	
   		dbv = tf.zeros((self.FLAGS.num_v))

   		[i,v0,vk,ph0,phk,dW,db_h,db_v] = tf.while_loop(condition,body,[i,v0,vk,ph0,phk,dW,dbh,dbv])

   		dW = tf.div(dW,self.FLAGS.batch_size)  #div()除法 求得平均值
   		dbh = tf.div(dbh, self.FLAGS.batch_size)
   		dbv = tf.div(dbv, self.FLAGS.batch_size)

   		return dW, dbh,dbv


   	def _gibbs_sampling(self,v):

   		def condition(i,vk,hk,v):
   			r = tf.less(i,k)
   			return r[0]

   		def body(i,vk,hk,v):
   			_,hk = self._sample_h(vk)
   			_,vk = self._sample_v(hk)

   			vk = tf.where(tf.less(v,0),v,vk)

   			return [i+1,vk,hk,v]

   		ph0,_ = self.sample_h(v)

   		vk = v 
   		hk = tf.zeros_like(ph0)

   		i = 0
   		k = tf.constant([self.FLAGS.k])
   		[i,vk,hk,v] = tf.while_loop(condition,body,[i,vk,hk,v])

   		phk, _ = self._sample_h(vk)

   		return v,vk,ph0,phk,i

   	def _bernouille_sampling(self,p,shape):  #伯努利采样，即阈值接受-拒绝采样
   		return tf.where(tf.less(p,tf.random_uniform(shape,minval = 0.0, maxval = 1.0)),
   						x = tf.zeros_like(p), y = tf.ones_like(p))





























