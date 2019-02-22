#-*- coding:utf-8 -*-

import tensorflow as tf
import random
import string
import re, argparse
import numpy, time, sys, math, os
from utils.dataio import RecDataIO

# A Highway Hierachical Attentive Network For Sequential Recommendatioin
# Author: Lu Yu
# Date: Dec. 24th 2017

# Turn off the debugging information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_args():
	"""
		get command-line args from terminal
	"""
	args = argparse.ArgumentParser(description="argument parser")

	args.add_argument("-m", type=str, default="ANRank", help="model name")

	args.add_argument("-lr", type=float, default=0.001)
	args.add_argument("-reg", type=float, default=0.0005)
	args.add_argument("-margin", type=float, default=4.0)
	args.add_argument("-sm", type=int, default=1000, help="maximum sampling steps")
	args.add_argument("-mi", type=int, default=150, help="maximum iteration numbers")
	args.add_argument("-ei", type=int, default=5, help="show progress every ei iterations")
	args.add_argument("-sp", type=bool, default=True, help="switch to show learning progress")
	args.add_argument("-k", type=str, default="3+5+10+15+20", help="top-k recommendation")
	args.add_argument("-data", type=str, default="data/ml100k.rating", help="data path")
	args.add_argument("-out", type=str, default="explogs/han/yelp_han_att_logs.txt")
	args.add_argument("-ml", type=int, default=6, help="maximum depend length")
	args.add_argument("-gl", type=int, default=6, help="maximum depend length")

	args.add_argument("-model", type=str, default="att")
	args.add_argument("-us", type=int, default=5, help="user threshold")
	args.add_argument("-sep", type=str, default="tab", help="tab, space")
	args.add_argument("-decay", type=float, default=0.25, help="skipgram weight decay")
	args.add_argument("-tp", type=float, default=0.2, help="train & test split")
	args.add_argument("-lf", type=str, default="l", help="l:leave-one-out or f:fold-out")
	args.add_argument("-ew", type=float, default=0.4, help="exp weight")

	# params for neural network
	args.add_argument("-n_h", type=int, default=64, help="hidden status dimension")
	args.add_argument("-gpu", dest="gpu", action="store_true", help="indicator to use gpu or not")
	args.set_defaults(gpu = False)

	args.add_argument("-exp", dest="exp", action="store_true", help="indicator to use gpu or not")
	args.set_defaults(exp = False)

	args.add_argument("-dn", type=int, default=0, help="device number")

	args.add_argument("-bs", type=int, default=1024, help="batch size")
	args.add_argument("-grad_clip", type=float, default=5.0)
	args.add_argument("-dropout", type=float, default=0.5)
	args.add_argument("-layers", type=int, default=2, help="number of linear neural layers")
	args.add_argument("-att_layers", type=int, default=2, help="number of attention layers")
	args.add_argument("-att_act", type=str, default="tanh",
					  help="available choices, relu, tanh")

	# activation function for residual neural layers
	args.add_argument("-in_act", type=str, default="relu",
					  help="available choices, sigmoid, tanh, relu")
	args.add_argument("-out_act", type=str, default="relu",
					  help="available choices, relu, identity, tanh")

	args.add_argument("-dyn", dest='dyn', action='store_true', help="dynamic or uniform sampling")
	args.set_defaults(dyn = False)

	args.add_argument("-prit",dest='prit', action='store_true')
	args.set_defaults(prit=False)

	args.add_argument("-beta1", type=float, default=0.9)
	args.add_argument("-beta2", type=float, default=0.99)
	args.add_argument("-epsilon", type=float, default=1e-4)

	args.add_argument("-opt", type=str, default="adam")

	return args.parse_args()

def leaveOneOut(data_path, threshold, sep):
	dataIo = RecDataIO()
	data = dataIo.data_filter(data_path, userthreshold=threshold, sep = sep)

	return dataIo.leaveOneOut(data)

def triplet_data(trainRating, testRating, maxLens, winSize):

	triplets = list()
	contexts = list()
	seqs = list()
	triplets_test = list()
	contexts_test = list()
	seqs_test = list()

	for u, urates in enumerate(trainRating):
		for i, item in enumerate(urates):
			if i == 0: continue

			uid, iid = u, item
			neg_iid = 0

			content_data = list()
			for cindex in range(max([0, i - maxLens]), i):
				cid = urates[cindex]

				content_data.append(cid)
			
			if(len(content_data)<maxLens):
				content_data=content_data+[content_data[-1] for i in range(maxLens-len(content_data))]

			triplets.append([uid,iid,neg_iid])
			contexts.append(content_data)
			

	for uid, iid in enumerate(testRating):
		content_data = list()
		urates = trainRating[uid]
		for cindex in range(max([0, len(urates) - maxLens]), len(urates)):
			content_data.append(urates[cindex])
		try:
			if(len(content_data)<maxLens):
				content_data = content_data+[content_data[-1] for i in range(maxLens-len(content_data))]
		except Exception,e:
			print uid, len(urates)
			print content_data
			sys.exit(0)

		triplets_test.append([uid,iid, 0])
		contexts_test.append(content_data)

	return triplets,contexts,triplets_test,contexts_test

def new_evaluate(triplets_test, user_latent, gru_latent, item_embs, item_lats):
	topK=args.k
	hr = list()
	ndcg = list()
	mapn = list()
	init_time = time.time()

	items = numpy.array(range(itemCount))

	for i in range(len(triplets_test)):
		score_list = list()
		pre_items = list()

		user_id = int(triplets_test[i][0])
		item_id = int(triplets_test[i][1])

		user_row = trainMatrix[user_id]
		cols = user_row.keys()
		pre_items = numpy.setdiff1d(items, cols)
		# pre_items = items

		score_list = 0
		# item_lats = item_latent[pre_items]
		user_lat  = user_latent[i]
		score_list += numpy.dot(user_lat, item_lats.T)
		score_list += numpy.dot(gru_latent[i], item_embs.T)

		score_list = score_list[pre_items]

		# score_list = numpy.array(score_list)
		rank = numpy.argpartition(score_list, -topK)[-topK:]
		rank = rank[numpy.argsort(score_list[rank])]
		
		rank_index_list = dict([ (pre_items[ttid],index) for index, ttid in enumerate(list(rank)[::-1])])
		
		if item_id in rank_index_list:
			index = rank_index_list[item_id]
			hr.append(1.0)
			ndcg.append(1.0 / math.log(index + 2,2))
			mapn.append(1.0 / (index+1))
		else:
			hr.append(0.0)
			ndcg.append(0.0)
			mapn.append(0.0)

	return float("%.4f"%numpy.mean(hr)), float("%.4f"%numpy.mean(ndcg)), float("%.4f"%numpy.mean(mapn))

def fold_out(data_path, threshold, sep, fold_percent=0.2):
	dataIo = RecDataIO()
	data = dataIo.data_filter(data_path, userthreshold=threshold, sep = sep)

	return dataIo.fold_out(data, fold_percent)

def fold_triplet_data(trainRating, testRating, maxLens):

	train_triplets = list()
	train_contents = list()

	weights = list()

	test_triplets = list()
	test_contents = list()

	for u, urates in enumerate(trainRating):
		for i, item in enumerate(urates):
			if i == 0: continue

			uid, iid = u, item
			neg_iid = 0

			content_data = list()
			for cindex in range(max([0, i - maxLens]), i):
				content_data.append(urates[cindex])
			
			if(len(content_data) < maxLens):
				content_data=content_data+[content_data[-1] for i in range(maxLens-len(content_data))]

			train_triplets.append([uid,iid,neg_iid])
			train_contents.append(content_data)
			weights.append(1.0)
			
	for uid, iid in enumerate(testRating):
		content_data = list()
		urates = trainRating[uid]
		for cindex in range(max([0, len(urates) - maxLens]), len(urates)):
			content_data.append(urates[cindex])
		try:
			if(len(content_data) < maxLens):
				content_data = content_data+[content_data[-1] for i in range(maxLens-len(content_data))]
		except Exception,e:
			print uid, len(urates)
			print content_data
			sys.exit(0)

		test_triplets.append(iid)
		test_contents.append(content_data)

	return [train_triplets, train_contents, weights], [test_triplets, test_contents]

def fold_out_eval(triplets_test, user_lats, item_lats, item_bias=0):
	"""
		function one: get performance on different lengths of list
		function two: get performance of different group of users, predifined 5 groups
	"""
	topKs = [int(v) for v in args.k.split("+")]
	maxK = max(topKs)

	hrs = [list() for i in range(len(topKs))]
	pres = [list() for i in range(len(topKs))]
	ndcgs = [list() for i in range(len(topKs))]
	mapns = [list() for i in range(len(topKs))]

	groups_hr = [list() for i in range(5)]
	groups_pre = [list() for i in range(5)]
	groups_ndcg = [list() for i in range(5)]
	groups_mapn = [list() for i in range(5)]

	init_time = time.time()

	items = numpy.array(range(itemCount))
	
	for user_id, uitems in enumerate(triplets_test):
		score_list = list()
		pre_items = list()

		user_row = trainMatrix[user_id]
		cols = user_row.keys()
		pre_items = numpy.setdiff1d(items, cols)
		# pre_items = items

		score_list = 0
		user_lat  = user_lats[user_id]
		score_list += numpy.dot(user_lat, item_lats.T)
		# score_list += numpy.dot(gru_lats[user_id], item_embs.T)

		score_list = score_list[pre_items]

		rank = numpy.argpartition(score_list, -maxK)[-maxK:]
		rank = rank[numpy.argsort(score_list[rank])]

		rank_index_list = [pre_items[ttid] for ttid in list(rank)[::-1]]

		intersect_rank_index = numpy.where(numpy.in1d(rank_index_list, uitems))[0]
		
		u_size = float(len(uitems))

		for index, k in enumerate(topKs):
			c_index = intersect_rank_index < k
			c_size = sum(c_index)
			if c_size == 0:
				hrs[index].append(0.0)
				pres[index].append(0.0)
				ndcgs[index].append(0.0)
				mapns[index].append(0.0)
				continue

			ranks = numpy.where(c_index)[0]
			ranks = intersect_rank_index[ranks]
			hr = c_size / u_size
			pre = c_size / float(k)
			ndcg = sum(1.0/numpy.log2(ranks + 2)) / idcg(k, u_size)
			mapn = sum(1.0/(ranks + 1)) / (min(k, u_size))

			hrs[index].append(hr)
			pres[index].append(pre)
			ndcgs[index].append(ndcg)
			mapns[index].append(mapn)

		u_train_size = len(cols)
		group_id = user_group(u_train_size)

		c_index = intersect_rank_index < maxK
		c_size = sum(c_index)
		if c_size == 0:
			groups_hr[group_id].append(0)
			groups_pre[group_id].append(0)
			groups_ndcg[group_id].append(0)
			groups_mapn[group_id].append(0)
			continue

		ranks = numpy.where(c_index)[0]
		ranks = intersect_rank_index[ranks]
		hr = c_size / u_size
		pre = c_size / float(maxK)

		ndcg = sum(1.0/numpy.log2(ranks + 2)) / idcg(maxK, u_size)
		mapn = sum(1.0/(ranks + 1)) / (min(maxK, u_size))

		# ndcg = sum(1.0/numpy.log2(ranks + 2))
		# mapn = sum(1.0/(ranks + 1))

		groups_hr[group_id].append(hr)
		groups_pre[group_id].append(pre)
		groups_ndcg[group_id].append(ndcg)
		groups_mapn[group_id].append(mapn)

	for index, k in enumerate(topKs):
		hrs[index] = float("%.4f"%numpy.mean(hrs[index]))
		pres[index] = float("%.4f"%numpy.mean(pres[index]))
		ndcgs[index] = float("%.4f"%numpy.mean(ndcgs[index]))
		mapns[index] = float("%.4f"%numpy.mean(mapns[index]))

	for group_id in range(5):

		grp_num = len(groups_hr[group_id])
		if grp_num == 0:
			groups_hr[group_id] = 0.0
			groups_pre[group_id] = 0.0
			groups_ndcg[group_id] = 0.0
			groups_mapn[group_id] = 0.0
		else:
			groups_hr[group_id] = float("%.4f"%numpy.mean(groups_hr[group_id]))
			groups_pre[group_id] = float("%.4f"%numpy.mean(groups_pre[group_id]))
			groups_ndcg[group_id] = float("%.4f"%numpy.mean(groups_ndcg[group_id]))
			groups_mapn[group_id] = float("%.4f"%numpy.mean(groups_mapn[group_id]))

	return hrs,pres,ndcgs,mapns,groups_hr,groups_pre,groups_ndcg,groups_mapn

def idcg(mk, usize):
	k = min(mk, usize)
	idea_dcg = sum(1.0 / numpy.log2(numpy.arange(k) + 2))

	return idea_dcg

def user_group(degree=5):
	if degree < 10: 
		return 0
	elif degree >= 10 and degree < 20: 
		return 1
	elif degree >= 20 and degree < 50:
		return 2
	elif degree >= 50 and degree < 100:
		return 3
	elif degree >= 100:
		return 4

def group_seg(group_id):
	if group_id == 0:
		return "<10"
	elif group_id == 1:
		return "[10,20)"
	elif group_id == 2:
		return "[20,50)"
	elif group_id == 3:
		return "[50,100)"
	elif group_id == 4:
		return ">=100"

def posWeight(x):
	"""
		@param
			x: rank position
		@return
			w: 1 + 0.5*[ ceil(log(x+1,2)) - 1]
	"""
	half = math.ceil(math.log(x+1,2)) - 1

	return 1 + 0.5*half

def newDynSampler(batch, user_latent, gru_latent, item_embs, item_lats):

	weights = list()
	cnt = 0

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = trainMatrix[uid]
		
		xui = numpy.dot(user_latent[index], item_lats[pid])
		xui += numpy.dot(gru_latent[index], item_embs[pid])

		maxi_steps = args.sm
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			while True:
				neg_j = random.randint(0, itemCount-1)
				if neg_j not in uitems:
					break
			
			xuj = numpy.dot(user_latent[index], item_lats[neg_j])
			xuj += numpy.dot(gru_latent[index], item_embs[neg_j])

			xuji = xuj - xui + args.margin
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = neg_j

			if xuji > 0: flag = False

			maxi_steps -= 1

		rank = int((itemCount - 1)/(args.sm - maxi_steps))

		weights.append(numpy.float32(posWeight(rank)/posWeight(itemCount)))
		batch[index][-1] = selected_neg

		if posWeight(rank)/posWeight(itemCount) < 1: cnt += 1

	return batch, weights, cnt

def uniformSampler(batch):

	weights = list()
	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = trainMatrix[uid]

		neg_j = 0
		while True:
			neg_j = random.randint(0, itemCount - 1)
			if neg_j not in uitems:
				break

		batch[index][-1] = neg_j

	return batch

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = numpy.zeros(K)
	J = numpy.zeros(K, dtype=numpy.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
	    q[kk] = K*prob
	    if q[kk] < 1.0:
	        smaller.append(kk)
	    else:
	        larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
	    small = smaller.pop()
	    large = larger.pop()

	    J[small] = large
	    q[large] = q[large] + q[small] - 1.0
	    if q[large] < 1.0:
	        smaller.append(large)
	    else:
	        larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(numpy.floor(numpy.random.rand()*K))
	if numpy.random.rand() < q[kk]:
		return kk
	else:
	    return J[kk]


def itemPopDist(trainRating, itemCount):
	"""
		sort items in descending order according to the popularity
	"""

	item_deg = numpy.zeros(itemCount, dtype=numpy.int32)

	for u in range(len(trainRating)):
		for i in trainRating[u]:
			item_deg[i] += 1

	nitem_deg = numpy.power(item_deg, args.decay)
	item_deg = nitem_deg / numpy.sum(nitem_deg)

	return item_deg,nitem_deg

def stat_samp(batch, J, q, itemCount):
	for index, triplet in enumerate(batch):
		uid,pid,nid = triplet
		uitems = trainMatrix[uid]

		neg_j = 0
		while True:
			neg_j = alias_draw(J, q)
			if neg_j not in uitems:
				break

		batch[index][-1] = neg_j

	return batch

def using_tocoo_izip(x):
    cx = x.tocoo()
    train_ratings = list() 
    for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
        train_ratings.append((i,j,v))

    return train_ratings

def trainDict(trainRating):
	
	user_items = dict()

	for uid, iids in enumerate(trainRating):
		for iid in iids:
			user_items.setdefault(uid, dict())
			user_items[uid].setdefault(iid, None)

	return user_items

class MultiRel(object):
	""" Multiple relationship fusion to generate next-item prediction"""
	def __init__(self, args, userCount, itemCount):
		super(MultiRel, self).__init__()
		self.args = args
		self.n_linear_layers = args.layers
		self.n_hidden = args.n_h
		self.itemCount = itemCount
		self.userCount = userCount
		self.maxLens = args.ml
		self.gloMaxLens = args.gl
		self.att_layers = args.att_layers
		self.att_act_fnc = args.att_act
		self.in_act = args.in_act
		self.out_act = args.out_act
		self.grad_clip = args.grad_clip
		self.dropout = args.dropout
		self.opt = args.opt
		self.batch_1_size = args.bs

		self.out_path = args.out

		self.device_name = "cpu"
		self.device_name_emb = "cpu"
		self.device_number = args.dn
		self.dynSampler = args.dyn
		self.trainOrInf = True
		self.neu_l2 = 1.0 / self.n_linear_layers

		if args.gpu:
			self.device_name = "gpu"

		self._init_graph()

	def _init_graph(self,):
		"""
			initialize computation graph
		"""
		with tf.device('/{0}'.format(self.device_name)):
			# ===================== define model parameters ============================
			self.weights, self.biases, self.embeddings = self._init_weights()

			# ===================== placeholders =======================================
			self.triplets = tf.placeholder(tf.int32, [None, 3])
			self.ws      = tf.placeholder(tf.float32, [None])
			self.local_contexts = tf.placeholder(tf.int32, [None, self.maxLens])
			self.global_contexts = tf.placeholder(tf.int32, [None, self.gloMaxLens])
			
			# ===================== build model =====================================
			user_latent = tf.nn.embedding_lookup(self.embeddings["user_latent"], self.triplets[:,0])
			
			pi_latent = tf.nn.embedding_lookup(self.embeddings["item_latent"], self.triplets[:,1])
			ni_latent = tf.nn.embedding_lookup(self.embeddings["item_latent"], self.triplets[:,2])
			
			p_lat = tf.nn.embedding_lookup(self.embeddings["item_context"], self.triplets[:,1])
			n_lat = tf.nn.embedding_lookup(self.embeddings["item_context"], self.triplets[:,2])

		with tf.device('/{0}'.format(self.device_name)):
			iloc_embs = tf.nn.embedding_lookup(self.embeddings["item_context"], self.local_contexts)
			iloc_att_embs = self.itemContAtt(iloc_embs, user_latent)

			# ===================== forward multiplayer perceptron =====================================
			uhid_units = self.ures_nn(user_latent)
			ihid_units = self.ires_nn(iloc_embs)
			uihid_atts = self.multRelAtt(uhid_units, ihid_units)

			uihid_atts.append(tf.reshape(iloc_att_embs, [-1,1,self.n_hidden]))

			uihid_att_embs = tf.concat(uihid_atts, 1)

			uihid_agg_emb = self.hidAtt(uihid_att_embs)

			aggre_embs = self.res_nn(uihid_agg_emb)
			self.neu_embs =  aggre_embs + uihid_agg_emb
			
			# ===================== prediction difference =====================================
			# diff = tf.reduce_sum(tf.multiply(self.neu_embs, tf.subtract(self.pi_embs, self.ni_embs)), axis=1)
			
			self.diff1 = tf.reduce_sum(tf.multiply(self.neu_embs, tf.subtract(p_lat, n_lat)), axis=1)
			self.diff2 = tf.reduce_sum(tf.multiply(user_latent, tf.subtract(pi_latent, ni_latent)), axis=1)
			diff = tf.add(self.diff1, self.diff2)

			# ===================== embedding regularization =====================================
			self.L2_emb = tf.reduce_sum(tf.multiply(user_latent, user_latent), axis=1) \
			+ tf.reduce_sum(tf.multiply(pi_latent, pi_latent), axis=1) \
			+ tf.reduce_sum(tf.multiply(ni_latent, ni_latent), axis=1) \
			+ tf.reduce_sum(tf.multiply(p_lat,p_lat), axis=1) \
			+ tf.reduce_sum(tf.multiply(n_lat,n_lat), axis=1) \
			+ tf.reduce_sum(tf.multiply(uihid_agg_emb, uihid_agg_emb),axis=1) \
			+ tf.reduce_sum(tf.multiply(aggre_embs, aggre_embs), axis = 1)

			self.Loss_0 = - tf.log(tf.sigmoid(diff))

			self.Loss_0 = self.ws * tf.add(self.Loss_0, self.args.reg * self.L2_emb)

			self.error_emb = tf.reduce_sum(self.Loss_0)
			self.error_nn = self.error_emb/self.batch_1_size/self.maxLens
			
			self.nn_tv = self.weights.values() + self.biases.values()

			self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.error_nn, self.nn_tv), self.grad_clip)

			# ===================== optimizers =====================================
			if self.opt == "adam":
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1, beta2=self.args.beta2, epsilon=self.args.epsilon)
			elif self.opt == "adagrad":
				self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.args.lr, initial_accumulator_value=self.args.epsilon)
			elif self.opt == "grad":
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.args.lr)
			elif self.opt == "momentum":
				self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.args.lr, momentum=0.95)
			elif self.opt == "rmsprop":
				self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.args.lr,epsilon=self.args.epsilon)
			else:
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr, beta1=self.args.beta1, beta2=self.args.beta2, epsilon=self.args.epsilon)

			self.minimize_nn = self.optimizer.apply_gradients(zip(self.grads, self.nn_tv))

			self.emb_tv = self.embeddings.values()
			self.minimize_emb = self.optimizer.minimize(self.error_emb, var_list=self.emb_tv)

			# ===================== initialize model parameters ========================
			opt = tf.global_variables_initializer()
			config = tf.ConfigProto(inter_op_parallelism_threads=3,
                   				intra_op_parallelism_threads=3)
			config.gpu_options.allow_growth = True
			self.sess = tf.Session(config=config)
			self.sess.run(opt)

	def metric_dist(self, src, tar):
		dist_vec = src - tar
		return tf.reduce_sum(tf.multiply(dist_vec, dist_vec))

	def res_nn(self, aggre_embs):
		"""
			forward to residual neural network
			fully connected network as kernel
		"""
		
		out = aggre_embs
		pre = out

		# hid_units = [None for i in range(self.n_linear_layers)]

		for i in range(self.n_linear_layers):
			layer = "layer_%d"%i
			dropout_layer = tf.layers.dropout(out, rate=self.dropout, training=self.trainOrInf)
			out = tf.add(tf.matmul(dropout_layer,self.weights[layer]), self.biases[layer])
			out = tf.add(out, pre)

			if i < self.n_linear_layers - 1: # not last layer
				if self.in_act == "sigmoid": # sigmoid, tanh, relu
					out = tf.sigmoid(out)
				elif self.in_act == "tanh":
					out = tf.tanh(out)
				else:
					out = tf.nn.relu(out)
			else:
				if self.out_act == "relu": # relu, identity, tanh
					out = tf.nn.relu(out)
				elif self.out_act == "tanh":
					out = tf.tanh(out)
				else:
					out = out

			pre = out

			# hid_units[i] = tf.reshape(out, [-1,1,self.n_hidden])

		# hid_units = tf.concat(hid_units, 1)

		# out = self.hidAtt(hid_units)
		
		return out

	def ures_nn(self, aggre_embs):
		"""
			forward to residual neural network
			fully connected network as kernel
		"""
		
		out = aggre_embs
		

		hid_units = [None for i in range(self.att_layers)]

		for i in range(self.att_layers):
			layer = "ulayer_%d"%i
			pre = out

			dropout_layer = tf.layers.dropout(out, rate=self.dropout, training=self.trainOrInf)
			out = tf.add(tf.matmul(dropout_layer,self.weights[layer]), self.biases[layer])
			out = tf.add(out, pre)
			
			out = tf.nn.relu(out)
			
			hid_units[i] = out

		# hid_units = tf.concat(hid_units, 1)

		# out = self.hidAtt(hid_units)
		
		return hid_units

	def ires_nn(self, aggre_embs):
		"""
			forward to residual neural network
			fully connected network as kernel
		"""
		
		pre = aggre_embs
		hid_units = [None for i in range(self.att_layers)]

		for i in range(self.att_layers):
			layer = "ilayer_%d"%i
			pre = tf.reshape(pre, [-1,self.n_hidden])

			dropout_layer = tf.layers.dropout(pre, rate=self.dropout, training=self.trainOrInf)
			out = tf.add(tf.matmul(dropout_layer,self.weights[layer]), self.biases[layer])
			out = tf.add(out, pre)
			pre = tf.reshape(tf.nn.relu(out), [-1, self.maxLens, self.n_hidden])
			
			hid_units[i] = pre
		
		return hid_units

	def itemContAtt(self, item_embs, user_embs):
		"""
			classical soft attention used in neural language model
			input item_embs shape: batch, row, col
			input user_embs shape: batch, col

			desired output tensor shape: batch, col
		"""

		drop_item_embs = tf.layers.dropout(item_embs,rate=self.dropout, training=self.trainOrInf)
		wi = tf.matmul(tf.reshape(drop_item_embs,[-1,self.n_hidden]), self.weights["i/att/item/weight"])
		wi = tf.reshape(wi, [-1, self.maxLens, self.n_hidden])

		drop_user_embs = tf.layers.dropout(user_embs, rate=self.dropout, training=self.trainOrInf)
		wu = tf.matmul(drop_user_embs, self.weights["i/att/user/weight"])
		wu = tf.reshape(wu, [-1,1,self.n_hidden])

		if self.att_act_fnc == "relu":
			w = tf.nn.relu( tf.add(tf.add(wi,wu),self.biases["i/att/weight/bias"]) )
		elif self.att_act_fnc == "tanh":
			w = tf.tanh( tf.add(tf.add(wi,wu),self.biases["i/att/weight/bias"]) )
		else:
			w = tf.sigmoid( tf.add(tf.add(wi,wu),self.biases["i/att/weight/bias"]) )

		w = tf.reshape(w, [-1, self.n_hidden])
		outs = tf.sigmoid(tf.matmul(w, self.weights["i/att/out"]))
		outs = tf.reshape(outs, [-1, self.maxLens, 1])
		
		# tensor shape: batch,1,row
		outs = tf.contrib.layers.softmax(tf.transpose(outs, perm=[0,2,1]))
		outs = tf.matmul(outs,item_embs)		

		outs = tf.add(outs, tf.reshape(item_embs[:,-1,:], [-1,1,self.n_hidden]) )

		outs = tf.reshape(outs, [-1, self.n_hidden])
		
		return outs

	def multRelAtt(self, uhid_units, ihid_units):
		"""
			classical soft attention used in neural language model
			input item_embs shape: batch, row, col
			input user_embs shape: batch, col

			desired output tensor shape: batch, col
		"""
		hid_atts = [None for i in range(self.att_layers)]
		
		for i in range(self.att_layers):
			
			drop_item_embs = tf.layers.dropout(ihid_units[i],rate=self.dropout, training=self.trainOrInf)
			wi = tf.matmul(tf.reshape(drop_item_embs,[-1,self.n_hidden]), self.weights["hid/att/rel/item/weight_%d"%i])
			wi = tf.reshape(wi, [-1, self.maxLens, self.n_hidden])

			drop_user_embs = tf.layers.dropout(uhid_units[i], rate=self.dropout, training=self.trainOrInf)
			wu = tf.matmul(drop_user_embs, self.weights["hid/att/rel/user/weight_%d"%i])
			wu = tf.reshape(wu, [-1,1,self.n_hidden])

			if self.att_act_fnc == "relu":
				w = tf.nn.relu( tf.add(tf.add(wi,wu),self.biases["hid/att/rel/bias_%d"%i]) )
			elif self.att_act_fnc == "tanh":
				w = tf.tanh( tf.add(tf.add(wi,wu),self.biases["hid/att/rel/bias_%d"%i]) )
			else:
				w = tf.sigmoid( tf.add(tf.add(wi,wu),self.biases["hid/att/rel/bias_%d"%i]) )

			w = tf.reshape(w, [-1, self.n_hidden])
			outs = tf.sigmoid(tf.matmul(w, self.weights["hid/att/rel/out_%d"%i]))
			outs = tf.reshape(outs, [-1, self.maxLens, 1])
			
			# tensor shape: batch,1,row
			outs = tf.contrib.layers.softmax(tf.transpose(outs, perm=[0,2,1]))
			outs = tf.matmul(outs,ihid_units[i])

			outs = tf.reshape(outs, [-1, 1, self.n_hidden])

			hid_atts[i] = outs
		
		return hid_atts

	def hidAtt(self, hidden_embs):
		"""
			input hidden_embs: [batch, layers, dim]
		"""

		drop_hid_embs = tf.layers.dropout(hidden_embs, rate = self.dropout, training = self.trainOrInf)
		wh = tf.matmul(tf.reshape(drop_hid_embs, [-1, self.n_hidden]), self.weights["hid/att/weight"])
		wh = tf.reshape(wh, [-1, self.att_layers+1, self.n_hidden])

		if self.att_act_fnc == "relu":
			w = tf.nn.relu(tf.add(wh, self.biases["hid/att/weight/bias"]))
		elif self.att_act_fnc == "tanh":
			w = tf.tanh(tf.add(wh, self.biases["hid/att/weight/bias"]))
		else:
			w = tf.sigmoid(tf.add(wh, self.biases["hid/att/weight/bias"]))

		w = tf.reshape(w, [-1, self.n_hidden])

		outs = tf.sigmoid(tf.matmul(w, self.weights["hid/att/out"]))
		outs = tf.reshape(outs, [-1, self.att_layers+1, 1])
		
		# tensor shape: batch,1,row
		outs = tf.contrib.layers.softmax(tf.transpose(outs, perm=[0,2,1]))
		outs = tf.reshape(tf.matmul(outs,hidden_embs), [-1, self.n_hidden])

		return outs

	def _init_weights(self,):
		weights = dict()
		biases = dict()
		embeddings = dict()

		# ================= multilayer perceptron weight matrix ================
		for i in range(self.n_linear_layers):
			weights.setdefault("layer_%d"%i, tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
			biases.setdefault("layer_%d"%i, tf.Variable(tf.random_normal([self.n_hidden]), trainable=True))
		
		for i in range(self.att_layers):
			weights.setdefault("ulayer_%d"%i, tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
			biases.setdefault("ulayer_%d"%i, tf.Variable(tf.random_normal([self.n_hidden]), trainable=True))
			
			weights.setdefault("ilayer_%d"%i, tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
			biases.setdefault("ilayer_%d"%i, tf.Variable(tf.random_normal([self.n_hidden]), trainable=True))
			
			weights.setdefault("hid/att/rel/user/weight_%d"%i, tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
			weights.setdefault("hid/att/rel/item/weight_%d"%i, tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
			biases.setdefault("hid/att/rel/bias_%d"%i, tf.Variable(tf.random_normal([self.n_hidden]), trainable=True))
			weights.setdefault("hid/att/rel/out_%d"%i, tf.Variable(tf.random_normal([self.n_hidden,1]), trainable=True))


		weights.setdefault("i/att/user/weight", tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
		weights.setdefault("i/att/item/weight", tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
		biases.setdefault("i/att/weight/bias", tf.Variable(tf.random_normal([self.n_hidden]), trainable=True))
		weights.setdefault("i/att/out", tf.Variable(tf.random_normal([self.n_hidden,1]), trainable=True))

		weights.setdefault("hid/att/weight", tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
		biases.setdefault("hid/att/weight/bias", tf.Variable(tf.random_normal([self.n_hidden]), trainable=True))
		weights.setdefault("hid/att/out", tf.Variable(tf.random_normal([self.n_hidden,1]), trainable=True))

		# ================= embeddings ================
		user_latent = tf.Variable(tf.random_normal([self.userCount, self.n_hidden], mean=0.0, stddev=0.01), trainable=True)
		embeddings.setdefault("user_latent", user_latent)
		
		item_latent = tf.Variable(tf.random_normal([self.itemCount, self.n_hidden], mean=0.0, stddev=0.01), trainable=True)
		embeddings.setdefault("item_latent", item_latent)

		embeddings.setdefault("item_context", tf.Variable(tf.random_normal([self.itemCount, self.n_hidden], mean=0.0, stddev=0.01), trainable=True))
		
		return weights, biases, embeddings

	def train_model(self,train_data,test_data):

		if not self.args.prit:
			output_file = open(self.out_path, "w", 0)
			stdo = sys.stdout
			sys.stdout = output_file

		train_triplets, train_contents, train_weights = train_data

		print self.args
		print "========================== Model Analysis ===================\n"
		print "User:{0};Item:{1};Train:{2}".format(self.userCount, self.itemCount, len(train_triplets))

		triplets_test, contexts_test = test_data
		
		
		triplets = train_triplets
		contexts = train_contents
		weights = train_weights

		###### Continue to work #####
		for epo in range(self.args.mi):

			begin_time = time.time()
			cnt = 0

			train_samples = zip(triplets, contexts, weights)
			random.shuffle(train_samples)

			triplets, contexts, weights = zip(*train_samples)

			batch_num = int(len(triplets)/self.batch_1_size)
			for j in range(batch_num):

				triplets_batch = triplets[j*self.batch_1_size:(j+1)*self.batch_1_size]
				contexts_batch = contexts[j*self.batch_1_size:(j+1)*self.batch_1_size]
				weight_batch = weights[j*self.batch_1_size:(j+1)*self.batch_1_size]

				self.trainOrInf = False
				feed_dict = {self.triplets: triplets_batch,
							self.local_contexts: contexts_batch,
							self.ws: weight_batch}
				if args.exp:
					triplets_batch = stat_samp(triplets_batch, J, q, itemCount)
				else:
					triplets_batch = uniformSampler(triplets_batch)

				self.trainOrInf = True

				try:
					self.sess.run([self.minimize_emb, self.minimize_nn], feed_dict)
				except Exception, e:
					print len(triplets_batch), len(contexts_batch[0])
					print e
					sys.exit(0)

			triplets_batch = triplets[j*self.batch_1_size:(j+1)*self.batch_1_size]
			contexts_batch = contexts[j*self.batch_1_size:(j+1)*self.batch_1_size]
			weight_batch = weights[j*self.batch_1_size:(j+1)*self.batch_1_size]
			
			self.trainOrInf = False
			feed_dict = {self.triplets: triplets_batch,
						self.local_contexts: contexts_batch,
						self.ws: weight_batch}

			if args.exp:
				triplets_batch = stat_samp(triplets_batch, J, q, itemCount)
			else:
				triplets_batch = uniformSampler(triplets_batch)
				
			self.trainOrInf = True
			self.sess.run([self.minimize_emb, self.minimize_nn], feed_dict)

			iter_time = time.time() - begin_time

			if((epo+1)%args.ei == 0):
				self.trainOrInf = False
				begin_time = time.time()

				test_users = [(i,0,0) for i in range(userCount)]
				feed_dict = {self.triplets: test_users,
							self.local_contexts: contexts_test}

				emb_ext_time = time.time()
				user_lats, neu_embs = self.sess.run([self.embeddings["user_latent"], self.neu_embs], feed_dict)
				item_embs, item_lats  = self.sess.run([self.embeddings["item_context"], self.embeddings["item_latent"]])
				
				end_ext_time = time.time()

				user_lats = numpy.concatenate((user_lats, neu_embs), axis = 1)
				item_lats = numpy.concatenate((item_lats, item_embs), axis = 1)

				hrs,pres,ndcgs,mapns,ghr,gpre,gndcg,gmapn = fold_out_eval(triplets_test,user_lats,item_lats)

				topKs = [int(v) for v in args.k.split("+")]

				print "=================== Iter-%d: %.4fm, Eva:%.4fm, Ext:%.4fs ==================="%(epo+1,iter_time/60,(time.time() - begin_time)/60, end_ext_time - emb_ext_time)
				res_str = ""
				for index, k in enumerate(topKs):
					res_str += "HR:{0},P:{1},NDCG:{2},MAP:{3},-N:{4}- ".format(hrs[index],pres[index],ndcgs[index],mapns[index],k)
				print res_str

				group_res_str = ""
				for index in range(5):
					gstr = group_seg(index)
					group_res_str += "HR:{0},P:{1},NDCG:{2},MAP:{3}, {4};".format(ghr[index],gpre[index],gndcg[index],gmapn[index],gstr)
				print group_res_str

		self.sess.close()

		if not self.args.prit:
			sys.stdout = stdo
			output_file.close()

args = get_args()
cwd = os.getcwd()

if args.sep == "tab":
	sep = "\t"
elif args.sep == "space":
	sep = " "
else:
	sep = ","

# project_base = os.path.dirname(cwd)
data_path = os.path.join(cwd, args.data)

# get sparse training matrix
if args.lf == "l":
	trainMatrix, trainRating, testRating = leaveOneOut(data_path, args.us, sep)
else:
	trainMatrix, trainRating, testRating = fold_out(data_path, args.us, sep, args.tp)

userCount, itemCount = trainMatrix.shape

itemDist, nonItemDist = itemPopDist(trainRating, itemCount)
J, q = alias_setup(itemDist)

trainMatrix = trainDict(trainRating)

if __name__ == "__main__":

	model = MultiRel(args, userCount, itemCount)
	lists = fold_triplet_data(trainRating, testRating, args.ml)

	model.train_model(train_data = lists[0], test_data = lists[1])

