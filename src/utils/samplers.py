# -*- coding:utf-8 -*-
import random
import numpy, time, sys, math, os, random
# from datasketch import MinHashLSHForest, MinHash, MinHashLSH
import argparse


def mf_lfm_warp(batch, user_lats, item_lats, userItems, 
		itemCount, margin, num_negs, 
		itemRankWeight, itemRankNormalization, decay, deg_pos, deg_neg):

	weights = list()
	steps = list()
	cnt = 0

	accu_deg_imbalance = numpy.power(deg_pos/deg_neg, decay)

	print max(accu_deg_imbalance), min(accu_deg_imbalance)

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			while True:
				neg_j = random.randint(0, itemCount - 1)
				if neg_j not in uitems:
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[neg_j])

			xuji = xuj - xui + margin
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = neg_j

			if xuji > 0: flag = False

			maxi_steps -= 1

		# if maxi_steps == 0:
		# 	rank = 0
		# 	# weights.append(0)
		# 	# cnt += 1
		# else:
		# 	rank = int((itemCount - 1)/(num_negs - maxi_steps))

		rank = int((itemCount - 1)/(num_negs - maxi_steps))
		if flag:
			weights.append(0)
		else:
			weights.append(numpy.float32(itemRankWeight[rank]/itemRankNormalization))
	
		steps.append(num_negs - maxi_steps)

		if itemRankWeight[rank]/itemRankNormalization < 1: cnt += 1

		batch[index][-1] = selected_neg

		deg_pos[pid] += 1
		deg_neg[selected_neg] += 1

	return batch, weights, cnt, steps

def mf_lfm_exp(batch, user_lats, item_lats, userItems, 
		itemCount, rho, num_negs=20):

	weights = list()
	cnt = 0

	lamb = 1.0 / (rho * num_negs)

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		negs = list()

		maxi_steps = num_negs
		
		while maxi_steps > 0:

			while True:
				neg_j = random.randint(0, itemCount - 1)
				if neg_j not in uitems:
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[neg_j])

			negs.append((neg_j, xuj))

			maxi_steps -= 1

		negs.sort(key = lambda x: x[-1], reverse = True)

		neg_pos = min([len(negs) - 1, int(exp_real(lamb))])
		selected_neg = negs[neg_pos][0]

		weights.append(numpy.float32(1.0) )
		batch[index][-1] = selected_neg

	return batch, weights, cnt

def mf_dns(batch, user_lats, item_lats, userItems, 
		itemCount, num_negs=10):

	weights = list()
	cnt = 0

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		negs = list()

		maxi_steps = num_negs
		
		while maxi_steps > 0:

			while True:
				neg_j = random.randint(0, itemCount - 1)
				if neg_j not in uitems:
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[neg_j])

			negs.append((neg_j, xuj))

			maxi_steps -= 1

		negs.sort(key = lambda x: x[-1], reverse = True)

		selected_neg = negs[0][0]

		weights.append(numpy.float32(1.0) )
		batch[index][-1] = selected_neg

	return batch, weights, cnt

def mf_dsam(batch, user_lats, item_lats, userItems, 
		itemCount, margin, J, q, exp_wei_alpha, num_negs, base):

	weights = list()
	steps = list()
	cnt = 0

	K = len(J)

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			while True:
				neg_j = alias_draw(J, q, K)
				if neg_j not in uitems:
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[neg_j])

			xuji = xuj - xui + margin
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = neg_j

			if xuji > 0: flag = False

			maxi_steps -= 1

		rank = int(itemCount / (num_negs - maxi_steps))
		w = posWeight(rank) / posWeight(itemCount)
		weights.append(w)
		steps.append(num_negs - maxi_steps)

		batch[index][-1] = selected_neg

		if w < 1: cnt += 1

	return batch, weights, cnt, steps

def mf_dsam_mh(batch, user_lats, item_lats, userItems, 
		itemCount, margin, itemDist, exp_wei_alpha,
		num_negs, decay, deg_pos, deg_neg):

	weights = list()
	steps = list()
	cnt = 0

	accu_deg_imbalance = numpy.power(deg_pos/deg_neg, 0.5)

	print max(accu_deg_imbalance), min(accu_deg_imbalance)

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])
		deg_i = itemDist[pid]

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			maxi_deg = -float("inf")
			select_rej_neg = 0
			maxi_rej_steps = 10
			while True and maxi_rej_steps > 0:
				neg_j = random.randint(0, itemCount - 1)
				
				deg_j = itemDist[neg_j]
				# if deg_j == 0: continue

				if deg_j > maxi_deg:
					maxi_deg = deg_j
					select_rej_neg = neg_j

				accept_ratio_threshold = min([deg_j / deg_i, 1])
				accept_ratio = random.random()
				
				maxi_rej_steps -= 1

				if accept_ratio > accept_ratio_threshold:
					continue

				if neg_j not in uitems:
					select_rej_neg = neg_j
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[select_rej_neg])

			xuji = xuj - xui + margin
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = select_rej_neg

			if xuji > 0: flag = False

			maxi_steps -= 1


		rank = int(itemCount / (num_negs - maxi_steps))
		w = posWeight(rank) / posWeight(itemCount)
		weights.append(w)

		steps.append(num_negs - maxi_steps)

		batch[index][-1] = selected_neg

		deg_pos[pid] += 1
		deg_neg[selected_neg] += 1

		if w < 1: cnt += 1

	return batch, weights, cnt, steps

def mf_dyn(batch, user_lats, item_lats, userItems, itemCount, margin, num_negs):

	weights = list()
	steps = list()
	cnt = 0

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			while True:
				neg_j = random.randint(0, itemCount - 1)
				if neg_j not in uitems:
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[neg_j])

			xuji = xuj - xui + margin
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = neg_j

			if xuji > 0: flag = False

			maxi_steps -= 1

		rank = int((itemCount - 1)/(num_negs - maxi_steps))
		w = numpy.float32(posWeight(rank) / posWeight(itemCount))
		weights.append(w)
	
		if w < 1: cnt += 1

		batch[index][-1] = selected_neg

	return batch, weights, cnt

def mf_dcasam_mh(batch, user_lats, item_lats, userItems, 
		itemCount, margin, itemDist, exp_wei_alpha, 
		num_negs, decay, deg_pos, deg_neg, bal):

	weights = list()
	steps = list()
	cnt = 0

	accu_deg_imbalance = numpy.power(deg_pos/deg_neg, decay)

	print max(accu_deg_imbalance), min(accu_deg_imbalance)

	for index, triplet in enumerate(batch):
		
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])
		deg_i = itemDist[pid]
		adi_i = accu_deg_imbalance[pid]

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			maxi_deg = -float("inf")
			select_rej_neg = 0
			maxi_rej_steps = 10
			while True and maxi_rej_steps > 0:
				neg_j = random.randint(0, itemCount - 1)
				deg_j = itemDist[neg_j]
				adi_j = accu_deg_imbalance[neg_j]
				
				# if deg_j == 0: continue

				if adi_j > maxi_deg:
					maxi_deg = adi_j
					select_rej_neg = neg_j

				deg_prob = min([deg_j / deg_i, 1])
				accu_prob = min([adi_j / adi_i, 1])

				accept_ratio_threshold = bal * deg_prob + (1 - bal) * accu_prob

				accept_ratio = random.random()
				
				maxi_rej_steps -= 1

				if accept_ratio > accept_ratio_threshold:
					continue

				if neg_j not in uitems:
					select_rej_neg = neg_j
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[select_rej_neg])

			xuji = xuj - xui + margin
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = select_rej_neg

			if xuji > 0: flag = False

			maxi_steps -= 1

		rank = int(itemCount / (num_negs - maxi_steps))
		w = posWeight(rank) / posWeight(itemCount)
		weights.append(w)

		steps.append(num_negs - maxi_steps)

		batch[index][-1] = selected_neg

		deg_pos[pid] += 1
		deg_neg[selected_neg] += 1


		if w < 1: cnt += 1

	return batch, weights, cnt, steps

def mf_dgsam_mh(batch, user_lats, item_lats, userItems, 
		itemCount, margin, itemDist, exp_wei_alpha, 
		num_negs, decay, deg_pos, deg_neg, bal):

	weights = list()
	steps = list()
	cnt = 0

	prob = lambda x,sigma: math.exp(- pow(x,2) / (2 * pow(sigma,2)))

	accu_deg_imbalance = numpy.power(deg_pos/deg_neg, decay)

	print max(accu_deg_imbalance), min(accu_deg_imbalance)

	for index, triplet in enumerate(batch):
		
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])
		deg_i = itemDist[pid]
		adi_i = accu_deg_imbalance[pid]

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:

			maxi_deg = -float("inf")
			select_rej_neg = 0
			maxi_rej_steps = 10
			while True and maxi_rej_steps > 0:
				neg_j = random.randint(0, itemCount - 1)
				deg_j = itemDist[neg_j]
				adi_j = accu_deg_imbalance[neg_j]

				if adi_j > maxi_deg:
					maxi_deg = adi_j
					select_rej_neg = neg_j

				deg_prob = min([deg_j / deg_i, 1])
				accu_prob = min([adi_j / adi_i, 1])

				accept_ratio_threshold = bal * deg_prob + (1 - bal) * accu_prob

				accept_ratio = random.random()
				
				maxi_rej_steps -= 1

				if accept_ratio > accept_ratio_threshold:
					continue

				if neg_j not in uitems:
					select_rej_neg = neg_j
					break
			
			xuj = numpy.dot(user_lats[uid], item_lats[select_rej_neg])

			xuji = xuj - xui + margin

			accept_ratio_threshold = prob(xuji, 1.0)
			accept_ratio = random.random()

			if accept_ratio <= accept_ratio_threshold:
				selected_neg = select_rej_neg
				flag = False

			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = select_rej_neg

			# if xuji > 0: flag = False

			maxi_steps -= 1

		rank = int(itemCount / (num_negs - maxi_steps))
		w = posWeight(rank) / posWeight(itemCount)
		weights.append(w)

		steps.append(num_negs - maxi_steps)

		batch[index][-1] = selected_neg

		deg_pos[pid] += 1
		deg_neg[selected_neg] += 1


		if w < 1: cnt += 1

	return batch, weights, cnt, steps

def mf_vins(batch, user_lats, item_lats, userItems, 
		itemCount, margin, itemDist, itemOrders, exp_wei_alpha,
		num_negs, base, rho):

	weights = list()
	steps = list()
	cnt = 0
	
	lamb = 1.0 / (rho * itemCount)

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		xui = numpy.dot(user_lats[uid], item_lats[pid])

		maxi_steps = num_negs
		flag = True

		neg_j = 0
		maxi_value = -float("inf")
		selected_neg = 0
		while maxi_steps > 0 and flag:
			maxi_steps -= 1

			while True:
				neg_pos = min([itemCount - 1, int(exp_real(lamb))])
				neg_j = itemOrders[neg_pos]
				
				# deg_j = itemDist[neg_j]
				# if deg_j == 0: continue
				if neg_j not in uitems: break

			xuj = numpy.dot(user_lats[uid], item_lats[neg_j]) + margin
			xuji = xuj - xui

			# in case of running out of sampling times
			if xuji > maxi_value:
				maxi_value = xuji
				selected_neg = neg_j

			if xuji > 0: flag = False

		w = 0
		# if maxi_steps == 0:
		# 	weights.append(0)
		# else:
			# rank = int(base /(num_negs - maxi_steps))

		# rank = num_negs - maxi_steps
		# w = numpy.float32(posWeight(rank, exp_wei_alpha, base))

		rank = int(itemCount / (num_negs - maxi_steps))
		w = posWeight(rank) / posWeight(itemCount)
		weights.append(w)
		steps.append(num_negs - maxi_steps)

		batch[index][-1] = selected_neg

		if w < 1: cnt += 1

	return batch, weights, cnt, steps

def stat_exp(batch, itemOrders, userItems, rho, itemCount):
	"""
		@param
		 batch: [uid, pid, nid]
		 itemOrders: ordered item list by decreasing order of item popularity
		 rho: expected sampling position
		 itemCount: number of items
	"""
	lamb = 1.0 / (rho * itemCount)
	weights = list()

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = userItems[uid]

		while True:
			neg_pos = min([itemCount - 1, int(exp_real(lamb))])
			neg_j = itemOrders[neg_pos]

			if neg_j not in uitems: break

		batch[index][-1] = neg_j
		weights.append(1)

	return batch, weights

def stat_alias(batch, userItems, J, q, itemCount):
	k = 1.0 / (args.lamb * itemCount)

	weights = list()

	K = len(J)

	for index, triplet in enumerate(batch):
		uid,pid,nid = triplet

		uitems = userItems[uid]

		while True:
			nid = alias_draw(J, q, K)
			if nid not in uitems: break
		
		batch[index][-1] = nid
		weights.append(1)

	return batch, weights

def uniformSampler(batch, trainDict, itemCount, decay, deg_pos, deg_neg):

	weights = list()

	accu_deg_imbalance = numpy.power(deg_pos/deg_neg, decay)

	print max(accu_deg_imbalance), min(accu_deg_imbalance)

	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = trainDict[uid]

		neg_j = 0
		while True:
			neg_j = random.randint(0, itemCount - 1)
			if neg_j not in uitems:
				break

		batch[index][-1] = neg_j

		weights.append(1)

		deg_pos[pid] += 1
		deg_neg[neg_j] += 1

	return batch, weights

def popSampler(batch, trainDict, itemCount, J, q):

	weights = list()
	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = trainDict[uid]

		neg_j = 0
		while True:
			neg_j = alias_draw(J, q)
			if neg_j not in uitems:
				break

		batch[index][-1] = neg_j

		weights.append(1)

	return batch, weights

def popMHSampler(batch, trainDict, itemCount, itemDist):

	weights = list()
	for index, triplet in enumerate(batch):
		uid, pid, nid = triplet
		uitems = trainDict[uid]

		deg_i = itemDist[pid]
		neg_j = 0
		while True:
			neg_j = random.randint(0, itemCount - 1)
			deg_j = itemDist[neg_j]
			if deg_j == 0: continue

			accept_ratio_threshold = min([deg_j / deg_i, 1])
			accept_ratio = random.random()

			if accept_ratio > accept_ratio_threshold: continue

			if neg_j not in uitems: break

		batch[index][-1] = neg_j

		weights.append(1)

	return batch, weights

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

def alias_draw(J, q, K):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''

	kk = int(numpy.floor(numpy.random.rand()*K))
	if numpy.random.rand() < q[kk]:
		return kk
	else:
	    return J[kk]

def exp_real(lamb):
	"""
		f(x) = exp(- x*lambda)
		return a random x from exponential distribution with rate lamb
		- log(1 - uniform()) / lamb
	"""
	return -math.log(1 - random.random()) / lamb

# def posWeight(x, exp_wei_alpha, base):
# 	"""
# 		@param
# 			x: rank position
# 		@return
# 			w: 1 + 0.5*[ ceil(log(x+1,2)) - 1]
# 	"""
# 	# half = math.ceil(math.log(x+1,2)) - 1

# 	# half = math.ceil(math.log(x,2))

# 	# return 1 + exp_wei_alpha*half

# 	smooth = 1 / exp_wei_alpha
# 	w = 1 - math.log(x,2) / ( smooth + math.log(base, 2))

# 	return float("%.4f"%w)

def posWeight(x):
	"""
		@param
			x: rank position
		@return
			w: 1 + 0.5*[ ceil(log(x+1,2)) - 1]
	"""
	half = math.ceil(math.log(x+1,2)) - 1

	return 1 + 0.5*half

