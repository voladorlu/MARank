#-*- coding:utf-8 -*-
import random
import string, sys
import numpy, math
from scipy import stats

def mf_fold_out_eval(triplets, user_lats, item_lats, userItems, ks, itemCount):
	"""
		function one: get performance on different lengths of list
		function two: get performance of different group of users, predifined 5 groups
	"""
	topKs = [int(v) for v in ks.split("+")]
	maxK = max(topKs)

	hrs = [list() for i in range(len(topKs))]
	pres = [list() for i in range(len(topKs))]
	ndcgs = [list() for i in range(len(topKs))]
	mrrs = [list() for i in range(len(topKs))]

	auc = list()

	groups_hr = [list() for i in range(5)]
	groups_pre = [list() for i in range(5)]
	groups_ndcg = [list() for i in range(5)]
	groups_mrr = [list() for i in range(5)]

	items = numpy.array(range(itemCount))

	account = 0
	auc_single = 0.0

	for user_id, uitems in enumerate(triplets):
		score_list = list()
		pre_items = list()

		user_row = userItems[user_id]
		cols = user_row.keys()
		pre_items = numpy.setdiff1d(items, cols)

		score_list = 0
		user_lat  = user_lats[user_id]
		score_list = numpy.dot(user_lat, item_lats.T)

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
				mrrs[index].append(0.0)
				continue

			ranks = numpy.where(c_index)[0]
			ranks = intersect_rank_index[ranks]
			hr = c_size / u_size
			pre = c_size / float(k)
			ndcg = sum(1.0/numpy.log2(ranks + 2)) / idcg(k, u_size)
			mrr = sum(1.0/(ranks + 1))

			hrs[index].append(hr)
			pres[index].append(pre)
			ndcgs[index].append(ndcg)
			mrrs[index].append(mrr)

		u_train_size = len(cols)
		group_id = user_group(u_train_size)

		c_index = intersect_rank_index < maxK
		c_size = sum(c_index)
		if c_size == 0:
			groups_hr[group_id].append(0)
			groups_pre[group_id].append(0)
			groups_ndcg[group_id].append(0)
			groups_mrr[group_id].append(0)
			continue

		ranks = numpy.where(c_index)[0]
		ranks = intersect_rank_index[ranks]
		hr = c_size / u_size
		pre = c_size / float(maxK)

		ndcg = sum(1.0/numpy.log2(ranks + 2)) / idcg(k, u_size)
		mrr = sum(1.0/(ranks + 1))

		groups_hr[group_id].append(hr)
		groups_pre[group_id].append(pre)
		groups_ndcg[group_id].append(ndcg)
		groups_mrr[group_id].append(mrr)

		# calculate AUC
		
		for iid in uitems:
			while True:
				neg_j = random.randint(0, itemCount - 1)
				if neg_j not in user_row and neg_j != iid:
					break
			pui = numpy.dot(user_lat, item_lats[iid])
			pui = float("%.4f"%pui)
			puj = numpy.dot(user_lat, item_lats[neg_j])
			puj = float("%.4f"%puj)

			if pui > puj: auc_single += 1.0
			elif pui == puj: auc_single += 0.5

		account += len(uitems)

	auc_single /= account

	for index, k in enumerate(topKs):
		hrs[index] = float("%.4f"%numpy.mean(hrs[index]))
		pres[index] = float("%.4f"%numpy.mean(pres[index]))
		ndcgs[index] = float("%.4f"%numpy.mean(ndcgs[index]))
		mrrs[index] = float("%.4f"%numpy.mean(mrrs[index]))

	for group_id in range(len(groups_hr)):

		grp_num = len(groups_hr[group_id])
		if grp_num == 0:
			groups_hr[group_id] = 0.0
			groups_pre[group_id] = 0.0
			groups_ndcg[group_id] = 0.0
			groups_mrr[group_id] = 0.0
		else:
			groups_hr[group_id] = float("%.4f"%numpy.mean(groups_hr[group_id]))
			groups_pre[group_id] = float("%.4f"%numpy.mean(groups_pre[group_id]))
			groups_ndcg[group_id] = float("%.4f"%numpy.mean(groups_ndcg[group_id]))
			groups_mrr[group_id] = float("%.4f"%numpy.mean(groups_mrr[group_id]))

	return hrs,pres,ndcgs,mrrs,groups_hr,groups_pre,groups_ndcg,groups_mrr, auc_single


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

def t_test(x,y):
	"""
		x,y are samples

		return p-value
	"""
	t2, p2 = stats.ttest_ind(a,b)

	return 2*p2
