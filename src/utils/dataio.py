#-*- coding:utf-8 -*-
import sys
sys.path.append('../..')

import csv, sys
import numpy as np
import pandas as pd
import os.path as op
from scipy.sparse import lil_matrix
import math
import cPickle

class RecDataIO(object):
	"""docstring for RecDataIO"""
	def __init__(self, arg=""):
		super(RecDataIO, self).__init__()
		self.arg = arg
		
	def data_filter(self, datapath, sep="\t", userthreshold=5, itemthreshold=0):
		"""
			filter out user and items whoes number of neighbors is less threshold
		"""
		base_dir = op.dirname(datapath)
		filename = op.basename(datapath)

		data = pd.read_csv(datapath, sep=sep, header = None, usecols=[0,1,3], dtype={3:np.float64})
		data.columns = ['user', 'item', 'timestamp']

		# print data.describe()
		# data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp())

		users = data.groupby("user").size()
		items = data.groupby("item").size()
		
		# data = data[(np.in1d(data.user, users[users>userthreshold].index)) & (np.in1d(data.item, items[items > itemthreshold].index))]
		filed_users = np.in1d(data.user, users[users>userthreshold].index)
		print sum(filed_users)
		
		print "Begin to filter......"
		data = data[np.in1d(data.user, users[users>userthreshold].index)]
		print "Finish filtering"

		data = data.sort_values(by=['user', 'timestamp'], ascending=True)

		return data

	def getUserItem(self,data):
		"""
			get user and item id mapping
		"""
		users = dict([(v,index) for index,v in enumerate(data.user.unique())])
		items = dict([(v,index) for index,v in enumerate(data.item.unique())])

		return users, items

	def getUserItemCount(self,data):
		return data.user.nunique(), data.item.nunique()

	def leaveOneOut(self,data):
		"""
			data: a sorted pandas data frame
		"""
		data_values = data.values

		self.group_stat(data)

		user_id_mapper, item_id_mapper = self.getUserItem(data)
		userCount, itemCount = self.getUserItemCount(data)
		trainMatrix = lil_matrix((userCount, itemCount), dtype = np.float64)

		trainRating = [list() for u in range(userCount)]
		testRating = [0 for u in range(userCount)]

		for user, item, timestamp in data_values:
			map_user = user_id_mapper[user]
			map_item = item_id_mapper[item]

			trainRating[map_user].append(map_item)
			trainMatrix[map_user,map_item] = timestamp

		for u in range(userCount):
			testRating[u] = [trainRating[u].pop()]
			iid = testRating[u][0]

			# trainMatrix[u,iid] = 0

		# filter out repeated ratings
		# for uid, iids in enumerate(trainRating):
		# 	l = list()
		# 	for iid in iids:
		# 		if trainMatrix[uid, iid] != 0:
		# 			l.append(iid)

		# 	trainRating[uid] = l

		return trainMatrix, trainRating, testRating

	def group_stat(self, data):
		users = data.groupby("user").size()
		max_rat, min_rat = max(users), min(users)

		print "Max {0}, Min {1} Ratings".format(max_rat, min_rat)


	def fold_out(self, data, fold_percent = 0.2):
		data_values = data.values
		user_dict = dict()

		user_id_mapper, item_id_mapper = self.getUserItem(data)
		userCount, itemCount = self.getUserItemCount(data)

		trainRating = [list() for u in range(userCount)]
		testRating = [0 for u in range(userCount)]

		trainMatrix = lil_matrix((userCount, itemCount), dtype = np.float64)

		for user, item, timestamp in data_values:
			map_user = user_id_mapper[user]
			map_item = item_id_mapper[item]

			trainRating[map_user].append(map_item)
			trainMatrix[map_user,map_item] = timestamp

		for u in range(userCount):
			t_size = int(math.ceil(len(trainRating[u]) * fold_percent))

			testRating[u] = trainRating[u][-t_size:]
			trainRating[u] = trainRating[u][:-t_size]

			# for iid in testRating[u]:
			# 	trainMatrix[u,iid] = 0

		# filter out repeated ratings
		# for uid, iids in enumerate(trainRating):
		# 	l = list()
		# 	for iid in iids:
		# 		if trainMatrix[uid, iid] != 0:
		# 			l.append(iid)

		# 	trainRating[uid] = l

		return trainMatrix, trainRating, testRating

	def fout_time(self, data, fold_percent = 0.2):
		data_values = data.values
		user_dict = dict()

		user_id_mapper, item_id_mapper = self.getUserItem(data)
		userCount, itemCount = self.getUserItemCount(data)

		trainRating = [list() for u in range(userCount)]
		testRating = [0 for u in range(userCount)]

		trainMatrix = lil_matrix((userCount, itemCount), dtype = np.float64)

		for user, item, timestamp in data_values:
			map_user = user_id_mapper[user]
			map_item = item_id_mapper[item]

			trainRating[map_user].append(map_item)
			trainMatrix[map_user,map_item] = timestamp

		for u in range(userCount):
			t_size = int(math.ceil(len(trainRating[u]) * fold_percent))

			testRating[u] = trainRating[u][-t_size:]
			trainRating[u] = trainRating[u][:-t_size]

		return trainMatrix, trainRating, testRating

def toSaveFoldOut(dataIo, filePath, dataName):
	data = dataIo.data_filter(filePath)

	user_id_mapper, item_id_mapper = dataIo.getUserItem(data)

	user_index_to_id = dict([(user_id_mapper[uid], uid) for uid in user_id_mapper])
	item_index_to_id = dict([(item_id_mapper[iid], iid) for iid in item_id_mapper])

	if dataName == "yelp": baseDir = "../../data/yelp-fold-80p/"
	elif dataName == "amovie": baseDir = "../../data/amovie-fold-80p/"
	elif dataName == "cds": baseDir = "../../data/cds-fold-80p/"

	cPickle.dump(user_id_mapper, open(baseDir + "user_id_to_index.cpk", "wb"))
	cPickle.dump(item_id_mapper, open(baseDir + "item_id_to_index.cpk", "wb"))
	cPickle.dump(user_index_to_id, open(baseDir + "user_index_to_id.cpk", "wb"))
	cPickle.dump(item_index_to_id, open(baseDir + "item_index_to_id.cpk", "wb"))

	trainMatrix, trainRating, testRating = dataIo.fold_out(data, 0.2)

	cPickle.dump(trainMatrix, open(baseDir + "trainMatrix.cpk", "wb"))
	cPickle.dump(trainRating, open(baseDir + "trainRating.cpk", "wb"))
	cPickle.dump(testRating, open(baseDir + "testRating.cpk", "wb"))


if __name__ == "__main__":
	dataIo = RecDataIO()
	toSaveFoldOut(dataIo, "../../data/amazon_cds_vinyl.rating", "cds")

	# data = dataIo.data_filter("../../data/yelp.rating")

	# trainMatrix, trainRating, testRating = dataIo.leaveOneOut(data)

	# trainMatrix, trainRating, testRating = dataIo.fold_out(data, fold_percent = 0.2)

	# userCount, itemCount = trainMatrix.shape

	# train = list()
	# test = list()
	# for u in range(userCount):
	# 	for i in trainRating[u]:
	# 		train.append([u,i,1])
	
	# for u in range(userCount):
	# 	iids = testRating[u]
	# 	for i in iids:
	# 		test.append([u,i,1])

	# train = pd.DataFrame(train)
	# test = pd.DataFrame(test)

	# train.to_csv("../baselines/caser_pytorch/datasets/acds/train_50p.txt", sep="\t", index=None, header=None)
	# test.to_csv("../baselines/caser_pytorch/datasets/acds/test_50p.txt", sep="\t", index=None, header=None)



