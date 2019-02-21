#-*- coding:utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from dataio import RecDataIO
import argparse
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from pylab import *
import sys


stop_words = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]
def get_args():
	"""
		get command-line args from terminal
	"""
	args = argparse.ArgumentParser(description="argument parser")

	args.add_argument("-data", type=str, default="data/yelp.rating", help="data path")
	args.add_argument("-us", type=int, default=10, help="user records threshold")

	return args.parse_args()

def getWords(dp=""):
	line = open(dp, "r").readline()
	words = line.strip().split(" ")
	words = Counter(words)

	# remove stop words
	for sw in stop_words:
		if sw in words:
			del words[sw]

	# get words freqs
	degs = words.values()
	degs_freqs = Counter(degs).items()

	degs_freqs.sort()

	x,y = zip(*degs_freqs)

	x = np.log(x)
	y = np.log(y)

	def func(x, a, b):
	    return a*x + b

	params = curve_fit(func, x, y)

	[a, b] = params[0]

	print "slop -> %.6f, bias -> %.6f"%(a, b)

args = get_args()
# getWords(dp=args.data)

def itemDist():
	dio = RecDataIO()

	data = dio.data_filter(datapath=args.data, userthreshold=args.us)

	items = data.groupby("item").size().tolist()
	degs_freqs = Counter(items).items()
	degs_freqs.sort()

	x,y = zip(*degs_freqs)

	x = np.log(x)
	y = np.log(y)

	def func(x, a, b):
	    return a*x + b

	params = curve_fit(func, x, y)

	[a, b] = params[0]

	print "slop -> %.6f, bias -> %.6f"%(a, b)

# itemDist()

def est_bias_curve(N=2000):

	ratio = list()
	max_harm = 10000
	for r in range(1, N):
		h = np.arange(2, max_harm, dtype=np.float32)
		h = 1.0 / h

		base = 1.0 - float(r) / N

		expo = np.array([math.pow(base, k - 1) for k in range(2, max_harm)])
		
		ratio_value = np.dot(expo, h)

		ratio.append(ratio_value)

	t_index = list()
	text_pos = list()

	for i in range(len(ratio)):
		if ratio[i] > 5:
			continue
		else:
			t_index.append((i + 1, ratio[i]))
			
			break

	for i in range(len(ratio)):
		if ratio[i] > 3:
			continue
		else:
			t_index.append((i + 1, ratio[i]))
			# text_pos.append(i + 10)
			break

	for i in range(len(ratio)):
		if ratio[i] > 2:
			continue
		else:
			t_index.append((i + 1, ratio[i]))
			break
	

	for i in range(len(ratio)):
		if ratio[i] > 1:
			continue
		else:
			t_index.append((i + 1, ratio[i]))
			break

	for i in range(len(ratio)):
		if ratio[i] > 0.5:
			continue
		else:
			t_index.append((i + 1, 0.5))
			break

	for i in range(len(ratio)):
		if ratio[i] > 0.1:
			continue
		else:
			t_index.append((i + 1, 0.1))
			break

	# print "Break point: {0}".format(t_index)

	ratio_fig(range(1,N), ratio,0,t_index)

def ratio_fig(X, Y, count, break_points):
	fig = figure(figsize = (4.05, 3.45), facecolor = "white")
	ax = axes([0.2, 0.2 ,0.7, 0.7])

	symb = ['o', 'D', 'h', '8', 'p', '+', 's', '*', 'd', 'v']
	col = ['r', 'g', 'b', 'violet', 'm', 'y', 'k', 'w', 'orange', 'brown']
	lines  = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']

	plt.plot(X, Y, 
			# marker = symb[count],
			# markersize=0.5,
			# mfc=col[count],
			# markeredgewidth=0.1,
			# marker = "+",
			# markersize=1.0,
            linestyle = lines[count],
            linewidth=0.5,
            color="r"
            )

	# ax.axhline(y=1,xmin=0,xmax=(break_point/float(len(X))),c="r",linewidth=0.5,zorder=0)
	# ax.axvline(x=break_point,ymin=0,ymax=Y[break_point - 1],c="r",linewidth=0.5,zorder=0)

	for index, bp in enumerate(break_points):
		if index == len(break_points) - 1:
			ax.annotate('(%d,%.1f)'%(bp[0],bp[1]), 
				xy=bp, xytext=(bp[0]-200,bp[1] + 0.6),
				size=10,
            	arrowprops=dict(facecolor='black', arrowstyle="->"),
            )
		else:
			ax.annotate('(%d,%.1f)'%(bp[0],bp[1]), 
				xy=bp, xytext=(bp[0]+5,bp[1] + 0.8),
				size=10,
            	arrowprops=dict(facecolor='black', arrowstyle="->"),
            )

	# plt.legend(loc='best', fontsize=10)
	plt.title("Z={0}".format(len(X)+1), fontsize=10)
	plt.xlabel(r'Rank Variable $r_i$', fontsize = 10)
	plt.ylabel('Estimation Bias Ratio', fontsize = 10)

	# plt.grid()
	# plt.show()
	plt.savefig("rank_variable_est.eps")
	plt.close()

est_bias_curve()