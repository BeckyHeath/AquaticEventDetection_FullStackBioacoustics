# Written by Avery Bick, June 2021
# Adapted from UMAP documentation: https://umap-learn.readthedocs.io/en/latest/supervised.html
"""
Apply UMAP dimensionality reduction to Audioset embeddings
"""

import umap
import matplotlib
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
import sys
from itertools import chain
from datetime import date, time, datetime
import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)

def splitEmbeddingsByColumnValue(df, column):
	dfs = [d for _, d in df.groupby([column])]
	values = df[column].unique()
	return(dfs, values)

def createTimeLabelsList(length, percentiles):
	outList = []
	labelCount = int(length/percentiles)

	for percentile in list(range(1,percentiles)):
		tmpList = [percentile]*labelCount
		outList.extend(tmpList)

	if len(outList) < length:
		nextNumber = outList[-1]+1
		print(nextNumber)
		while len(outList) < length:
			outList.append(nextNumber)

	if len(outList) > length:
		while len(outList) > length:
			outList = outList[:-1]

	outArray = np.array(outList)

	return(outArray)

def calculateCentroids_Continuous(embeddings, timeLabels):
	classCentroids = []
	arr = np.column_stack((embeddings,timeLabels))
	split_arrs = np.split(arr, np.where(np.diff(arr[:,2]))[0]+1)
	for array in split_arrs:
		length = array.shape[0]
		sum_x = np.sum(array[:, 0])
		sum_y = np.sum(array[:, 1])
		centroid = [sum_x/length,sum_y/length]
		classCentroids.append(centroid)
	outArray = np.array(classCentroids)
	return outArray

def calculateCentroids_Classes(l):
	array = np.array(l)
	length = array.shape[0]
	sum_x = np.sum(array[:, 0])
	sum_y = np.sum(array[:, 1])
	centroid = [sum_x/length,sum_y/length]
	return centroid

def plotUMAP_Continuous(audioEmbeddingsList, percentiles, title, 
                        colormap, classNames=None):
	"""
	Create a UMAP plot for audio embeddings with categories by time percentile
	Pass in:
		audioEmbeddings: Single list of audio embeddings
		percentiles: How many times to split the list
		classNames: List of names of classes
		title: name of plot
		colormap: matplotlib colormap for plot

	"""

	print(audioEmbeddingsList)
	embeddings = umap.UMAP(n_neighbors=10).fit_transform(audioEmbeddingsList)
	timeLabels = createTimeLabelsList(len(audioEmbeddingsList), percentiles)
	classes = list(set(timeLabels))
	if classNames == None:
		classNames = classes

	centroids = calculateCentroids_Continuous(embeddings,timeLabels)

	fig, ax = plt.subplots(1, figsize=(14, 10))
	plt.scatter(*embeddings.T, s=10., c=timeLabels, cmap=colormap, alpha=.8)
	plt.scatter(*centroids.T, s=500, c=classes, cmap=colormap, alpha=1.0) ,#edgecolors='black')
	plt.setp(ax, xticks=[], yticks=[])	
	cbar = plt.colorbar(boundaries=np.arange(percentiles))
	cbar.set_ticks(np.arange(percentiles))
	cbar.set_ticklabels(classNames)
	plt.title('UMAP Embedding for {}'.format(title))
	plt.show()
 
 
def plotUMAP_Continuous_plotly(audioEmbeddingsList, percentiles, title, 
                        colormap, files, lengths, classNames=None):
	import plotly.express as px
	import plotly.graph_objects as go
	from pathlib import Path
	# print(audioEmbeddingsList)
	embeddings = umap.UMAP(n_neighbors=10).fit_transform(audioEmbeddingsList)
	timeLabels = createTimeLabelsList(len(audioEmbeddingsList), percentiles)
	classes = list(set(timeLabels))
	if classNames == None:
		classNames = classes

	centroids = calculateCentroids_Continuous(embeddings,timeLabels)

	embeddings = np.array(embeddings)
	centroids = np.array(centroids)
	test = ['a'] * len(embeddings[:,1])
	
	lin_array = np.linspace(0, 600, lengths[0])
	divisions = lin_array // 60 + np.mod(lin_array, 60)/100
	divisions = np.round(divisions, 3)
	div_strings = divisions.astype(str)
 
	files_array = []
	divisions_array = []
	for i in range(len(lengths)):
		for j in range(lengths[i]):
			files_array.append(files[i])
			divisions_array.append(div_strings[j])
 
	data = pd.DataFrame({'x' : embeddings[:,0], 'y':embeddings[:,1],
                      'time_within_file' : divisions_array,
                      'filename' : files_array})

	fig = px.scatter(data, x='x', y='y', color=timeLabels, opacity = 0.2,
                  hover_data = ['time_within_file', 'filename'],
				  title = 'UMAP Embedding for {}'.format(title))
	fig.add_trace(
      go.Scatter(
          x = centroids[:,0], y= centroids[:,1], mode = 'markers',
		  marker = dict(
			 color = classes,
			 size = [20]*10
		  ) ) )
	fig.show()
 
 
	# px.scatter(*centroids.T, s=500, c=classes, cmap=colormap, alpha=1.0) ,#edgecolors='black')
	# plt.setp(ax, xticks=[], yticks=[])	
	# cbar = plt.colorbar(boundaries=np.arange(percentiles))
	# cbar.set_ticks(np.arange(percentiles))
	# cbar.set_ticklabels(classNames)
	# plt.title('UMAP Embedding for {}'.format(title))
	# plt.show()

# def plotUMAP_Classes(df, title=None, colors=None):
# 	"""
# 	Create a UMAP plot for audio embeddings with categories by class
# 	Pass in:
# 		audioEmbeddingsLists: List of lists of audio embeddings for each class
# 		classNames: List of names of classes
# 		title: name of plot
# 		colors: list of colors for each class

# 	Need to calculate UMAP fit_transform on entire audio embeddings set, then calculate centroids and plot for each class

# 	"""
# 	# Initialize Plot
# 	fig, ax = plt.subplots(1, figsize=(14, 10))

# 	# Calculate UMAP embeddings for all audio features
# 	UMAPembeddings = umap.UMAP(n_neighbors=10).fit_transform(df['embeddings'].tolist())
# 	df['UMAP'] = UMAPembeddings.tolist()
# #	df.to_pickle('./UMAP_df.pickle')
# #	df = pd.read_pickle('./UMAP_df.pickle')

# 	# Split df into dfs based on a column value
# 	dfs, classes = splitEmbeddingsByColumnValue(df, 'month')

# 	# Create a list to store centroids
# 	centroids = []

# 	# Plot each class on same plot
# 	for df, color, className in zip(dfs, colors, classes):
# 		centroid = calculateCentroids_Classes(df.UMAP.tolist())
# 		centroids.append(centroid)
# 		plt.scatter(*zip(*df.UMAP.tolist()), s=1., c=color, alpha=.6, label=className)

# 	# Plot each centroid
# 	for centroid, color, className in zip(centroids, colors, classes):
# 		plt.scatter(centroid[0], centroid[1], s=500, c=color, alpha=1.)#, edgecolors='black')
# 		#plt.text(centroid[0], centroid[1], className, fontsize=12, weight='bold')


# 	plt.setp(ax, xticks=[], yticks=[])
# 	plt.title('UMAP Embedding for {}'.format(title))
# 	plt.legend()
# 	lgnd = plt.legend(loc="upper right", fontsize=14)
# 	for handle in lgnd.legendHandles:
# 		handle.set_sizes([20.0])
# 	plt.show()

if __name__ == "__main__":
#	audioEmbeddings = pickle.load(open('/Volumes/HighTideSSD/_PhD_Data/cornell_data_for_avery/unbal_cornell_seasonal_mic_raw_audioset_feats_300s.pickle', "rb"))
#	print(audioEmbeddings)
	folders = glob.glob('../data/*Aug3')[1:]
	df_all = pd.DataFrame()
	file_list= []
	lenghts = []
	for folder in folders:
		files = glob.glob(folder + '/*.pickle')
		for file in files:
			with open(file, 'rb') as savef:
				audioEmbeddings = pickle.load(savef)
				# print(audioEmbeddings[0])
				audioEmbeddingsList = [arr.tolist() for arr in audioEmbeddings]
				df = pd.DataFrame({'embeddings':audioEmbeddingsList})
				print(df)
				'''
				audioEmbeddings, labs, datetimes, recorders, unique_ids, classes, mins_per_feat = pickle.load(savef)
				print(audioEmbeddings)
				print(audioEmbeddings.shape, labs.shape, datetimes.shape, recorders.shape, unique_ids.shape)
				audioEmbeddingsList = [arr.tolist() for arr in audioEmbeddings]
				print('listCreated')
				df = pd.DataFrame({'embeddings':audioEmbeddingsList,'labs':labs, 'datetimes':datetimes, 'recorders':recorders,'unique_ids':unique_ids})
				df['datetimes'] = pd.to_datetime(df['datetimes'])
				df['year'], df['month'], df['day'], df['hour'] = df['datetimes'].dt.year, df['datetimes'].dt.month, df['datetimes'].dt.day, df['datetimes'].dt.hour

				df = df.loc[df['hour'].isin([4,5,6,7])]
				'''
				df_all = df_all.append(df, ignore_index = True)
				lenghts.append(len(df))
				file_list.append(file)


	percentiles = 10
	plotUMAP_Continuous_plotly(df_all['embeddings'].tolist(), percentiles, f'Aug2-Aug3 + {folders}', 'plasma', file_list, lenghts)
	# plotUMAP_Continuous(df['embeddings'].tolist(), percentiles, '630 - 640 AM', 'plasma', files)
	
	sys.exit()

	cmap = matplotlib.cm.get_cmap('viridis')
	hours = list(range(13))[1:]
	hours = [x/12. for x in hours]
	colors = [cmap(x) for x in hours]

#	plotUMAP_Classes(df, colors=colors,title='Hourly Embeddings in April and May')
#	plotUMAP_Classes(df, colors=[cmap(0.0),cmap(0.083),cmap(0.166),cmap(0.249),cmap(0.332),cmap(0.415),cmap(0.498),cmap(0.581),cmap(0.664),cmap(0.747),cmap(0.83),cmap(0.913)], title='Cornell Forest Soundscapes')
#	plotUMAP_Classes(df, colors=[cmap(0.0),cmap(0.332),cmap(0.664),cmap(0.913)], title='Cornell Forest Soundscapes')