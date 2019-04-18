import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config
from math import sqrt

class Embedding (nn.Module):

	def __init__ (self):
		super(Embedding, self).__init__()
		self.model = models.vgg16(pretrained = True)
		self.features = nn.Sequential(
				*list(self.model.features.children())[:-3])
	def forward (self, x):
		x = self.features(x)
		return x

class XCorrelation (nn.Module):

	def __init__ (self):
		pass

	def forward (self, x, kernel):
		x = torch.Tensor(x)
		kernel = torch.Tensor(kernel)

		assert(len(x.shape) == 4)
		assert(len(kernel.shape) == 4)
	
		print x.shape, '*', kernel.shape
		
		op = nn.Conv2d(in_channels = kernel.size(0), 
					   out_channels = 1,
					   kernel_size = (kernel.size(1), kernel.size(2)),
					   stride = 1,
					   bias = False)
		op.weight.data = kernel
		y = op.forward(x)
		return y

def readImg(path, dSize = None, scale = None, expand_dim = True, asTensor = True):
	x = cv2.imread(path)
	if dSize is not None and scale is None:
		dSize = (int(dSize[0]), int(dSize[1]))
		x = cv2.resize(x, dSize)
	elif scale is not None and dSize is None:
		x = cv2.resize(x, (0, 0), fx = scale, fy = scale)
	elif scale is None and dSize is None:
		pass
	else:
	 	print 'dSize: ', dSize
	 	print 'scale: ', scale
		print 'readImg: only one between dSize and scale should be None value'
		exit(-1)

	x = np.asarray(x)
	x = x.transpose(2,0,1)
	if expand_dim:
		x = np.expand_dims(x, axis = 0)
	if asTensor:
		x = torch.Tensor(x)
	return x
	
def getScalePyramid(path, search_size):
	# search_size hould be given in opencv order (W, H)
	patch = cv2.imread(path)
	patch_area = patch.shape[0] * patch.shape[1]
	search_area = search_size[0] * search_size[1]
	scales = config.scales
	sizes = [(patch.shape[0]*scale, patch.shape[1]*scale)
			 for scale in scales]
	pyramid = list()
	for i, size in enumerate(sizes):
		print 'Generating image of size ', size
		print search_size, ' x ' , scales[i]
		img = readImg(path, dSize=size)
		'''
		vis = readImg(path, dSize=size, 
					  expand_dim=False, asTensor=False)
		plt.imshow(vis.transpose(1,2,0))
		vis[:,:,0], vis[:,:,2] = vis[:,:,2], vis[:,:,0]
		plt.show()
		'''
		pyramid.append(img)
	return pyramid, sizes

if __name__ == '__main__':

	search_img = config.search_img
	target_img = config.target_img
# scale = config.scale

	embed = Embedding()
	embed.eval()
	corr = XCorrelation()

	x = readImg(search_img, expand_dim = True)
	print 'x: ', x.shape
	# input_size should be in opencv order
	input_size = x.shape[3], x.shape[2] 
	x = embed(x)
	print 'x embedded to: ', x.shape

	pyramid, patch_sizes = getScalePyramid(target_img, input_size)
	max_score = 0.0
	final_pos = None
	final_patchsize = None
	for i, y in enumerate(pyramid):
		print 'y: ', y.shape
		y = embed(y)
		print 'y embedded to: ', y.shape

		smap_gray = corr.forward(x, y)
		print 'score map: ', smap_gray.shape

		smap_gray = smap_gray.detach().numpy().squeeze()
		smap_gray = np.asarray(cv2.resize(smap_gray, input_size))
		smap = np.zeros((input_size[1], input_size[0], 3))
		
		similiarity = np.amax(smap_gray)/sqrt(np.prod(patch_sizes[i]))
		smap[:,:,2] = smap_gray/np.amax(smap_gray)
		
		max_pos = np.unravel_index(np.argmax(smap_gray, axis = None), 
								   smap_gray.shape)
		if max_score < similiarity:
			final_pos = max_pos
			final_patchsize = patch_sizes[i]
		print 'smap:', smap.shape
		print 'maximum similiarity', similiarity, 'at', max_pos
		print 'normalized by patch size ', patch_sizes[i]
		'''
		smap[max_pos[0]-3:max_pos[0]+4, 
			 max_pos[1]-3:max_pos[1]+4, 0] = 1.0
		smap[max_pos[0]-3:max_pos[0]+4, 
			 max_pos[1]-3:max_pos[1]+4, 1:] = 0.0
		'''

	fig, ax = plt.subplots(1)
	x_vis = cv2.imread(search_img)
	x_vis = cv2.cvtColor(x_vis, cv2.COLOR_BGR2RGB)
	plt.imshow(x_vis)
# plt.imshow(smap)
	patch_size = patch_sizes[i]
	print 'patch_size:', patch_size	
	rect = patches.Rectangle(
		   (final_pos[1]-final_patchsize[0]/2, 
			final_pos[0]-final_patchsize[1]/2),
		    final_patchsize[0], final_patchsize[1],
		    linewidth=3, edgecolor='r', facecolor = 'none')
	ax.add_patch(rect)
	plt.show()

