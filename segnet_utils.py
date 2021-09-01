#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import numpy as np


class segmentationBuffers:
    def __init__(self, net, args):
        self.net = net
        self.mask = None
        self.overlay = None
        self.composite = None
        self.class_mask = None
        
        self.use_stats = args.stats
        self.use_mask = "mask" in args.visualize
        self.use_overlay = "overlay" in args.visualize
        self.use_composite = self.use_mask and self.use_overlay
        
        if not self.use_overlay and not self.use_mask:
            raise Exception("invalid visualize flags - valid values are 'overlay' 'mask' 'overlay,mask'")
             
        self.grid_width, self.grid_height = net.GetGridSize()	
        self.num_classes = net.GetNumClasses()

    @property
    def output(self):
        if self.use_overlay and self.use_mask:
            return self.composite
        elif self.use_overlay:
            return self.overlay
        elif self.use_mask:
            return self.mask
            
    def Alloc(self, shape, format):
        if self.overlay is not None and self.overlay.height == shape[0] and self.overlay.width == shape[1]:
            return

        if self.use_overlay:
            self.overlay = jetson.utils.cudaAllocMapped(width=shape[1], height=shape[0], format=format)

        if self.use_mask:
            mask_downsample = 2 if self.use_overlay else 1
            self.mask = jetson.utils.cudaAllocMapped(width=shape[1]/mask_downsample, height=shape[0]/mask_downsample, format=format) 

        if self.use_composite:
            self.composite = jetson.utils.cudaAllocMapped(width=self.overlay.width+self.mask.width, height=self.overlay.height, format=format) 

        if self.use_stats:
            self.class_mask = jetson.utils.cudaAllocMapped(width=self.grid_width, height=self.grid_height, format="gray8")
            self.class_mask_np = jetson.utils.cudaToNumpy(self.class_mask)
            
    def ComputeStats(self):
        if not self.use_stats:
            return
            
        # get the class mask (each pixel contains the classID for that grid cell)
        self.net.Mask(self.class_mask, self.grid_width, self.grid_height)

        # compute the number of times each class occurs in the mask
        class_histogram, _ = np.histogram(self.class_mask_np, self.num_classes)

        print('grid size:   {:d}x{:d}'.format(self.grid_width, self.grid_height))
        print('num classes: {:d}'.format(self.num_classes))

        print('-----------------------------------------')
        print(' ID  class name        count     %')
        print('-----------------------------------------')

        for n in range(self.num_classes):
            percentage = float(class_histogram[n]) / float(self.grid_width * self.grid_height)
            print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, self.net.GetClassDesc(n), class_histogram[n], percentage)) 


    def GetClassName(self):
	if not self.use_stats:
            return
            
        # get the class mask (each pixel contains the classID for that grid cell)
        self.net.Mask(self.class_mask, self.grid_width, self.grid_height)
	print('massssskkk')	
	print(self.class_mask_np.shape)
	sq = np.squeeze(self.class_mask_np)
	result = np.where(sq == 3)
	y,x = result
	#print(x)
	#print(y)
	bin2 = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
	one,two = np.histogram(x,bin2)
	print(one,two)
	if len(one) % 3 ==0:
		A = np.split(one,3)
	elif len(one) % 3 == 1:
		A = np.split(one[0:])
	elif len(one) % 3 ==2:
		one.append(one[0])
		A = np.split(one,3)
	#print(np.sum(A[0]))
	return A
	#print(sq[7][1],sq[7][2],sq[7][3],sq[7][14])
	#if sq[6,6]==3 and sq[6,7]== 3 and sq[6,8]==3 and sq[6,9]==3 and sq[5,6]==3 and sq[5,7]== 3 and sq[5,8]==3 and sq[5,9]==3:
		#print('continue straight')
	

        # compute the number of times each class occurs in the mask
        #class_histogram, _ = np.histogram(self.class_mask_np, self.num_classes)

        #print('grid size:   {:d}x{:d}'.format(self.grid_width, self.grid_height))
        #print('num classes: {:d}'.format(self.num_classes))

        #print('-----------------------------------------')
        #print(' ID  class name        count     %')
        #print('-----------------------------------------')

        #for n in range(self.num_classes):
        #    percentage = float(class_histogram[n]) / float(self.grid_width * self.grid_height)
        #    if percentage > 0.2:
	#	print(self.net.GetClassDesc(n))
	#	print(class_histogram[n]) 
        #   print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, self.net.GetClassDesc(n), class_histogram[n], percentage)) 
	#print(self.net.GetClassDesc(self.num_classes))


