#!/usr/bin/env python
"""This module is an example implementation how to zoom to a cursor

The rectangle is meant to be the viewport to the image and the circle is 
the current postion of the cursor. After zooming, the image should be bigger
and the center of the circle should be over the same point in the scaled
image.

New version is with opencv and by clicking the red vieport area change
the cursor position
"""

from skimage.data import lena
import matplotlib.pyplot as plt
import cv2
import numpy as np


class Shape(object):
	"""docstring for shape"""
	def __init__(self):
		super(Shape, self).__init__()

	def shift(self, x, y):
		self.x += x
		self.y += y
		

class Rect(Shape):
	"""docstring for Rect"""
	def __init__(self, x, y, width, height, circ):
		super(Rect, self).__init__()
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.circ = circ

	def get_box(self):
		return self.x, self.y, self.x+self.width, self.y+self.height

	def overlay(self, img, color=None):
		if color is None:
			color = np.array([0,0,255])
		im = np.copy(img)
		x1, y1, x2, y2 = self.get_box()
		im[y1:y2, x1] = color
		im[y1:y2, x2] = color

		im[y1, x1:x2] = color
		im[y2, x1:x2] = color

		im = self.circ.overlay(im)

		return im

	def crop(self, img):
		im = np.copy(img)
		x1, y1, x2, y2 = self.get_box()
		im = im[y1:y2, x1:x2]
		return im

	def shift(self, x, y):
		super(Rect, self).shift(x, y)
		self.circ.shift(x, y)


class Circle(Shape):
	"""docstring for Circle"""
	def __init__(self, x, y, radius):
		super(Circle, self).__init__()
		self.x = x
		self.y = y
		self.radius = radius

	def overlay(self, img, color=None):
		im = np.copy(img)
		if color is None:
			color = np.array([0,0,255])
		# for a in np.linspace(0, np.pi, 50):
		# 	x = np.cos(a) * self.radius
		# 	y = np.sin(a) * self.radius
		# 	x = int(round(x))
		# 	y = int(round(y))
		# 	im[y + self.y, x + self.x] = color
		# 	im[-y + self.y, x + self.x] = color

		im = cv2.circle(img,(self.x,self.y),self.radius,(255,0,0),1)

		return im


# mouse callback function
def mouse_callback(event,x,y,flags,param):
	global cur_x
	global cur_y
	if event == cv2.EVENT_LBUTTONDOWN:
		cur_x, cur_y = x, y


def scale(cur_x, cur_y, im):
	cursor = np.array([cur_x, cur_y])
	cursor_circ = Circle(cursor[0], cursor[1], 6)
	rect = Rect(200, 200, 200, 200, cursor_circ)
	# plt.imshow(rect.overlay(im))
	cv2.imshow('orig', rect.overlay(im))

	scale = 3
	new_cursor = scale * cursor

	shift = cursor * (scale - 1)
	rect.shift(shift[0], shift[1])

	resized = cv2.resize(im, (im.shape[0]*scale, im.shape[1]*scale))
	crop = rect.crop(rect.overlay(resized))
	cv2.imshow('scaled', crop)

cur_x, cur_y = 250, 250

def main():
	im = lena()
	im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

	cv2.namedWindow('orig')
	cv2.setMouseCallback('orig', mouse_callback)

	cv2.namedWindow('scaled')
	
	while (1):
	    scale(cur_x, cur_y, im)
	    if cv2.waitKey(20) & 0xFF == 27:
	        break

	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()

