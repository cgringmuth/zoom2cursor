#!/usr/bin/env python
"""This module is an example implementation how to zoom to a cursor

The rectangle is meant to be the viewport to the image and the circle is 
the current postion of the cursor. After zooming, the image should be bigger
and the center of the circle should be over the same point in the scaled
image.
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
	def __init__(self, x, y, width, height):
		super(Rect, self).__init__()
		self.x = x
		self.y = y
		self.width = width
		self.height = height

	def overlay(self, img, color=None):
		if color is None:
			color = np.array([0,0,255])
		im = np.copy(img)
		x, y, width, height = self.x, self.y, self.width, self.height
		im[y:y+height, x] = color
		im[y:y+height, x+width] = color

		im[y, x:x+width] = color
		im[y+height, x:x+width] = color

		return im


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
		for a in np.linspace(0, np.pi, 50):
			x = np.cos(a) * self.radius
			y = np.sin(a) * self.radius
			x = int(round(x))
			y = int(round(y))
			im[y + self.y, x + self.x] = color
			im[-y + self.y, x + self.x] = color

		return im




def main():
	im = lena()
	plt.figure()

	cursor = np.array([265, 265])
	circ = Circle(cursor[0], cursor[1], 6)
	
	rect = Rect(200, 200, 100, 100)
	plt.imshow(circ.overlay(rect.overlay(im)))

	scale = 2
	new_cursor = scale * cursor

	shift = cursor * (scale - 1)
	rect.shift(shift[0], shift[1])
	circ.shift(shift[0], shift[1])

	resized = cv2.resize(im, (im.shape[0]*scale, im.shape[1]*scale))

	plt.figure()
	plt.imshow(circ.overlay(rect.overlay(resized)))

	plt.show()


if __name__ == '__main__':
	main()

