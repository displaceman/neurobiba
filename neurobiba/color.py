from numpy import array, dot
from random import randint
import colorsys

class Color():
	def __init__(self, color, norm=True):
	
		self.color = tuple(color)
		self.norm = norm

		self.r = self.color[0]
		self.g = self.color[1]
		self.b = self.color[2]

	def get_255(self):
		if self.norm==True:
			return Color(int(i*255) for i in self.color)
		else:
			return self

	def get_1(self):
		if self.norm==True:
			return self
		else:
			return Color(int(i*255) for i in self.color)

	def clamp(self):
		if norm==True:
			n = 1
		else:
			b = 255
		return Color(max(min(i, n), 0) for i in self.color)

	def __abs__(self):
		return Color(abs(i) for i in self.color)

	def __add__(self, other): # +
		return Color(i+other for i in self.color)
	def __sub__(self, other): # -
		return Color(i-other for i in self.color)
	def __mul__(self, other): # *
		return Color(i*other for i in self.color)
	def __floordiv__(self, other): # //
		return Color(i//other for i in self.color)
	def __truediv__(self, other): # /
		return Color(i/other for i in self.color)
	def __mod__(self, other): # %
		return Color(i%other for i in self.color)
	def __pow__(self, other): # **
		return Color(i**other for i in self.color)
	def mix(self, oter, v):
		return Color(self.color[i]*(1-v)+oter.color[i]*v for i in range(3))

	def rgb_to_hls(self):
		return Color(colorsys.rgb_to_hls(self.r, self.g, self.b))

	def hls_to_rgb(self):
		return Color(colorsys.hls_to_rgb(self.r, self.g, self.b))



def randcolor():
	return tuple([randint(0,255) for _ in range(3)])


def mix(
	kef = [1,1,1],
	colors = [
		[244, 215, 94],
		[233, 114, 61],
		[11, 127, 171]
	]):

	colors = array(colors)
	sum_kef = sum(kef)
	kef = array([i/sum_kef for i in kef])

	return tuple(map(int, kef.dot(colors)))



if __name__ == "__main__":
	pass
	#print(a.__dir__())


# Палитры

# [244, 215, 94],
# [233, 114, 61],
# [11, 127, 171]

# [249, 222, 89],
# [249, 131, 101],
# [161, 223, 251]