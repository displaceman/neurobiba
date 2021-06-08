from numpy import array, dot

class Color():
	def __init__(self, color, norm=False):
		if norm:
			self.color = tuple(int(i*255) for i in color)
			self.norm = tuple(color)
		else:
			self.color = tuple(color)
			self.norm = tuple(i/255 for i in color)
	def __add__(self, other): # +
		return tuple(i+other for i in self.color)
	def __sub__(self, other): # -
		return tuple(i-other for i in self.color)
	def __mul__(self, other): # *
		return tuple(i*other for i in self.color)
	def __floordiv__(self, other): # //
		return tuple(i//other for i in self.color)
	def __truediv__(self, other): # /
		return tuple(i for i in self.color)
	def __mod__(self, other): # %
		return tuple(i%other for i in self.color)
	def __pow__(self, other): # **
		return tuple(i**other for i in self.color)
	def mix(self, oter):
		return tuple(int((self.color[i]+oter.color[i])*0.5) for i in range(3))


def mix3(
	kef = [1,1,1],
	colors = [
		[244, 215, 94],
		[233, 114, 61],
		[11, 127, 171],
		]
	):

	colors = array(colors)
	sum_kef = sum(kef)
	kef = array([i/sum_kef for i in kef])
	return tuple(map(int, kef.dot(colors)))



if __name__ == "__main__":
	a = Color([1,1,1])
	if (a+1)[0] != 2:
		print("Error +")
	if (a-1)[0] != 0:
		print("Error -")
	if (a*2)[0] != 2:
		print("Error *")
	if (a//1)[0] != 1:
		print("Error //")
	if (a/1)[0] != 1:
		print("Error /")
	if (a%1)[0] != 0:
		print("Error %")
	if (a**2)[0] != 1:
		print("Error **")
	if(a.mix(Color([0,0,0])))[0] != 0:
		print("Error mix")
	print("test passed")
	#print(a.__dir__())

# Палитры

# c0 = [244, 215, 94],
# c1 = [233, 114, 61],
# c2 = [11, 127, 171]

# c0 = [249, 222, 89],
# c1 = [249, 131, 101],
# c2 = [161, 223, 251]