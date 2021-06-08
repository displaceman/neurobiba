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
	kef,
	c0 = [244, 215, 94],
	c1 = [233, 114, 61],
	c2 = [11, 127, 171]
	):

	sum_kef = sum(kef)
	kef = list(i/sum_kef for i in kef)

	r = c0[0]*kef[0] + c1[0]*kef[1] + c2[0]*kef[2]
	g = c0[1]*kef[0] + c1[1]*kef[1] + c2[1]*kef[2]
	b = c0[2]*kef[0] + c1[2]*kef[1] + c2[2]*kef[2]
	
	return (int(r), int(g), int(b))


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