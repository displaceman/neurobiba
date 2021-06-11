from numpy import array, dot
from random import randint

def randcolor():
	# Возвращает случайный цвет в формате (255, 255, 255)
	return tuple([randint(0,255) for _ in range(3)])


def mix(
	# Смешивает любое количество colors в пропорциях указанных в coefficient
	coefficient = [1,1,1],
	colors = [
		[244, 215, 94],
		[233, 114, 61],
		[11, 127, 171]
	]):

	sum_kef = sum(coefficient)
	coefficient = array([i/sum_kef for i in coefficient])

	return tuple(map(int, coefficient.dot(array(colors))))