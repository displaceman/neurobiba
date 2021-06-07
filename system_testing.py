from neurobiba import Weights, load_weights, save_weights
import os

def test():
    # Прогоняет выполнение всех основных функций нейробибы

    inp = [0, 1]
    outp = [1, 0]

    w = Weights([2,5,10,4,2], bias = False)
    w.train(inp, outp)

    for i in range(2):
        r = w.feed_forward(inp)
        r = w.feed_backward(r)

    save_weights(w)
    w = load_weights()
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), w.name)
    os.remove(path)

    name = 'test_file_name_weights'
    save_weights(w, name)
    w = load_weights(name)
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    os.remove(path)

    w = Weights([2, 1, 30, 2], bias = True)
    w.train(inp, outp)
    r = w.feed_forward(inp)

    print("Тест завершился без ошибок.")


if __name__ == '__main__':
    test()
