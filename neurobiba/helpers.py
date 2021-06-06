def counter():
    def count():
        nonlocal value
        value += 1
        return value

    value = -1
    return count


default_counter = counter()
