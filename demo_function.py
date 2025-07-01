def add(x: float, y: float = 0.0) -> float:
    """
    Add 2 numbers
    :param x: first param
    :param y: second param
    :return: x + y
    """
    return x + y


if __name__ == '__main__':  # main
    result = add(2.0)
    print(result)
    print(add(y=3, x=2))
