def indicator(x,a):
    if(x>=a):
        return 1
    else:
        return 0



def harmonic_sequence(n):
    """
    Generate the harmonic sequence 1/n for n = 1, 2, 3, ..., up to n terms.
    """
    return [1 / i for i in range(1, n + 1)]

