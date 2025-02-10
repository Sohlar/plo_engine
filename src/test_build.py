import train

class args:
    interactive = False
    numhands = 100


def check():
    return train.main(args)

def test_answer():
    assert check() # This is the assert that needs to pass
