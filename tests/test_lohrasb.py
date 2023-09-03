from lohrasb import __version__
lohrasb_version = __version__ 

def test_version():
    """test version"""
    print(lohrasb_version)
    assert __version__ == lohrasb_version


if __name__=="__main__":
    print(__version__)

