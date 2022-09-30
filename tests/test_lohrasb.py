from lohrasb import __version__
import pkg_resources
lohrasb_version = pkg_resources.get_distribution('lohrasb').version

def test_version():
    """test version"""
    print(lohrasb_version)
    assert __version__ == lohrasb_version


if __name__=="__main__":
    print(__version__)