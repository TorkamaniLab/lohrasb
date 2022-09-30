# package version in pypi
import pkg_resources
lohrasb_version = pkg_resources.get_distribution('lohrasb').version

__version__ = lohrasb_version



if __name__=="__main__":
    print(__version__)
