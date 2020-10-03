import os

from log import _P, _L, _D, _S

def download_dataset(urls, path):
    """
    Download dataset from the web
    
    Args:
        urls (dic)      : urls to download the dataset
        path (string)   : path where the dataset will be downloaded 
    """

    # check if the path exist or not
    os.makedirs(os.path.normpath(path), exist_ok=True)

    # Download the dataset
    for key in urls:
        _L("Downloading " + _P(urls[key]) + " in " + _S(path))
        # if (urls[key].split('.')[-1] != 'tar'):
        os.system("wget {} -P {}".format(urls[key], path))