import os
from log import _P, _L, _D, _S


class Dataset:
    """
    Class to download Dataset
    """
    def __init__(self, urls, path):
        """
        Constructor for creating the Dataset instance
        """
        self._urls = urls
        self._path = path

        # check if the path exist or not 
        os.makedirs(os.path.normpath(path), exist_ok=True)
    
    def download(self):
        """
        Download the dataset from the web
        """
        for url in self._urls:
            _L('Downloading ' + _P(url) + ' in ' + self._path)
            os.system('wget {} -P {path}'.format(url, path=self._path))

def download_daquar(path="./data/daquar"):
    """
    Download DAQUAR dataset, including images and question answer pairs, more
    info can be found at 
        https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge
        
    """
    
    # Urls for the dataset 
    urls = [
        "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.train.txt",
        "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.test.txt"
        "https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.txt",
        "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/train.txt",
        "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/test.txt",
        # images
        "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar",
    ]

    daquar = Dataset(urls, path)

    # download the dataset from the web
    daquar.download()

    # untar the downloaded files
    images_dir = os.path.abspath(
        os.path.join(path, urls[-1].split("/")[-1].split(".")[0])
    )
    
    _L("Extracting the files in " + _P(images_dir))
    os.system(
        "tar xvfj {} -C {}".format(
            os.path.join(path, urls[-1].split("/")[-1]), os.path.join(path)
        )
    )
    _L(_P(len(os.listdir(images_dir))) + " Images available in " + _S(images_dir))

    paths = [images_dir]
    for url in urls:
        paths.append(os.path.abspath(os.path.join(path, url.split("/")[-1])))
    return paths


if __name__ == "__main__":
    files = download_daquar()
