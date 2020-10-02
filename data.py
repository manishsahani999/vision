from log import _P, _L, _D, _S
from torch.utils.data import Dataset
from PIL import Image
import os
import natsort

DAQUAR_URLS = {
    "qa_all"            : "https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.txt",
    "qa_train"          : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.train.txt",
    "qa_test"           : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.test.txt",
    "training_images"   : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/train.txt",
    "test_image"        : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/test.txt",
    "images"            : "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar",
}

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
        os.system("wget {} -P {}".format(urls[key], path))


class Daquar(Dataset):
    """
    DAQUAR dataset, more info can be found at 
    """

    def __init__(self, main_dir, transform):
        """
        Constructor for the Daquar
        """
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = natsort.natsorted(os.listdir(main_dir))

    def __len__(self):
        """
        returns the length of the dataset
        """
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """
        return the item from the dataset
        """
        return self.transform(
            Image.open(os.path.join(self.main_dir, self.total_imgs[idx])).convert("RGB")
        )


def download_daquar(path="./data/daquar", force=False):
    """
    Download DAQUAR dataset, including images and question answer pairs, more
    info can be found at https://www.mpi-inf.mpg.de    
    """

    # urls for the dataset
    urls = DAQUAR_URLS

    # images dir
    images_dir = os.path.abspath(
        os.path.join(path, urls["images"].split("/")[-1].split(".")[0])
    )

    if force or os.path.exists(images_dir) == False:
        # download the dataset from the web
        _L('Downloading ' + _P('DAQUAR') + ' in ' + _S(path))
        # download_dataset(urls, path)

        # untar the downloaded files
        _L("Extracting the images in " + _P(images_dir))
        os.system(
            "tar xvfj {} -C {}".format(
                os.path.join(path, urls['images'].split("/")[-1]), os.path.join(path)
            )
        )
        _L('Extracted ' + _P(len(os.listdir(images_dir))) + ' Images in ' + _S(images_dir))

    paths = {}
    for key in urls:
        paths[key] = os.path.abspath(os.path.join(path, urls[key].split("/")[-1]))

    paths["images"] = images_dir
    
    return paths
