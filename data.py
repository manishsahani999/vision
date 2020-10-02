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


def download_daquar(path="./data/daquar", force=False, verbose=False):
    """
    Download DAQUAR dataset, including images and question answer pairs, more
    info can be found at https://www.mpi-inf.mpg.de    
    """

    # urls for the dataset
    urls = DAQUAR_URLS

    # images dir
    images_download_dir = os.path.abspath(
        os.path.join(path, urls["images"].split("/")[-1].split(".")[0])
    )
    test_img_dir = os.path.abspath(
        os.path.join(path, 'test_images')
    )
    train_img_dir = os.path.abspath(
        os.path.join(path, 'train_images')
    )

    paths = {
        'test_images'   : test_img_dir,
        'train_images'  : train_img_dir
    }

    for key in urls:
        paths[key] = os.path.abspath(os.path.join(path, urls[key].split("/")[-1]))

    if force or (os.path.exists(test_img_dir) == False and os.path.exists(train_img_dir)):
        # Del existing directories 
        os.system('rm -rf {} {}'.format(test_img_dir, train_img_dir))

        # download the dataset from the web
        _L('Downloading ' + _P('DAQUAR') + ' in ' + _S(path))
        download_dataset(urls, path)

        # untar the downloaded files
        _L("Extracting the images in " + _P(images_download_dir))
        os.system(
            "tar xvfj {} -C {} {}".format(
                os.path.join(path, urls['images'].split("/")[-1]), os.path.join(path),
                '>/dev/null 2>&1' if verbose == False else ' '
            )
        )
        os.system('mv {} {}'.format(images_download_dir, test_img_dir))

        # Seperate files according to train.txt and test.txt
        _L('Seperating files from {} to {}'.format(_P(test_img_dir), _S(train_img_dir)))
        os.makedirs(train_img_dir, exist_ok=True)
        with open(paths['training_images']) as training_images_list:
            for image in [line.rstrip('\n') for line in training_images_list]:
                from_ = os.path.abspath(os.path.join(test_img_dir, image)) + '.png'
                to_ = os.path.abspath(os.path.join(train_img_dir, image)) + '.png'
                os.system('mv {} {}'.format(from_, to_))
                
                if verbose: _L('{} moved to {}'.format(_P(from_), _S(to_)))

        _L('Extracted ' + _P(len(os.listdir(test_img_dir))) + ' Images in ' + _S())
        _L('Extracted ' + _P(len(os.listdir(train_img_dir))) + ' Images in ' + _S())

    paths.pop('images', None)

    return paths
