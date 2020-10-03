import os
import json
import natsort
import torch

from PIL import Image
from log import _P, _L, _D, _S
from torch.utils.data import Dataset
from datasets.downloader import download_dataset

DAQUAR_URLS = {
    "qa_all": "https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.txt",
    "qa_train": "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.train.txt",
    "qa_test": "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.test.txt",
    "training_images": "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/train.txt",
    "test_image": "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/test.txt",
    "images": "http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar",
}

class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    SOS_token = 1   # Start-of-sentence token
    EOS_token = 2   # End-of-sentence token

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.num_words = 3
        self.num_sentences = 0
        self.longest_sentence = 0

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]


class Daquar(Dataset):
    """
    DAQUAR dataset, more info can be found at 
    """

    def __init__(self, paths, transform):
        """
        Constructor for the Daquar
        """
        self._paths = paths
        self._images = paths["images_train"]
        self._qa_json = paths["qa_train"]

        self._questions = {}
        self._answers = {}

        self._vocab = Vocabulary('qa')
        self._ans_vocab = Vocabulary('answer')

        self.transform = transform
        self.total_imgs = natsort.natsorted(os.listdir(self._images))

        self._build_vocab()
        self._process_qa()
    
    def _build_vocab(self):
        with open(self._qa_json) as f:
            json_pairs = json.load(f)
            for pair in json_pairs:
                self._vocab.add_sentence(pair["question"])

                for ans in pair["answers"]:
                    self._ans_vocab.add_sentence(ans)

    def _process_qa(self):

        with open(self._qa_json) as f:
            json_pairs = json.load(f)
            for pair in json_pairs:
                q = pair["question"]
                a = pair["answers"]

                self._questions[pair['image_id']] = self._encode_question(q)
                self._answers[pair['image_id']] = self._encode_answers(a)

    def _encode_question(self, question):
        vec = torch.zeros(self._vocab.longest_sentence).long()
        for i, token in enumerate(question.split(' ')):
            vec[i] = self._vocab.to_index(token)

        return vec

    def _encode_answers(self, answers):
        vec = torch.zeros(self._ans_vocab.num_words)
        for ans in answers:
            idx = self._ans_vocab.to_index(ans)
            if idx is not None:
                vec[idx] = 1
        
        return vec
    
    def __len__(self):
        """
        returns the length of the dataset
        """
        return len(self.total_imgs)

    def __getitem__(self, idx):
        """
        return the item from the dataset
        """
        v = self.transform(
            Image.open(os.path.join(self._images, self.total_imgs[idx])).convert("RGB")
        )
        image_id = self.total_imgs[idx].split('image')[-1].split('.png')[0]
        if (image_id not in self._questions.keys()):
            return v, torch.zeros(self._vocab.longest_sentence), self._encode_answers([])

        q = self._questions[image_id]
        a = self._answers[image_id]
        # return v, q, a
        return v, q, a


class DaquarDataset:
    """
    Handler for DAQUAR dataset including images and question answer pairs, more
    info can be found at https://www.mpi-inf.mpg.de    
    """

    def __init__(self, path="./data/daquar", force=False, verbose=False):
        self._path = path
        self._force = force
        self._verbose = verbose
        self._urls = DAQUAR_URLS

        # temp path where the images will be extracted
        self._images_extracted_path = os.path.abspath(
            os.path.join(self._path, self._urls["images"].split("/")[-1].split(".")[0])
        )
        # paths for test and train images
        self._images_test_path = os.path.abspath(
            os.path.join(self._path, "test_images")
        )
        self._images_train_path = os.path.abspath(
            os.path.join(self._path, "train_images")
        )

        # Paths to downloaded data
        self._images_tar_path = os.path.abspath(
            os.path.join(self._path, self._urls["images"].split("/")[-1])
        )
        self._qa_train_path = os.path.abspath(
            os.path.join(self._path, self._urls["qa_train"].split("/")[-1])
        )
        self._train_txt_path = os.path.abspath(
            os.path.join(self._path, self._urls["training_images"].split("/")[-1])
        )
        self._qa_train_json_path = os.path.abspath(
            os.path.join(self._path, "questions_train.json")
        )

        self._paths = {
            "images_test": self._images_test_path,
            "images_train": self._images_train_path,
            "qa_train": self._qa_train_json_path,
        }

        if force or (
            os.path.exists(self._images_test_path) == False
            and os.path.exists(self._images_train_path) == False
        ):
            self._download()
            self._extract_images()
            self._resolve_dirs()
            self._process_images()
            self._process_questions()

    def _download(self):
        # Download the dataset from the web, urls are predefined in the config
        _L("Downloading " + _P("DAQUAR") + " in " + _S(self._path))

        download_dataset(self._urls, self._path)

    def _extract_images(self):
        # untar the download images
        _L("Extracting the images in " + _P(self._images_extracted_path))

        os.system(
            "tar xvfj {} -C {} {}".format(
                self._images_tar_path,
                os.path.join(self._path),
                ">/dev/null 2>&1" if self._verbose == False else " ",
            )
        )

    def _resolve_dirs(self):
        # Del existing directories
        os.system(
            "rm -rf {} {}".format(self._images_test_path, self._images_train_path)
        )

        # Rename the intermediate folder to test_images
        os.system(
            "mv {} {}".format(self._images_extracted_path, self._images_test_path)
        )
        os.makedirs(self._images_train_path, exist_ok=True)

    def _process_images(self):
        # ----------------------------------------------------------------------
        #
        #   Process the downloaded dataset, split the images into test & train
        #   directories, and remove the intermediate files.
        #
        #   The dataset is split according to the list provided in the files in
        #   dataset with names test.txt and train.txt.
        #
        # Seperate files according to train.txt and test.txt

        _L(
            "Seperating files from {} to {}".format(
                _P(self._images_test_path), _S(self._images_train_path)
            )
        )

        with open(self._train_txt_path) as training_images_list:
            for image in [line.rstrip("\n") for line in training_images_list]:

                # mv  files from from_ to to_
                from_ = (
                    os.path.abspath(os.path.join(self._images_test_path, image))
                    + ".png"
                )
                to_ = (
                    os.path.abspath(os.path.join(self._images_train_path, image))
                    + ".png"
                )

                os.system("mv {} {}".format(from_, to_))

                if self._verbose:
                    _L("{} moved to {}".format(_P(from_), _S(to_)))

        _L(
            "Extracted "
            + _P(len(os.listdir(self._images_test_path)))
            + " Images in "
            + _S(self._images_test_path)
        )
        _L(
            "Extracted "
            + _P(len(os.listdir(self._images_train_path)))
            + " Images in "
            + _S(self._images_train_path)
        )

    def _process_questions(self):
        # process the questions
        with open(self._qa_train_path) as questions_json:
            processed_questions = []
            for idx in questions_json:
                idx_n = next(questions_json)

                q = idx.rstrip("\n")
                a = idx_n.rstrip("\n")

                # process questions
                processed = {
                    "image_id": q.split("image")[-1].split(" ?")[0],
                    "question": q.rsplit(" ", 4)[0].replace(",", " ").replace("'", " "),
                    "answers": [ans.strip() for ans in a.split(",")],
                }

                processed_questions.append(processed)

            with open(self._qa_train_json_path, "w", encoding="utf-8") as f:
                json.dump(processed_questions, f, ensure_ascii=False, indent=4)

