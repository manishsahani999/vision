# Copyright (C) 2020 Manish Sahani <rec.manish.sahani@gmail.com>.
#
# This code is Licensed under the Apache License, Version 2.0 (the "License");
# A copy of a License can be obtained at:
#                 http://www.apache.org/licenses/LICENSE-2.0#
#
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and 
# limitations under the License.

# --*= daquar.py =*----

import os
import json
import torch
import natsort

from PIL import Image
from log import _P, _L, _D, _S 
from torch.utils.data import Dataset

# dataset modules developed as part of this resarch project 
from datasets.vocabulary import Vocabulary
from datasets.downloader import download_dataset
from datasets.urls import DAQUAR_URLS, DAQUAR_IM, DAQUAR_QA_TRAIN, DAQUAR_IM_TRAIN

###############################################################################
# 
#   Daquar is a major public and opensource dataset for visaul question answers
#   it has around 1500 images of total size ~450MB, and around 16K training que
#   -stion answers pair.
# 
###############################################################################
class DaquarDataFolder:
    """
    DaquarDataFolder is the Handler for DAQUAR dataset, includes the 
    utilites to download images, question answer pairs and txt files. 

    Also has the data processing processing utils, including spliting of
    images, translating question answer pairs in a json.
    """

    def __init__(self, path="./data/daquar", force=False, verbose=False):
        """Construct a brand new Dqauar Data Folder

        Args:
            path (str, optional): path for the folder. Defaults to "./data/daquar".
            force (bool, optional): force download the dataset. Defaults to False.
            verbose (bool, optional): detailed loggin while processing. Defaults to False.
        """
        self._path = os.path.abspath(path)
        self._force = force
        self._verbose = verbose
        self._urls = DAQUAR_URLS

        self._IM_DIR_TEST = 'test_images'
        self._IM_DIR_TRAIN = 'train_images'
        self._QA_JSON_TRAIN = 'qa_train.json'

        ###############   Paths for image directories and files ###############

        # images 
        self._im_extracted_path = os.path.join(
            self._path, self._urls[DAQUAR_IM].split('/')[-1].split('.')[0]
        )
        self._im_test_path = os.path.join(self._path, self._IM_DIR_TEST)
        self._im_train_path = os.path.join(self._path, self._IM_DIR_TRAIN)
        self._im_tar_path = os.path.join(self._path, self._urls[DAQUAR_IM].split('/')[-1])
        self._im_train_list_path = os.path.join(
            self._path, self._urls[DAQUAR_IM_TRAIN].split('/')[-1]
        )
        
        # qa pairs
        self._qa_train_txt_path = os.path.join(
            self._path, self._urls[DAQUAR_QA_TRAIN].split('/')[-1]
        )
        self._qa_train_json_path = os.path.join(self._path, self._QA_JSON_TRAIN)

        #----------------------------------------------------------------------
        #  
        #   logging if verbose is set to true 
        # 
        #
        if self._verbose:
            _L('Images .tar path '          + _P(self._im_tar_path))
            _L('Images extraction path '    + _P(self._im_extracted_path))
            _L('Test images path '          + _P(self._im_test_path))
            _L('Train images path '         + _P(self._im_train_path))
            _L('Train pairs text '          + _P(self._qa_train_txt_path))
            _L('Train processed json '      + _P(self._qa_train_json_path))
            _L('Images train list path'     + _P(self._im_train_list_path))

        self.paths = {
            self._IM_DIR_TEST : self._im_test_path,
            self._IM_DIR_TRAIN: self._im_train_path,
            "qa_train": self._qa_train_json_path,
        }

        if force or (
            os.path.exists(self._im_train_path) == False and 
            os.path.exists(self._im_test_path) == False
        ):
            self._download()
            self._extract_images()
            self._resolve_dirs()
            self._process_images()
            self._process_questions()

    def _download(self):
        """Download the dataset from the web, urls are predefined in the config
        """
        if self._verbose: _L("Downloading " + _P("DAQUAR") + " in " + _S(self._path))

    #     download_dataset(self._urls, self._path)

    def _extract_images(self):
        """extract the downloaded images
        """
        if self._verbose: _L("Extracting the images in " + _P(self._im_extracted_path))

        os.system(
            "tar xvfj {} -C {} {}".format(
                self._im_tar_path, self._path,
                ">/dev/null 2>&1" if self._verbose == False else " ",
            )
        )

    def _resolve_dirs(self):
        """Resolve directories, delete old directories and create new ones
        """
        if self._verbose: _L("Resolving directories")
        # Del existing directories
        os.system(
            "rm -rf {} {}".format(self._im_test_path, self._im_train_path)
        )

        # Rename the intermediate folder to test_images
        os.system(
            "mv {} {}".format(self._im_extracted_path, self._im_test_path)
        )
        os.makedirs(self._im_train_path, exist_ok=True)

    def _process_images(self):
        #   Process the downloaded dataset, split the images into test & train
        #   directories, and remove the intermediate files.
        #
        #   The dataset is split according to the list provided in the files in
        #   dataset with names test.txt and train.txt.
        #

        # Seperate files according to train.txt and test.txt

        if self._verbose: _L("Seperating files from {} to {}".format(
                _P(self._im_test_path), _S(self._im_train_path)
            ))

        with open(self._im_train_list_path) as training_images_list:
            for image in [line.rstrip("\n") for line in training_images_list]:

                # mv  files from from_ to to_
                from_ = (os.path.join(self._im_test_path, image) + ".png")
                to_ = (os.path.join(self._im_train_path, image)+ ".png")

                os.system("mv {} {}".format(from_, to_))

                if self._verbose:
                    _L("{} moved to {}".format(_P(from_), _S(to_)))
        
        if self._verbose: 
            _L(
                "Extracted " + _P(len(os.listdir(self._im_test_path)))
                + " Images in " + _S(self._im_test_path)
            )
            _L(
                "Extracted " + _P(len(os.listdir(self._im_train_path)))
                + " Images in " + _S(self._im_train_path)
            )

    def _process_questions(self):
        # process the questions
        with open(self._qa_train_txt_path) as questions_json:
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


class Daquar(Dataset):
    """
    DAQUAR dataset, more info can be found at 
    """

    def __init__(self, paths, transform):
        """
        Constructor for the Daquar
        """
        self._paths = paths
        self._images = paths["train_images"]
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
        vec = torch.zeros(self._vocab._longest_sentence).long()
        for i, token in enumerate(question.split(' ')):
            vec[i] = self._vocab.to_index(token)

        return vec

    def _encode_answers(self, answers):
        vec = torch.zeros(self._ans_vocab._num_words)
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
            return v, torch.zeros(self._vocab._longest_sentence), self._encode_answers([])

        q = self._questions[image_id]
        a = self._answers[image_id]
        # return v, q, a
        return v, q, a
