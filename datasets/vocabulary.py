# Copyright (C) 2020 Manish Sahani.
#
# This code is Licensed under the Apache License, Version 2.0 (the "License");
# A copy of a License can be obtained at:
#                 http://www.apache.org/licenses/LICENSE-2.0#
#
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# --*= vocabulary.py =*----


class Vocabulary:
    """Vocabulary, helps in the preprocessing of the corpus text 
    """

    PAD_TOKEN = 0  # Used for padding short sentences
    SOS_TOKEN = 1  # Start-of-sentence token
    EOS_TOKEN = 2  # End-of-sentence token

    def __init__(self, name):
        """Construct a brand new Vocabulary

        Args:
            name (string): Name of the vocabulary
        """
        self._name = name
        self._word2index = {}
        self._word2count = {}
        self._index2word = {self.PAD_TOKEN: 'PAD', self.SOS_TOKEN: 'SOS', self.EOS_TOKEN: 'EOS'}
        self._num_words = 3
        self._num_sentences = 0
        self._longest_sentence = 0

    def add_word(self, word):
        """Add word to the vocabulary

        Args:
            word (string): word to be added in the vocabulary
        """
        if word not in self._word2index:
            # First entry of word into vocabulary
            self._word2index[word] = self._num_words
            self._word2count[word] = 1
            self._index2word[self._num_words] = word
            self._num_words += 1
        else:
            # Word exists; increase word count
            self._word2count[word] += 1

    def add_sentence(self, sentence):
        """Add a sentence to the vocabulary

        Args:
            sentence (string): sentence to be added in the vocabulary
        """
        sentence_len = 0
        for word in sentence.split(" "):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self._longest_sentence:
            # This is the longest sentence
            self._longest_sentence = sentence_len
        # Count the number of sentences
        self._num_sentences += 1

    def to_word(self, index):
        """Return word mapped to the given index

        Args:
            index (int): index for the word

        Returns:
            string: word at index 
        """
        return self._index2word[index]

    def to_index(self, word):
        """Return index of the word

        Args:
            word (string): find word

        Returns:
            int: index of the given word
        """
        return self._word2index[word]
