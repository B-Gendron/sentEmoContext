# Torch utils
import torch
from torch.utils.data import Dataset
torch.set_default_dtype(torch.float32)

# General purposes modules
import numpy as np
import random
random.seed(42)
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

# From others files of this repo
from utils import *


class UtteranceVectorsEmotionDataset(torch.utils.data.Dataset):
    """
    A dataset class to build triplets of isolated utterance representations, according to the triplet loss setting.
    
    Attributes:
        args (dict): Arguments or configuration settings.
        data (dataset): List of utterance data.
        grouped_indexes (dict): Dictionary mapping class labels to lists of indexes of utterances.
        length (int): Total number of utterances in the dataset.
    """

    def __init__(self, data, args):
        """
        Initialize the UtteranceVectorsEmotionDataset.

        Args:
            data (list): The dataset containing utterance information.
            args (dict): Additional arguments or configuration settings.
        """
        self.args = args
        self._data = data
        self.indexes_by_class()
        self.length = len(self._data)

    @property
    def data(self):
        """
        Access the dataset.

        Returns:
            dataset: The dataset containing utterance information.
        """
        return self._data

    def __len__(self):
        """
        Get the number of utterances in the dataset.

        Returns:
            int: The number of utterances.
        """
        return self.length
    
    def indexes_by_class(self):
        """
        Classify all the utterances in the data based on their class.
        This function builds a dictionary where the data is sorted by class labels.
        """
        # get all the labels
        all_labels = np.array(deepcopy(self._data["label"])) # deepcopy instead of clone because data format is list here
        self.grouped_indexes = {i: np.where(all_labels == i)[0] for i in range(7)}

    def __getitem__(self, idx):
        """
        Get a triplet of utterances to apply the siamese network model on.

        The 3 elements of the triplet are chosen randomly through the following process:
        - A class is randomly picked to be the anchor class.
        - Two samples from this class are randomly picked to be the anchor and the positive samples.
        - Another class is picked (different from the first one), and an entry is chosen to be the negative sample.

        Args:
            idx (int): Index (not used in this method but required by PyTorch Dataset class).

        Returns:
            item (dict): A dictionary containing the triplet of utterances (anchor, positive, and negative) and their associated labels.
        """
        # choose a random class for anchor and positive
        anchor_class = random.randint(0, 6)
        # choose a distinct random class for negative
        negative_class = random.choice([c for c in range(7) if c != anchor_class])

        # pick random indexes in the grouped utterances from the selected classes
        index_anchor = random.choice(self.grouped_indexes[anchor_class])
        index_positive = random.choice(self.grouped_indexes[anchor_class])
        while index_positive == index_anchor:
            index_positive = random.choice(self.grouped_indexes[anchor_class])
        index_negative = random.choice(self.grouped_indexes[negative_class])

        anchor = torch.tensor(self._data[int(index_anchor)]["vectors"], dtype=torch.float32)
        positive = torch.tensor(self._data[int(index_positive)]["vectors"], dtype=torch.float32)
        negative = torch.tensor(self._data[int(index_negative)]["vectors"], dtype=torch.float32)

        item = {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "label": torch.tensor([anchor_class, anchor_class, negative_class], dtype=torch.float32)
        }
        return item
    
    
class SentenceEmotionDatasetBERT(torch.utils.data.Dataset):
    '''
        This class considers a preprocessing based on mapping utterances to their Sentence BERT embeddings. It prepares the use of a Transformer-encoder-based architecture to be found in models.py. 
    '''
    def __init__(self, data, args):
        self._data = data
        self.spreading = args['spreading']
        self.length = len(self._data)

    @property
    def data(self):
        return self._data
    
    def __len__(self):
        if len(self._data) < 2000:
            return 1000
        else:
            # return 1000
            return self.length

    def __getitem__(self, idx):
        if self.spreading:
            label_vector = emotional_label_spreading(self._data[idx]["label"])
            item = {
                "label" : torch.tensor(label_vector, dtype=torch.float32),
                "embedding" : torch.tensor(self._data[idx]["embedding"], dtype=torch.float32)
            }

        else:
            item = {
                "label" : torch.tensor(self._data[idx]["label"], dtype=torch.float32),
                "embedding" : torch.tensor(self._data[idx]["embedding"], dtype=torch.float32)
            }

        return item
    

class SentenceEmotionDatasetUtterances(torch.utils.data.Dataset):
    '''
        This class considers a preprocessing based on mapping utterances to their Sentence BERT embeddings. It prepares the use of a Transformer-encoder-based architecture to be found in models.py. In order to get a Transformer encoder input dimension similar to the one expected in Sentence-BERT model, this dataset class provides 15 utterances along with their associated labels for each data sample.
    '''
    def __init__(self, data, args, nb_indexes=15):
        """
        Initialize the SentenceEmotionDatasetBERT class.

        Args:
            data (dataset): The dataset containing utterance information.
            args (dict): Additional arguments or configuration settings.
        """
        self.args = args
        self.bsize = args['train_bsize']
        self._data = data
        self.nb_indexes = nb_indexes
        self.length = len(self._data)

    @property
    def data(self):
        """
        Access the dataset.

        Returns:
            dataset: The dataset containing utterance information.
        """
        return self._data
    
    def __len__(self):
        """
        Get the number of utterances in the dataset.

        If the dataset contains fewer than 2000 utterances, returns 1000.
        Otherwise, returns the actual length of the dataset.

        Returns:
            int: The number of utterances.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Depending on the spreading flag, this method returns either the original label or a unique label spread across the emotion label vector.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            item (dict): A dictionary containing the label and the Sentence BERT embedding of the utterance.
        """
        item = {
        "label" : np.array(self._data[idx]["label"]),
        "embedding" : np.array(self._data[idx]["text"])
        }

        return item