# Torch utils
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_dtype(torch.float32)

# From other scripts
from utils import *


class SiameseNetworkLinear(nn.Module):
    '''
        A siamese network model for multi-class classification on text data.
        The model structure here is a MLP with a number of hidden layers specified by the user.
    '''
    def __init__(self, n_layers, input_dim, hidden_dim):
        super(SiameseNetworkLinear, self).__init__()

        self.n_layers = n_layers

        self.hidden_layer = torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True)

        self.relu = nn.ReLU()

        self.drop = nn.Dropout(p=0.2)

        self.linear = nn.Linear(input_dim*hidden_dim, hidden_dim) 

    def forward_once(self, x):
        '''
            Realize the forward path for one model.

            @param x (tensor): input tensor

            @return output (tensor): output tensor of the model after all the model layers
        '''
        # First hidden layer to instantiate the output variable
        output = self.hidden_layer(x)
        output = self.relu(output)
        output = self.drop(output)

        # Add the required number of hidden layers
        for _ in range(self.n_layers-1):
            output = self.hidden_layer(output)
            output = self.relu(output)
            output = self.drop(output)

        # Last linear + ReLU
        output = torch.reshape(output, (output.size()[0], 20*300))
        output = self.linear(output)
        output = self.relu(output)

        return output
    
    def forward(self, input1, input2, input3):
        '''
            Realize all the forward paths for all the inputs.

            @param input1 (tensor): first input (anchor in our case)
            @param input2 (tensor): second input (positive in our case)
            @param input3 (tensor): third input (negative in our case)

            @return A (tensor): the model output for the anchor sample
            @return P (tensor): the model output for the positive sample
            @return N (tensor): the model output for the negative sample
        '''
        A = self.forward_once(input1)
        P = self.forward_once(input2)
        N = self.forward_once(input3)

        return A, P, N
    

class SiameseNetworkLSTM(nn.Module):
    '''
        A siamese network model for multi-class classification on text data.
        The model structure here is LSTM with a number of hidden layers specified by the user.
    '''
    def __init__(self, n_layers, input_dim, hidden_dim):
        super(SiameseNetworkLSTM, self).__init__()

        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim, 
                            num_layers=self.n_layers, 
                            dropout=0.3,
                            batch_first=True)

        self.linear = nn.Linear(input_dim*hidden_dim, hidden_dim) 

        self.relu = nn.ReLU()

    def forward_once(self, x):
        '''
            Realize the forward path for one model.

            @param x (tensor): input tensor

            @return output (tensor): output tensor of the model after the LSTM layers
        '''
        # LSTM layers
        output_lstm, _ = self.lstm(x)

        # Last linear + ReLU
        output = torch.reshape(output_lstm, (output_lstm.size()[0], 20*300))
        output = self.linear(output)
        output = self.relu(output)

        return output
    
    def forward(self, input1, input2, input3):
        '''
            Realize all the forward paths for all the inputs.

            @param input1 (tensor): first input (anchor in our case)
            @param input2 (tensor): second input (positive in our case)
            @param input3 (tensor): third input (negative in our case)

            @return A (tensor): the model output for the anchor sample
            @return P (tensor): the model output for the positive sample
            @return N (tensor): the model output for the negative sample
        '''
        A = self.forward_once(input1)
        P = self.forward_once(input2)
        N = self.forward_once(input3)

        return A, P, N
    

class EmotionalLabelsClassifier(nn.Module):
    '''
        A multiclass classifier used for transformer encoder pre-training
    '''
    def __init__(self, args, device, max_len, n_layers=1, nb_classes=7):
        super(EmotionalLabelsClassifier, self).__init__()

        # constants
        self.hidden_size = max_len

        # TRE
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=24, batch_first=True, dropout=0.3)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=16, batch_first=True, dropout=0.3) # for roberta
        self.tfencoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # classification-related layers
        self.classification_layer = nn.Linear(in_features=self.hidden_size, out_features=nb_classes, bias=True)
        self.softmax = nn.Softmax(dim=-1)

        # utils
        self.args = args
        self.device = device

    def forward(self, embedding):
        '''
            Realize the forward path for the encoder.

            @param embedding (tensor): input tensor

            @return output (tensor): output tensor of the model after the desired model layers depending on the model status
        '''
        x = embedding.to(torch.float32)
        output = self.tfencoder(x)
        output = self.classification_layer(output)
        classes_proba = self.softmax(output)
        return classes_proba


class SentenceEmCoBERT(nn.Module):
    '''
        Model to evaluate emotion classifier on triplets

        A siamese network model for multi-class classification on text data.
        This model assumes that the data has been previously tokenized using Sentence BERT.
    '''
    def __init__(self, args, device, n_layers, sm, max_len=384, nb_classes=7):
        super(SentenceEmCoBERT, self).__init__()

        # constants
        self.hidden_size = 15 * max_len # utterance_limit * max_len
        self.linear_hidden_size = max_len

        # tre layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=24, batch_first=True, dropout=0.3)
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # retrieve pre-trained classifier
        self.classifier = EmotionalLabelsClassifier(args, device, max_len, n_layers, nb_classes)
        self.classifier.load_state_dict(load_emotions_classifier(sm))

        # other parameters
        self.args = args
        self.nb_classes = nb_classes 
        self.device = device

        # utterances representations
        self.utterances = None

        # train or inference status
        self.status = None


    def get_all_utterances(self):
        '''
            A method to retrieve the utterances representations generated by BERT model. These correspond to the formatted output to which is applied a linear layer in order to come up with a flat utterance representation.
        '''
        return self.utterances
    
    def set_status_to_triplet(self):
        '''
            Set the model status to train, which means that all the layers of the model will be applied.
        '''
        self.status = 'triplet'

    def set_status_to_encoder(self):
        '''
            Set the model status to pretrain, meaning we only consider the transformer encoder layers.
        '''
        self.status = 'encoder'

    def get_status(self):
        '''
            For debugging purposes.
        '''
        return self.status
    

    def forward(self, embedding):
        '''
            Realize the forward path for the encoder.

            @param embedding (tensor): input tensor

            @return output (tensor): output tensor of the model after the desired model layers depending on the model status
        '''
        if self.status == 'encoder':
            x = embedding.to(torch.float32)
            tfe_x = self.attention(x)
            # tfe_dialogues = 0.75*x + 0.25*tfe_x
            tfe_dialogues = tfe_x

            splits = []
            for dialogue in tfe_dialogues:
                # split for each element of the batch according to indexes
                start_indexes = [self.linear_hidden_size] + [k*(self.linear_hidden_size) for k in range(2, len(dialogue)//self.linear_hidden_size)]
                dialogue_splits = torch.tensor_split(dialogue, start_indexes)
                splits.append(torch.stack(dialogue_splits))

            # flatten all the splits to get rid of the dialog dimension
            all_utterances_tensor = torch.cat(splits)
            self.utterances = all_utterances_tensor

            return self.utterances

        elif self.status == 'triplet':
            utterance = torch.unsqueeze(embedding, 0) 
            # if in triplet mode, apply classification layer and softmax
            classes_proba = self.classifier(utterance)
            return classes_proba