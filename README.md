# SEC: Context-Aware Metric Learning for Efficient Emotion Recognition in Conversation
This repository presents a PyTorch implementation of meta-learning models applied on the DailyDialog dataset, performing emotions predictions. It consists in a Master thesis work for the Master of Data Science at the University of Luxembourg.  

This work has been accepted at two venues:
- [JEP-TALN 2024](https://jep-taln2024.sciencesconf.org/) "SEC : Contexte Émotionnel Phrastique pour la Reconnaissance Émotionnelle Efficiente dans la Conversation" (in French)
- [WASSA @ ACL 2024](https://workshop-wassa.github.io/) "SEC: Context-Aware Metric Learning for Efficient Emotion Recognition in Conversation"  

Citations of both papers will soon be available in the [Cite](#cite) section.

## Table of Contents

- [Data](#data)  
- [Models](#models)  
- [Usage](#usage)  
  - [Preprocessing](#preprocessing)
  - [Training and evaluation](#training-and-evaluation)  
  - [Qualitative insights](#qualitative-insights)
- [Cite](#cite)

<a name="data"></a>
## Data

DailyDialog dataset was introduced by [Li et al.](https://aclanthology.org/I17-1099/) in 2017 and consists in more than 12,000 generated dialogues intended to be representative of daily concerns. 
These dialogues are labelled with the 6 Ekman's emotions. More details about DailyDialog can be found [here](http://yanran.li/dailydialog.html).

<a name="models"></a>
## Models

In this work, we perform Emotion Recognition in Conversation (ERC) on DailyDialog conversation using meta-learning approaches. 
More precisely, this aim is to represent utterances from each dialog in its conversational context in order to predict accurate emotions. 
The model architecture used in the experiments in the [Siamese Networks](https://www.semanticscholar.org/paper/Siamese-Neural-Networks-for-One-Shot-Image-Koch/f216444d4f2959b4520c61d20003fa30a199670a). 
Another architecture called [MAML](https://arxiv.org/abs/1703.03400) (Model-Agnostic Meta-Learning) is yet to be explored.  

In the provided code for Siamese Networks, we implemented to models. 
The first one, referred to as **isolated utterance model**, is bound to perform emotion recognition on isolated utterances, without any contextual information at dialog level.
The second one, referred to as **contextual utterance model**, uses a BERT-based encoding that takes the conversational context into account, as BERT tokenization is performed at dialog level.
Below are illustrated the two different training pipeline of the aforementioned models:

<table align="center"><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img alt="Forwarding" src="https://github.com/B-Gendron/meta_dyda/assets/95307996/f81c8745-6ad9-4e6f-ad2e-d169ae9340f9" width="320">
    <br>
    <em style="color: grey">Isolated utterance model</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img alt="Routing" src="https://github.com/B-Gendron/meta_dyda/assets/95307996/d5504f67-c90f-4890-8e97-9304476c70e5" width="515">
    <br>
    <em style="color: grey">Contextual utterance model</em>
  </p> 
</td>
</tr></table>

<a name="usage"></a>
## Usage

The provided code allows to run data preprocessing, training and inference for both model. In order to achieve full reproducibility, please refer to the file `requirements.txt` by using:

```bash
pip install -r requirements.txt
```

To start with, clone the repository and go inside `SiameseNetworks` directory:

```bash
git clone https://github.com/B-Gendron/meta_dyda.git
cd meta_dyda
cd SiameseNetworks
```

It is possible to reproduce the following experiments:
- static utterance model training and evaluation
- contextual utterance model training and evaluation
- qualitative insights on static utterance model predictions
- qualitative insights on contextual utterance model predictions

### Preprocessing

Because the processed datasets are not available in the repository, it is necessary to perform preprocessing before training on data. Three different preprocessing pipelines are available:

- Conversation-aware dialog representations  
This experiment is the main contribution of this software. It is required to perform preprocessing on both isolated and contextual data, because isolated utterance data will be use te train the auxiliary emotion classifier, while the contextual data is used to perform the main experiment, meaning emotion recognition in conversational context.
```bash
python3 preprocessing.py -l s -t u
python3 preprocessing.py -l s -t c
```

- Isolated utterance representations  
In this case that accounts for a baseline, of course only isolated data preprocessing is needed.
```bash
python3 preprocessing.py -l u
```

### Training and evaluation

> Please note that, due to their sizes, the models are not provided in this repository. Therefore, one needs to train the model in order to perform inference on it. Furthermore, the qualitative experiments can be performed only if the models are already stored.

To run the experiments, one calls `main.py` script with some arguments: 
- `-m` for specifying the model (isolated or contextual)
- `-t` to run training and inference
- `-q` to run qualitative analysis
The experiments triggered when specifying `-t` and `-q` depends on the model value in `-m`. This is therefore a required argument.

Below are given all commands for each experiment so they can be directly copied and pasted. If you encounter any issue running the following code snippets, don't hesitate to display help first:

```bash
python3 main.py -h
```

Use the following command to train the model you want. The **isolated utterance model** is based on static utterance representations, therefore it does not take the conversational context into account. The selected model in this case is a 5-layer LSTM. On the other hand, the **contextual utterance model** takes the conversational context into account. The selected model is based on Sentence BERT. Here is the command for the isolated utterance model:

```bash
python3 main.py -m i -t
```

And for contextual utterance model:

```bash
python3 main.py -m s -t
```

### Qualitative insights

The following commands will proceed some qualitative analysis on parts of DailyDialog test set. Concretely, it will generate an output `.txt` file that contains dialogs along with the considered utterance and prediction compared to ground truth labels. For isolated utterance model:

```bash
python3 main.py -m i -q
```

And for contextual utterance model:

```bash
python3 main.py -m s -q
```
<a name="cite"></a>
## Cite

BibTeX citations to be added

### JEP-TALN 2024

### WASSA 2024 (ACL workshop)