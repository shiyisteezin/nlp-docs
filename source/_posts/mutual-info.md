---
title: Mutual Information
date: 2024-08-03 8:45:14
categories:
  - Math
tags:
  - nlp-theories
---


### More on Mutual Information

Below is the code for WAPMI or the Weighted Average Point-wise Mutual Information. And this code measures the distance between two probabilities distributions.

It is a method used in computational linguistics to measure the strength of association between words in a given context, typically in the analysis of text data.

Imagine you have a box of different colored marbles, and you want to know which colors tend to appear together. WAPMI helps you figure out how often certain words (or colors) appear together more often than by random chance. It's like a smart way to understand word relationships in sentences!

```python
import math
import collections

def calculate_pmi(joint_prob, marginal_prob1, marginal_prob2):
    """
    Calculate the pointwise mutual information (PMI) between two words.

    :param joint_prob: The joint probability of the two words
    :param marginal_prob1: The marginal probability of the first word
    :param marginal_prob2: The marginal probability of the second word
    :return: The PMI score
    """
    if joint_prob == 0 or marginal_prob1 == 0 or marginal_prob2 == 0:
        return 0  # Avoid division by zero
    return math.log(joint_prob / (marginal_prob1 * marginal_prob2), 2)

def calculate_pmi_corpus_optimized(corpus):
    """
    Calculate the PMI scores for all pairs of words in a corpus.

    :param corpus: The corpus of text
    :return: A dictionary of PMI scores
    """
    word_counts = collections.defaultdict(int)
    cooccurrence_counts = collections.defaultdict(int)
    total_sentences = len(corpus)

    # Precompute word counts and co-occurrence counts
    for sentence in corpus:
        unique_words = set(sentence)  # Avoid counting duplicates within the same sentence
        for word in unique_words:
            word_counts[word] += 1
        for word1 in unique_words:
            for word2 in unique_words:
                if word1 != word2:
                    cooccurrence_counts[(word1, word2)] += 1

    # Calculate PMI scores
    pmi_scores = {}
    for (word1, word2), joint_count in cooccurrence_counts.items():
        joint_prob = joint_count / total_sentences
        marginal_prob1 = word_counts[word1] / total_sentences
        marginal_prob2 = word_counts[word2] / total_sentences
        pmi = calculate_pmi(joint_prob, marginal_prob1, marginal_prob2)
        pmi_scores[(word1, word2)] = pmi

    return pmi_scores

# Example usage
corpus = [
    ["this", "is", "a", "foo", "bar"],
    ["bar", "black", "sheep"],
    ["foo", "bar", "black", "sheep"],
    ["sheep", "bar", "black"]
]

pmi_scores = calculate_pmi_corpus_optimized(corpus)
print(pmi_scores)

```

A walkthrough of the code part by part.

### **Imports**

```python
import math
import collections
```

- **`import math`**: Imports the `math` module, which provides mathematical functions such as logarithms.
- **`import collections`**: Imports the `collections` module, which provides specialized container datatypes, like `defaultdict`.

### **Function Definitions**

#### **1. `calculate_pmi` Function**

```python
def calculate_pmi(word1, word2, joint_prob, marginal_prob1, marginal_prob2):
    """
    Calculate the point-wise mutual information (PMI) between two words.

    :param word1: The first word
    :param word2: The second word
    :param joint_prob: The joint probability of the two words
    :param marginal_prob1: The marginal probability of the first word
    :param marginal_prob2: The marginal probability of the second word
    :return: The PMI score
    """
    pmi = math.log(joint_prob / (marginal_prob1 * marginal_prob2), 2)
    return pmi
```

- **Purpose**: Calculates the Pointwise Mutual Information (PMI) score between two words.
- **Parameters**:
  - `word1` and `word2`: Words for which PMI is calculated.
  - `joint_prob`: Probability of both words appearing together.
  - `marginal_prob1`: Probability of `word1` appearing.
  - `marginal_prob2`: Probability of `word2` appearing.
- **`pmi` Calculation**:
  - `math.log(joint_prob / (marginal_prob1 * marginal_prob2), 2)`: Computes the logarithm (base 2) of the ratio of the joint probability to the product of the marginal probabilities.
- **Returns**: PMI score.

#### **2. `calculate_joint_prob` Function**

```python
def calculate_joint_prob(word1, word2, corpus):
    """
    Calculate the joint probability of two words in a corpus.

    :param word1: The first word
    :param word2: The second word
    :param corpus: The corpus of text
    :return: The joint probability
    """
    joint_count = 0
    for sentence in corpus:
        if word1 in sentence and word2 in sentence:
            joint_count += 1
    joint_prob = joint_count / len(corpus)
    return joint_prob
```

- **Purpose**: Calculates the joint probability of two words appearing together in the same sentence.
- **Parameters**:
  - `word1` and `word2`: Words to check.
  - `corpus`: List of sentences (each sentence is a list of words).
- **`joint_count`**: Counts how many sentences contain both `word1` and `word2`.
- **`joint_prob` Calculation**: Divides `joint_count` by the total number of sentences to get the joint probability.
- **Returns**: Joint probability of the two words.

#### **3. `calculate_marginal_prob` Function**

```python
def calculate_marginal_prob(word, corpus):
    """
    Calculate the marginal probability of a word in a corpus.

    :param word: The word
    :param corpus: The corpus of text
    :return: The marginal probability
    """
    word_count = 0
    for sentence in corpus:
        if word in sentence:
            word_count += 1
    marginal_prob = word_count / len(corpus)
    return marginal_prob
```

- **Purpose**: Calculates the marginal probability of a single word.
- **Parameters**:
  - `word`: The word for which probability is calculated.
  - `corpus`: List of sentences.
- **`word_count`**: Counts how many sentences contain `word`.
- **`marginal_prob` Calculation**: Divides `word_count` by the total number of sentences to get the marginal probability.
- **Returns**: Marginal probability of the word.

#### **4. `calculate_pmi_corpus` Function**

```python
def calculate_pmi_corpus_optimized(corpus):
    """
    Calculate the PMI scores for all pairs of words in a corpus.

    :param corpus: The corpus of text
    :return: A dictionary of PMI scores
    """
    word_counts = collections.defaultdict(int)
    cooccurrence_counts = collections.defaultdict(int)
    total_sentences = len(corpus)

    # Precompute word counts and co-occurrence counts
    for sentence in corpus:
        unique_words = set(sentence)  # Avoid counting duplicates within the same sentence
        for word in unique_words:
            word_counts[word] += 1
        for word1 in unique_words:
            for word2 in unique_words:
                if word1 != word2:
                    cooccurrence_counts[(word1, word2)] += 1

    # Calculate PMI scores
    pmi_scores = {}
    for (word1, word2), joint_count in cooccurrence_counts.items():
        joint_prob = joint_count / total_sentences
        marginal_prob1 = word_counts[word1] / total_sentences
        marginal_prob2 = word_counts[word2] / total_sentences
        pmi = calculate_pmi(joint_prob, marginal_prob1, marginal_prob2)
        pmi_scores[(word1, word2)] = pmi

    return pmi_scores
```

- **Purpose**: Calculates PMI scores for all pairs of words in the corpus.
- **Parameters**:
  - `corpus`: List of sentences.
- **`word_counts`**: A `defaultdict` to count occurrences of each word.
- **Count Words**: Iterates over each sentence to count occurrences of each word.
- **Compute PMI**:
  - Iterates over all pairs of words (excluding pairs where `word1` is the same as `word2`).
  - Calculates joint and marginal probabilities, then computes PMI for each pair.
- **Returns**: A dictionary of PMI scores for all word pairs.

### **Example Usage**

```python
corpus = [
    ["this", "is", "a", "foo", "bar"],
    ["bar", "black", "sheep"],
    ["foo", "bar", "black", "sheep"],
    ["sheep", "bar", "black"]
]

pmi_scores = calculate_pmi_corpus(corpus)
print(pmi_scores)
```

- **Purpose**: Runs the PMI calculations on a sample corpus and prints the PMI scores for all word pairs.

### **Summary**

- **`calculate_pmi`** computes PMI given probabilities.
- **`calculate_joint_prob`** finds how often two words appear together.
- **`calculate_marginal_prob`** finds how often one word appears.
- **`calculate_pmi_corpus`** calculates PMI for all word pairs in a corpus.

This code helps measure how strongly two words are associated compared to what you would expect by chance.

### Comparing Mutural Informaiton, WAPMI, VAE, and KL Divergence.


Let's break down Wappmi, mutual information, and how they relate to Variational Autoencoders (VAEs) and KL divergence in simple terms, like you're five.

### Mutual Information
Imagine you have two sets of toys, and you want to know how much one set tells you about the other. Mutual information is a way to measure how much knowing about one set of toys helps you predict what's in the other set. If you always find a red truck when you find a blue car, that's high mutual information.

### Wappmi (Weighted Average Prediction Pointwise Mutual Information)
Now, Wappmi takes this idea of mutual information and looks at words in sentences. It asks, "How often do these words appear together, and is it more than just by chance?" It's like seeing if certain toys always end up next to each other more than they would randomly.

### Variational Autoencoders (VAEs) and KL Divergence
Imagine you have a machine that tries to guess which toys you might pull out of a box based on previous toys. VAEs are like that machine—they learn patterns to predict what comes next.

The KL Divergence (Kullback-Leibler Divergence) is a way for the machine to measure how different its guesses (predictions) are from what actually happens. It helps the machine get better by adjusting its guesses to be more like what it observes.

### How They Relate
- **Mutual Information:** Helps understand how related different pieces of data are, like words in a sentence.
- **Wappmi:** Uses the idea of mutual information to find strong word pairings in text.
- **VAE:** Tries to model data (like sentences) to understand it better.
- **KL Divergence:** Helps the VAE improve its understanding by comparing its guesses to reality and adjusting accordingly.

### Example Code Walkthrough

Let’s imagine you have a simple code where you're trying to guess if two words often appear together:

```python
import collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

vocab = {'new': 2, 'news': 6, 'newer': 3}
pairs = get_stats(vocab)

print(pairs)
```

This code checks how often pairs of words (or symbols) appear together. If "new" and "news" show up together a lot, the program will tell you. This is similar to how Wappmi works.

In a VAE, you might use something like KL divergence to measure how far off your model’s guesses are from reality, then adjust it to improve. The model might then get better at understanding the relationships between words (like mutual information).

This approach helps in building smarter models that can understand and predict data more accurately.


### Simple code demonstration of each one of them


### 1. **Mutual Information**

**Concept**:
Mutual Information (MI) measures how much knowing one random variable reduces uncertainty about another. It quantifies the "shared information" between two variables.

**Code Example**:
```python
from sklearn.metrics import mutual_info_score

# Example data
X = [0, 0, 1, 1]
Y = [0, 1, 0, 1]

# Calculate mutual information
mi = mutual_info_score(X, Y)
print(f'Mutual Information: {mi}')
```

**Explanation**:
In this example, `mutual_info_score` from `sklearn` calculates the MI between two lists `X` and `Y`. MI quantifies how much knowing `X` helps predict `Y`.

### 2. **WAPPMI (Weighted Average Pointwise Mutual Information)**

**Concept**:
WAPPMI is used in NLP to find out how often words co-occur in a text corpus beyond what would be expected by chance. It gives more weight to frequently co-occurring pairs.

**Code Example**:
```python
import collections
import math

def pmi(x, y, corpus):
    px = corpus.count(x) / len(corpus)
    py = corpus.count(y) / len(corpus)
    pxy = corpus.count(x + ' ' + y) / len(corpus)
    return math.log(pxy / (px * py), 2)

corpus = "this is a simple corpus with some simple words in this simple text"
pairs = collections.defaultdict(int)

# Example calculation
words = corpus.split()
for i in range(len(words) - 1):
    pairs[(words[i], words[i + 1])] += 1

for pair, freq in pairs.items():
    print(f'PMI({pair}): {pmi(pair[0], pair[1], corpus)}')
```

**Explanation**:
The PMI function calculates the pointwise mutual information between word pairs in the `corpus`. WAPPMI extends this by weighting these values.

### 3. **Variational Autoencoder (VAE)**

**Concept**:
VAEs are used to generate data that’s similar to a given dataset. They work by learning a probabilistic model of the data and then sampling from it.

**Code Example**:
```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar

# Instantiate and run the model
vae = VAE(input_dim=784, hidden_dim=400, z_dim=20)
sample_input = torch.randn((64, 784))
output, mu, logvar = vae(sample_input)
```

**Explanation**:
The VAE class in this example defines an encoder and decoder. The encoder outputs a mean (`mu`) and log-variance (`logvar`) for a latent variable `z`. The decoder reconstructs the input data from `z`. The KL divergence between the latent distribution and a normal distribution helps regularize the model.

### 4. **KL Divergence**

**Concept**:
KL Divergence measures how one probability distribution diverges from a second, expected probability distribution. It’s used in VAEs to ensure that the latent variables follow a desired distribution (usually a Gaussian).

**Code Example**:
```python
import torch
import torch.nn.functional as F

# KL Divergence between the prior (standard normal) and the approximate posterior (q(z|x))
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

mu = torch.zeros(64, 20)
logvar = torch.zeros(64, 20)
kl = kl_divergence(mu, logvar)
print(f'KL Divergence: {kl}')
```

**Explanation**:
This code calculates the KL Divergence for a batch of latent variables with `mu` and `logvar` as parameters. It shows how much the approximate posterior (the learned distribution) diverges from the prior (standard normal distribution).

### **Comparison**:
- **Mutual Information** tells us how much knowing one variable helps us know another.
- **WAPPMI** applies this idea to words in text, showing how often they appear together.
- **VAEs** are generative models that learn to create data similar to a training set. They use **KL Divergence** to ensure the generated data follows a specific distribution.

**Overall**, these concepts are connected through their use in understanding and modeling the relationships within data, whether through measuring information (MI, WAPPMI) or generating and adjusting data distributions (VAEs, KL Divergence).
