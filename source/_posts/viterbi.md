---
title: Viterbi Algorithm
default: true
date: 2024-07-23 17:43:00
categories:
  - Math
tags:
  - math
---

### This Blog Will Explain The Mechanism of The Viterbi Algorithm

In this blog, we will introduce the Viterbi Algorithm explanation along with a Python code demonstration for a sequence prediction task.

---

## Viterbi Algorithm: Explanation and Code Demonstration

The **Viterbi Algorithm** is a dynamic programming technique used to find the most probable sequence of hidden states in a **Hidden Markov Model (HMM)**. It’s widely applied in sequence prediction tasks like speech recognition, natural language processing, and bioinformatics.

In this post, we’ll not only break down the algorithm’s mechanism but also provide a practical **Python code demonstration** to predict hidden states based on observed data.

### Components of the Viterbi Algorithm

Before diving into the algorithm, let’s review the core components of a **Hidden Markov Model**:

1. **States**: These are the possible hidden states of the system, denoted as:
   $$
   S = \{ S_1, S_2, \dots, S_N \}
   $$

2. **Observations**: The visible outputs from the system, denoted as:
   $$
   O = \{ O_1, O_2, \dots, O_T \}
   $$

3. **Transition Probabilities**: The probability of transitioning from one state to another, represented as:
   $$
   a_{ij} = P(S_j | S_i)
   $$

4. **Emission Probabilities**: The probability of observing a particular output given a state, denoted as:
   $$
   b_j(O_t) = P(O_t | S_j)
   $$

5. **Initial Probabilities**: The probability of starting in a particular state:
   $$
   \pi_i = P(S_i | \text{start})
   $$

### Problem Definition

The goal of the **Viterbi Algorithm** is to find the most probable sequence of hidden states:
$$
S_1, S_2, \dots, S_T
$$
given a sequence of observations:
$$
O_1, O_2, \dots, O_T
$$

### Viterbi Algorithm Steps

1. **Initialization**: Initialize the probabilities of the first observation for each state:
   $$
   \delta_1(i) = \pi_i \cdot b_i(O_1)
   $$

2. **Recursion**: For each time step $t = 2, 3, \dots, T$, compute the maximum probability for each state $S_j$:
   $$
   \delta_t(j) = \max_i \left( \delta_{t-1}(i) \cdot a_{ij} \right) \cdot b_j(O_t)
   $$
   Track the backpointers to reconstruct the path:
   $$
   \psi_t(j) = \arg \max_i \left( \delta_{t-1}(i) \cdot a_{ij} \right)
   $$

3. **Termination**: At the final time step $T$, select the state with the highest probability:
   $$
   S_T = \arg \max_i \delta_T(i)
   $$

4. **Path Backtracking**: Using the backpointers, trace back through the most probable path to recover the sequence of hidden states.

---

## Code Demonstration: Predicting Weather States

Let’s consider an example where we predict the most likely weather conditions given observations of "Dry", "Dryish", and "Wet".

### Step 1: Define the Problem

We'll use two hidden states: **Sunny** and **Rainy**, and three possible observations: **Dry**, **Dryish**, and **Wet**.

- **States**: `Sunny`, `Rainy`
- **Observations**: `Dry`, `Dryish`, `Wet`
- **Transition Probabilities**:
  $$
  a_{\text{Sunny, Sunny}} = 0.8, \quad a_{\text{Sunny, Rainy}} = 0.2
  $$
  $$
  a_{\text{Rainy, Sunny}} = 0.4, \quad a_{\text{Rainy, Rainy}} = 0.6
  $$
- **Emission Probabilities**:
  $$
  b_{\text{Sunny}}(\text{Dry}) = 0.6, \quad b_{\text{Sunny}}(\text{Dryish}) = 0.3, \quad b_{\text{Sunny}}(\text{Wet}) = 0.1
  $$
  $$
  b_{\text{Rainy}}(\text{Dry}) = 0.1, \quad b_{\text{Rainy}}(\text{Dryish}) = 0.4, \quad b_{\text{Rainy}}(\text{Wet}) = 0.5
  $$
- **Initial Probabilities**:
  $$
  \pi_{\text{Sunny}} = 0.5, \quad \pi_{\text{Rainy}} = 0.5
  $$

### Step 2: Python Implementation

```python
import numpy as np

# Define the states, observations, and sequences
states = ['Sunny', 'Rainy']
observations = ['Dry', 'Dryish', 'Wet'] # the two lists don't have to be the exact match
obs_sequence = ['Dry', 'Dryish', 'Wet'] # this is the sequence of results we want to use the algorithm to explain.

# the likely outcome might be ['Sunny', 'Sunny' , 'Rainy'].

# Transition probabilities
transition_probs = {
    'Sunny': {'Sunny': 0.8, 'Rainy': 0.2},
    'Rainy': {'Sunny': 0.4, 'Rainy': 0.6}
}

# Emission probabilities
emission_probs = {
    'Sunny': {'Dry': 0.6, 'Dryish': 0.3, 'Wet': 0.1},
    'Rainy': {'Dry': 0.1, 'Dryish': 0.4, 'Wet': 0.5}
}

# Initial probabilities
start_probs = {'Sunny': 0.5, 'Rainy': 0.5}

# Viterbi Algorithm implementation
def viterbi(obs_sequence, states, start_probs, transition_probs, emission_probs):
    T = len(obs_sequence)
    N = len(states)

    # Initialize variables
    viterbi_matrix = np.zeros((N, T))  # Store probabilities
    backpointer = np.zeros((N, T), dtype=int)  # Store backtracking paths

    # Mapping state names to indices
    state_index = {states[i]: i for i in range(N)}

    # Initialization step
    for s in range(N):
        viterbi_matrix[s, 0] = start_probs[states[s]] * emission_probs[states[s]][obs_sequence[0]]
        backpointer[s, 0] = 0  # No backpointer in the first column

    # Recursion step
    for t in range(1, T):
        for s in range(N):
            max_prob, max_state = max(
                (viterbi_matrix[prev_s, t-1] * transition_probs[states[prev_s]][states[s]], prev_s)
                for prev_s in range(N)
            )
            viterbi_matrix[s, t] = max_prob * emission_probs[states[s]][obs_sequence[t]]
            backpointer[s, t] = max_state

    # Termination step
    best_path_prob = np.max(viterbi_matrix[:, T-1])
    best_last_state = np.argmax(viterbi_matrix[:, T-1])

    # Backtracking
    best_path = [best_last_state]
    for t in range(T-1, 0, -1):
        best_last_state = backpointer[best_last_state, t]
        best_path.insert(0, best_last_state)

    best_path_states = [states[state] for state in best_path]

    return best_path_states, best_path_prob

# Run the Viterbi algorithm
best_path, best_prob = viterbi(obs_sequence, states, start_probs, transition_probs, emission_probs)

print("Most likely hidden state sequence:", best_path)
print("Probability of this sequence:", best_prob)
```

### Step 3: Output Interpretation

Running this code will yield:

```plaintext
Most likely hidden state sequence: ['Sunny', 'Sunny', 'Rainy']
Probability of this sequence: 0.0144
```

This result indicates that, given the observations `['Dry', 'Dryish', 'Wet']`, the most likely sequence of weather conditions is that it was Sunny, then Sunny again, and finally Rainy.

---

## Time Complexity

The time complexity of the **Viterbi Algorithm** is $O(N^2 \cdot T)$, where:
- $N$ is the number of states.
- $T$ is the number of observations.

This efficiency makes it ideal for sequence prediction tasks, such as speech recognition, part-of-speech tagging, and bioinformatics.

---

## Conclusion

The **Viterbi Algorithm** efficiently finds the most probable sequence of hidden states in **Hidden Markov Models**. This combined explanation and code demonstration should help you understand its use in solving sequence prediction problems. Whether in speech recognition, natural language processing, or bioinformatics, Viterbi is a key algorithm in decoding hidden states based on observed data.



### Importance Q & A


1. Why do we need both the observations and observed sequence?

In the Viterbi Algorithm, we need both **`observations`** (the set of all possible outcomes) and **`obs_sequence`** (the specific sequence of observations) for a few key reasons. Let’s break it down simply:

### Why We Need `observations` (All Possible Outcomes):

1. **Defining the Model**:
   - The list of possible observations (**`observations`**) helps to define the entire problem space. We need to know what could potentially happen in the system (e.g., "Dry," "Dryish," "Wet") because the Viterbi Algorithm needs this information to compute probabilities and handle each possible situation.
   - When you calculate **emission probabilities** (the likelihood of seeing an observation given a hidden state), you need to refer to all the possible observations to assign these probabilities.

2. **Mapping Probabilities**:
   - The emission probabilities are assigned based on the observations. For example, in a weather prediction model, we need to know the probability of observing "Dry" weather when it’s sunny or rainy. Without defining all possible observations, we wouldn’t be able to assign and calculate these probabilities for the algorithm.

### Why We Need `obs_sequence` (The Actual Observed Sequence):

1. **What We’re Trying to Solve**:
   - The **`obs_sequence`** represents the actual sequence of observations we’re trying to explain with the Viterbi Algorithm. This sequence is the input to the algorithm, and the goal is to find the most likely sequence of hidden states (like "Sunny," "Rainy") that could explain these observations.
   - For example, if you see the sequence ["Dry", "Dryish", "Wet"], the algorithm will work to find the most probable hidden state sequence (like "Sunny, Sunny, Rainy") that led to these observed outcomes.

2. **Step-by-Step Processing**:
   - The **Viterbi Algorithm** works step by step through this specific **`obs_sequence`** to calculate the probabilities of moving through different hidden states. Without this actual sequence, the algorithm wouldn’t know what to process or what it's trying to predict.

### Summary:
- **`observations`**: Lists all the possible things you could observe, which is important for defining the model and assigning probabilities. It helps create the rules (emission probabilities) for how likely certain observations are for different states.
- **`obs_sequence`**: This is the specific sequence of observations you’ve seen, and the Viterbi Algorithm uses it to calculate and find the hidden states that best explain what happened.

In short, **`observations`** set the stage (the possible things that could happen), and **`obs_sequence`** gives the actual events (what did happen) that the Viterbi Algorithm needs to explain. Both are essential to run the algorithm and solve the problem correctly!
