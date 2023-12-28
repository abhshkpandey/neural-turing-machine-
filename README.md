# Neural Turing Machine (NTM) Project

## Overview

This project implements a Neural Turing Machine (NTM) using PyTorch. The NTM is trained using the Learning without Forgetting (LwF) algorithm to perform a copy sequence task.

## Model Architecture

The NTM consists of the following components:

- Controller: LSTM cell
- Memory: Parameterized memory matrix
- Read and Write Heads: Linear layers for reading and writing to memory
- Output Layer: Linear layer for final output

## Task

The implemented task is the copy sequence task, where the NTM is trained to copy an input sequence to the output.

## File Structure

- `ntm.py`: Contains the NTM model definition.
- `lwf.py`: Implements the Learning without Forgetting (LwF) algorithm.
- `tasks.py`: Defines the copy sequence task and loss function.
- `main.py`: Example usage of the NTM and LwF algorithm.

## Usage

1. Install the required dependencies:

   ```bash
   pip install torch
   ```

2. Run the `main.py` script to train and test the NTM.

   ```bash
   python main.py
   ```

## Hyperparameters

- Input Size: 1
- Output Size: 1
- Controller Size: 100
- Memory Size: 128
- Memory Vector Size: 20
- Meta Iterations: 1000
- Meta Learning Rate: 0.001
- Sequence Length: 10

## Results

After training, the NTM is tested on a copy sequence task, and the predictions are printed.

## Acknowledgments

This project is based on the Neural Turing Machine (NTM) model and the Learning without Forgetting (LwF) algorithm. Credits to the original authors and contributors.

---

Feel free to customize this README file based on your project's specific details and requirements. Additionally, you may want to include information about data sources, additional functionalities, or any other relevant information.
