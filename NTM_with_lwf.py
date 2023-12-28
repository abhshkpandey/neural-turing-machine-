import torch
import torch.nn as nn
import torch.optim as optim

class NTM(nn.Module):
    def __init__(self, input_size, output_size, controller_size, memory_size, memory_vector_size):
        super(NTM, self).__init__()

        # Define components of the NTM
        self.controller = nn.LSTMCell(input_size + memory_vector_size, controller_size)
        self.memory = nn.Parameter(torch.randn(1, memory_size, memory_vector_size))
        self.read_head = nn.Linear(controller_size, memory_vector_size)
        self.write_head = nn.Linear(controller_size, memory_vector_size)
        self.output = nn.Linear(controller_size + memory_vector_size, output_size)

    def read_memory(self, weights):
        return torch.matmul(weights, self.memory)

    def write_memory(self, weights, erase_vector, add_vector):
        self.memory = self.memory * (1 - torch.matmul(weights.unsqueeze(2), erase_vector.unsqueeze(1)))
        self.memory = self.memory + torch.matmul(weights.unsqueeze(2), add_vector.unsqueeze(1))

    def forward(self, x, prev_state):
        prev_controller_state, prev_memory_state = prev_state

        # Concatenate input with read from memory
        input_plus_read = torch.cat([x, self.read_memory(prev_memory_state['read_weights'])], dim=1)

        # Controller LSTM
        controller_state = self.controller(input_plus_read, prev_controller_state)

        # Read from memory
        read_weights = torch.softmax(self.read_head(controller_state[0]), dim=1)

        # Write to memory
        write_weights = torch.softmax(self.write_head(controller_state[0]), dim=1)
        erase_vector = torch.sigmoid(self.write_head(controller_state[0]))
        add_vector = torch.tanh(self.write_head(controller_state[0]))
        self.write_memory(write_weights, erase_vector, add_vector)

        # Concatenate controller output with read from memory
        output = torch.cat([controller_state[0], self.read_memory(read_weights)], dim=1)
        output = self.output(output)

        # Package the state for the next iteration
        memory_state = {'read_weights': read_weights, 'write_weights': write_weights}
        new_state = (controller_state, memory_state)

        return output, new_state

def generate_copy_sequence_task(seq_length):
    input_sequence = torch.randint(0, 2, (seq_length,)).float()
    target_sequence = input_sequence.clone()
    return input_sequence, target_sequence


def copy_sequence_task_loss(output, target, memory_state):
    # MSE loss for the copy sequence task
    loss = nn.MSELoss()(output, target)

    # Regularization term to prevent memory write interference
    write_weights = memory_state['write_weights']
    loss += torch.sum(torch.abs(write_weights - 1.0 / write_weights.size(1)))

    return loss

# Learning without Forgetting (LwF) algorithm
def apply_lwf(original_model, task_model, criterion, optimizer, sequence_length, meta_iterations, meta_lr, beta=0.5):
    for meta_iteration in range(meta_iterations):
        # Generate a meta-learning task
        input_sequence, target_sequence = generate_copy_sequence_task(sequence_length)

        # Meta-update
        for t in range(sequence_length):
            input_t = input_sequence[t].view(1, 1)
            target_t = target_sequence[t].view(1, 1)

            # Forward pass with the original model
            output, original_memory_state = original_model(input_t, (None, {'read_weights': torch.zeros(1, 1, 1)}))
            original_loss = copy_sequence_task_loss(output, target_t, original_memory_state)

            # Forward pass with the task-specific model
            task_output, task_memory_state = task_model(input_t, (None, {'read_weights': torch.zeros(1, 1, 1)}))
            task_loss = criterion(task_output, target_t)

            # LwF loss
            lwf_loss = beta * original_loss + (1 - beta) * task_loss

            optimizer.zero_grad()
            lwf_loss.backward()
            optimizer.step()
           


    return task_model

# Define NTM hyperparameters
input_size = 1
output_size = 1
controller_size = 100
memory_size = 128
memory_vector_size = 20
meta_iterations = 1000
meta_lr = 0.001
sequence_length = 10

# Create original NTM and task-specific NTM
original_ntm = NTM(input_size, output_size, controller_size, memory_size, memory_vector_size)
task_ntm = NTM(input_size, output_size, controller_size, memory_size, memory_vector_size)
task_ntm.load_state_dict(original_ntm.state_dict())  # Initialize task-specific NTM with original NTM's parameters

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(task_ntm.parameters(), lr=meta_lr)

# Apply LwF
task_ntm = apply_lwf(original_ntm, task_ntm, criterion, optimizer, sequence_length, meta_iterations, meta_lr)

# Test the trained NTM
test_input_sequence, _ = generate_copy_sequence_task(sequence_length)
for t in range(sequence_length):
    input_t = test_input_sequence[t].view(1, 1)  # Assuming input_size is 1
    output, _ = task_ntm(input_t, (None, {'read_weights': torch.zeros(1, 1, 1)}))
    prediction = torch.sigmoid(output).round().int().item()
    print(f"Input: {int(test_input_sequence[t])}, Prediction: {prediction}")
