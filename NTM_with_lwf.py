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

        # Additional components for MAS
        self.erase_linear = nn.Linear(controller_size, memory_vector_size)
        self.add_linear = nn.Linear(controller_size, memory_vector_size)

    def read_memory(self, weights):
        return torch.matmul(weights, self.memory)

    def write_memory(self, weights):
        erase_vector = torch.sigmoid(self.erase_linear(controller_state[0]))
        add_vector = torch.tanh(self.add_linear(controller_state[0]))
        self.memory = self.memory * (1 - torch.matmul(weights.unsqueeze(2), erase_vector.unsqueeze(1)))
        self.memory = self.memory + torch.matmul(weights.unsqueeze(2), add_vector.unsqueeze(1))

    def compute_importance(self, gradients):
        return torch.abs(gradients)

    def forward(self, x, prev_state):
        # (same as the original code)

# Function to generate a copy sequence task
def generate_copy_sequence_task(seq_length):
    input_sequence = torch.randint(0, 2, (seq_length,)).float()
    target_sequence = input_sequence.clone()
    return input_sequence, target_sequence

# Copy sequence task loss with MAS
def copy_sequence_task_loss(output, target, memory_state, mas_lambda=0.01):
    loss = nn.MSELoss()(torch.sigmoid(output), target)

    # Regularization term to prevent memory write interference (LwF)
    write_weights = memory_state['write_weights']
    loss += torch.sum(torch.abs(write_weights - 1.0 / write_weights.size(1)))

    # MAS regularization term
    mas_importance = task_ntm.compute_importance(torch.autograd.grad(loss, task_ntm.parameters(), retain_graph=True))
    mas_loss = mas_lambda * torch.sum(mas_importance)
    loss += mas_loss

    return loss

# Learning without Forgetting (LwF) algorithm with MAS
def apply_lwf_and_mas(original_model, task_model, criterion, optimizer, sequence_length, meta_iterations, meta_lr, beta=0.5, mas_lambda=0.01):
    for meta_iteration in range(meta_iterations):
        input_sequence, target_sequence = generate_copy_sequence_task(sequence_length)

        for t in range(sequence_length):
            input_t = input_sequence[t].view(1, 1)
            target_t = target_sequence[t].view(1, 1)

            # Forward pass with the original model
            output, original_memory_state = original_model(input_t, (None, {'read_weights': torch.zeros(1, 1, 1)}))
            original_loss = copy_sequence_task_loss(output, target_t, original_memory_state, mas_lambda)

            # Forward pass with the task-specific model
            task_output, task_memory_state = task_model(input_t, (None, {'read_weights': torch.zeros(1, 1, 1)}))
            task_loss = criterion(task_output, target_t)

            # LwF loss with MAS
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

# Apply LwF and MAS
task_ntm = apply_lwf_and_mas(original_ntm, task_ntm, criterion, optimizer, sequence_length, meta_iterations, meta_lr)

# Test the trained NTM
test_input_sequence, _ = generate_copy_sequence_task(sequence_length)
for t in range(sequence_length):
    input_t = test_input_sequence[t].view(1, 1)  # Assuming input_size is 1
    output, _ = task_ntm(input_t, (None, {'read_weights': torch.zeros(1, 1, 1)}))
    prediction = torch.sigmoid(output).round().int().item()
    print(f"Input: {int(test_input_sequence[t])}, Prediction: {prediction}")
