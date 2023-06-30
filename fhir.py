import torch
import torch.nn as nn

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Assume we've trained the GAN at this point, so we have a trained generator
# To generate text, we start with a random noise vector
noise = torch.randn(1, 1)
# Create the generator
netG = Generator(1,27)
# Load the generator's trained weights (not provided here)
netG.load_state_dict(torch.load("generator.pth"))
# Generate a sequence of characters
char_probs = netG(noise)
# Choose the most probable character at each position
_, generated_sequence = torch.max(char_probs, dim=1)
# This will be a sequence of numbers; you would need a mapping from numbers to characters
# to convert this to human-readable text
print(generated_sequence)

# Suppose you have the following list of characters that you used to train your GAN:
chars = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
         'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A',
         'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2',
         '3', '4', '5', '6', '7', '8', '9', '!', '?', '.', ',', '-', ':', ';',
         '(', ')', '[', ']', '{', '}', '"', "'", '@', '#', '$', '%', '^', '&',
         '*', '_', '+', '=', '|', '~', '<', '>', '/']

# Create a mapping from integers to characters:
int_to_char = dict(enumerate(chars))

# Let's say generated_sequence is your output tensor from the GAN.
generated_list = generated_sequence.tolist()

# Convert the sequence of integers to a sequence of characters
generated_text = ''.join(int_to_char[i] for i in generated_list)
print(generated_text)