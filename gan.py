import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchtext.utils as tutils
import json
import fhirtorch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

# Define the Generator model
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
       
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Custom Dataset class for loading FHIR data in NDJSON format
class FHIRDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, encoding='utf8', mode='r') as f:
            for line in f:
                json_obj = json.loads(line)
                self.data.append(json_obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print('converting fhir data for patient ' + str(index) + ' to tensor.....')
        return fhirtorch.fhir_to_tensor(self.data[index])
    
def train_gan(generator, discriminator, dataloader, num_epochs, device):
    # Check if the input data is empty
    if len(dataloader.dataset) == 0:
        raise ValueError("The input dataloader is empty. Please make sure it contains data.")

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (real_data, labels) in enumerate(dataloader):
            real_data = real_data.to(device)
            if labels !=None: labels.to(device)

            # Train discriminator with real data
            discriminator.zero_grad()
            real_labels = torch.ones(real_data.shape[0], 1, 1).to(device)  # Adjust the shape of real_labels
            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            real_loss.backward()
            real_cpu = real_data[0].to(device)

            # Train discriminator with generated data
            noise = torch.randn(real_data.shape[0], input_dim).to(device)  # Adjust the shape of the noise
            fake_data = generator(noise).detach()
            fake_labels = torch.zeros(real_data.shape[0], 1).to(device)  # Adjust the shape of fake_labels
            fake_output = discriminator(fake_data)
            fake_loss = criterion(fake_output, fake_labels)
            fake_loss.backward()
            discriminator_loss = real_loss + fake_loss
            discriminator_optimizer.step()

            # Clip discriminator's gradients
            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train generator
            generator.zero_grad()
            real_labels.fill_(1)  # Reset real_labels to 1s for the generator loss
            fake_output = discriminator(fake_data)
            generator_loss = criterion(fake_output.squeeze(), real_labels.squeeze())
            generator_loss.backward()
            generator_optimizer.step()

            if batch_idx % 100 == 0:  # Only print the stats on the batch
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch complete with [{real_data.shape[0]} passes], "  # Print the actual batch size
                    f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                    f"Generator Loss: {generator_loss.item():.4f}")
                # Print the generated text after each epoch
                generated_text = fake_data[0].detach().cpu().numpy()  # Convert tensor to numpy array
                print(f"Generated Text: {generated_text}")

    # Define a padding function that pads dimension 1 to max_dim1_size
def pad_dim1_to_max(tensor, max_dim1_size):
    padding = (0, max_dim1_size - tensor.size(1))
    return pad(tensor, padding)
    
def collate_fn(batch):
    # Check if each sample in your dataset is a tuple (data, label)
    try:
        data, labels = zip(*batch)
        labels = torch.stack(labels)
    except ValueError:
        # If not, handle data only
        data = batch
        labels = None
    #max_length = max([x.shape[0] for x in data])
    # Your data processing here. For example, if your data is a list of tensors of different lengths, 
    # you might want to pad them to the same length before stacking:
    #data = [F.pad(x, (0, max_length - x.shape[0])) for x in data]  # assuming data is 1D
    #data = pad_sequence([d.clone().detach().requires_grad_(True) for d in data], batch_first=True)
    #data = torch.stack(data)
    # Find the maximum length of data in dimension 1
    max_dim1_size = max([d.size(1) for d in data])
     # Apply the padding function to each data sample
    data = [pad_dim1_to_max(d, max_dim1_size) for d in data]
    # We also pad the sequences in the batch to the maximum length in dimension 0
    data = pad_sequence(data, batch_first=True)
    if labels !=None : labels = pad_sequence(labels, batch_first=True)
    return data, labels

# Set input dim
input_dim = 1  # Dimension of the random noise input for the generator
output_dim = 256  # Dimension of the generated output
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Entry point of the script
if __name__ == "__main__":

    # Set other training parameters
    lr = 0.0002  # Learning rate
    batch_size = 50 # Batch size for training
    num_epochs = 4000

    # Initialize generator and discriminator
    generator = Generator(input_dim, 1, output_dim).to(device)
    discriminator = Discriminator(input_dim, 256, output_dim).to(device)
    print('Loading the dataset.....')
    # Load the FHIR dataset
    dataset = FHIRDataset('data/Patient.ndjson')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=3)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Train the GAN
    train_gan(generator, discriminator, dataloader, num_epochs, device)

    # Save trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

class FHIRModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FHIRModel, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x