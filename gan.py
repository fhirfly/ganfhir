import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchtext.utils as tutils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
import json
import fhirtorch
from torch.optim import Adam
import os
#import jpype

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.autograd.set_detect_anomaly(True)
# Generator
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim=1)  # Global Average Pooling
        return self.fc(out)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.sigmoid(self.fc(h_n[-1]))
        return out


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
        #print('converting fhir data for patient ' + str(index) + ' to tensor.....')
        return fhirtorch.fhir_to_tensor(self.data[index])
    
def train_gan(device, dataloader):
    # Check if the input data is empty
    if len(dataloader.dataset) == 0:
        raise ValueError("The input dataloader is empty. Please make sure it contains data.")
    
    # Hyperparameters
    batch_size = 16
    hidden_dim = 128
    lr = 0.001
    num_epochs = 1000

    # Start JVM
    #jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', "-Djava.class.path=/path/to/your/java/class")
    # Import the Java class
    #ValidationClass = jpype.JClass("your.package.ValidationClass")
    #validator = ValidationClass()

    # Training loop
    for epoch in range(num_epochs):
        for i, real_data in enumerate(dataloader):
           
            # Every nth batch (for example, every 10th), generate data for validation
            #if i % 10 == 0:
                #with torch.no_grad():
                    #fake_data = generator(noise)
                # Assuming your Java validation class has a method `isValid()` that returns True if data is valid
                #We would have to deserialize the tensor to a dataframe and then a fhir resource object
                #if validator.isValid(fake_data.tolist()):
                    # If data is valid, append to the dataloader
                    # Assuming your data loader dataset supports appending of new data
                    #dataloader.dataset.data.append(fake_data.tolist())
             
             # Prepare real data
            real_data = real_data.to(device)    
            # Update input_dim based on the size of the padded batch for each iteration
            input_dim = real_data.size(2)
            output_dim = real_data.size(2)

            # Create Generator and Discriminator instances inside the training loop with updated input_dim
            generator = Generator(input_dim, hidden_dim, output_dim).to(device)
            discriminator = Discriminator(input_dim, hidden_dim).to(device)
            
            # Define loss function and optimizers
            criterion = nn.BCELoss()
            optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)
            optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, weight_decay=1e-5)
            # Train Discriminator
            optimizer_D.zero_grad()

            # Generate fake data
            noise = torch.randn(batch_size, 1, input_dim).to(device)
            fake_data = generator(noise).to(device)

            # Discriminator loss for real and fake data
            real_output = discriminator(real_data)
           
            fake_output = discriminator(fake_data.detach())

            loss_D = criterion(real_output, torch.ones_like (real_output)) + \
                 criterion(fake_output, torch.zeros_like(fake_output))
    
            loss_D.backward()
            optimizer_D.step()

            # Clip discriminator's gradients
            #for p in discriminator.parameters():
               # p.data.clamp_(-0.01, 0.01)
            # Train Generator
            optimizer_G.zero_grad()

            # Generate fake data again
            noise = torch.randn(batch_size, 1, input_dim).to(device)
            fake_data = generator(noise).to(device)

            # Discriminator loss on generated data (to fool the discriminator)
            output = discriminator(fake_data)
            loss_G = criterion(output, torch.ones_like(output).to(device))
            loss_G.backward()
            optimizer_G.step()

            # Print loss values
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i+1}/{len(dataloader)}] Loss D: {loss_D:.4f}, Loss G: {loss_G:.4f}")    
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')
    #jpype.shutdownJVM()

def collate_fn(batch):
    # Assuming each item in the batch is a tensor representing a JSON data sample

    # Find the maximum sequence length in the batch
    max_length = max(data.size(1) for data in batch)

    # Initialize a tensor to hold the padded sequences
    padded_batch = torch.zeros(len(batch), batch[0].size(0), max_length, *batch[0].shape[2:])

    # Pad the sequences and fill the padded tensor
    for i, data in enumerate(batch):
        padding_size = max_length - data.size(1)
        padded_batch[i, :, :data.size(1), ...] = data

    return padded_batch

# Entry point of the script
if __name__ == "__main__":
    batch_size =32
    print('Loading the dataset.....')
    # Load the FHIR dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = FHIRDataset('data/Patient.ndjson')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=12)

    # Train the GAN
    train_gan(device, dataloader)