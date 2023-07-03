import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchtext.utils as tutils
import json

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

# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
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
        fhir_profile_resource = resource_from_profile(self.data[index].get('resourceType'), fhir_profiles_resources_json)
        return fhir_resource_to_tensor(self.data[index], self.data[index].get('resourceType'), fhir_profile_resource, fhir_value_set)  # Assuming each line is a tensor

def resource_from_profile(fhir_resource, fhir_profiles_resources):
    for i, resource in enumerate(fhir_profiles_resources['entry']):
        if resource['resource'].get('id')==fhir_resource:
            return resource

#Convert FHIR to Tensor
def fhir_resource_to_tensor(fhir_resource_json, fhir_resource, fhir_profile_resource, fhir_value_set):
    # Parse the FHIR resource
    #fhir_resource = fhir_types.fhir_resource(fhir_resource)

    # Get the list of elements from the StructureDefinition for the current resource type
    
    elements = fhir_profile_resource['resource'].get('differential')['element'][1:]  #the first elelemt is the resource itself, so skip that
    # Create an empty tensor with the shape of the elements
    tensor_shape = (1, len(elements))
    output_dim = len(elements)
    tensor = torch.empty(tensor_shape)      
    # Iterate through the elements and populate the tensor
    for i, element in enumerate(elements):
        fhir_element = element['id'].split('.')[1]
        value = fhir_resource_json.get(fhir_element, None)
        if value is not None:
            if isinstance(value, list):
                tensor[0,i] = torch.tensor(len(value))
            elif element.get('type')[0].get('code') == 'date':
                tensor[0,i] = torch.tensor(len(date_to_one_hot(value)))
            elif element.get('type')[0].get('code') == 'CodeableConcept':
                tensor[0,i] = torch.tensor(len(value))
            else:  #if its a value from a valueset, get the index of the value gtom the FHIR valuesets
                tensor[0,i] = torch.tensor(get_concept_index_from_codesystem(fhir_value_set, element['binding'].get('valueSet').split('|')[0], value))
        else:
            tensor[0,i] = -1

    return tensor.to(device)


def date_to_one_hot(date):
    # Split the date string into year, month, and day components
    year, month, day = date.split('-')

    # Define the possible values for year, month, and day
    years = [str(i) for i in range(1900, 2101)]  # You can adjust the range of years as needed
    months = [str(i).zfill(2) for i in range(1, 13)]
    days_in_month = [str(i).zfill(2) for i in range(1, 32)]

    # Create the one-hot encoded vectors for year, month, and day
    year_vector = [1 if year == y else 0 for y in years]
    month_vector = [1 if month == m else 0 for m in months]
    day_vector = [1 if day == d else 0 for d in days_in_month]

    # Combine the one-hot encoded vectors into a single vector
    one_hot_vector = year_vector + month_vector + day_vector

    return one_hot_vector

def get_concept_index_from_codesystem(fhir_value_set, fhir_value_set_url, concept_code):
    for entry in fhir_value_set['entry']:
        if entry['resource'].get('valueSet') == fhir_value_set_url:
            concept_index = 0
            for concept in entry['resource'].get('concept'):
                if concept.get('code') == concept_code:
                    return concept_index
                concept_index +=1

def train_gan(generator, discriminator, dataloader, num_epochs, device):
    # Define loss function and optimizers
    criterion = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Training loop
    num_epochs = 200
    for epoch in range(num_epochs):
        for batch_idx, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            
            # Train discriminator with real data
            discriminator.zero_grad()
            real_labels = torch.ones(batch_size, 1, 1).to(device)
            real_output = discriminator(real_data)
            real_loss = criterion(real_output, real_labels)
            real_loss.backward()
            real_cpu = real_data[0].to(device)

            # Train discriminator with generated data
            noise = torch.randn(batch_size, input_dim).to(device)
            fake_data = generator(noise).detach()
            fake_labels = torch.zeros(batch_size, 1).to(device)
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
            # Use inverted labels for generator loss
            real_labels.fill_(1)
            fake_output = discriminator(fake_data)
            generator_loss = criterion(fake_output.squeeze(), real_labels.squeeze())
            generator_loss.backward()
            generator_optimizer.step()

            if batch_idx % 100 == 0: #Only print the stats on the batch
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Batch complete with [{batch_size} passes, "
                    f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                    f"Generator Loss: {generator_loss.item():.4f}")
                # Print the generated text after each epoch
                generated_text = fake_data[0].detach().cpu().numpy()  # Convert tensor to numpy array
                # Convert the sequence of integers to a sequence of characters
                generated_list = generated_text.tolist()
                #generated_text = ''.join(int_to_char[i] for i in generated_list)
                print(f"Generated Text: {generated_text}")

# Entry point of the script
if __name__ == "__main__":
    # Set input dim
    input_dim = 1  # Dimension of the random noise input for the generator
    output_dim = 27  # Dimension of the generated output

    # Set other training parameters
    lr = 0.0002  # Learning rate
    batch_size = 1000  # Batch size for training
    num_epochs = 200

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize generator and discriminator
    generator = Generator(input_dim, output_dim).to(device)
    discriminator = Discriminator(output_dim).to(device)

    # Load the FHIR dataset
    dataset = FHIRDataset('data/Patient.ndjson')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizers
    criterion = nn.BCELoss()
    generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    # Train the GAN
    train_gan(generator, discriminator, dataloader, num_epochs, device)

    # Save trained models
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

