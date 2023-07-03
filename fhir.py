import torch
import torch.nn as nn
import json
from gan import Generator, input_dim, output_dim, device, get_concept_index_from_codesystem

# Load the FHIR ValueSets and Profiles-Resources JSON data
with open('fhir/valuesets.json', encoding='utf8', mode='r') as f:
    fhir_value_set = json.load(f)

with open('fhir/profiles-resources.json', encoding='utf8', mode='r') as f:
    fhir_profiles_resources_json = json.load(f)

# Load the pre-trained generator model
generator = Generator(input_dim, output_dim).to(device)
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Function to generate patient data using the generator model
def generate_patient_data(num_samples):
    generated_data = []
    with torch.no_grad():
        for _ in range(num_samples):
            noise = torch.randn(1, input_dim).to(device)
            fake_data = generator(noise)
            generated_data.append(fake_data.cpu().numpy()[0])

    return generated_data

def date_to_one_hot(date):
    # Check if the date tensor has only one element
    if date.numel() == 1:
        date = date.item()  # Convert the tensor to a scalar

    # Split the date into year, month, and day components
    year, month, day = int(date * 100), int((date * 100) % 100), int((date * 10000) % 100)

    # Create the date string in the format "YYYY-MM-DD"
    date_str = f"{year:04d}-{month:02d}-{day:02d}"

    return date_str

if __name__ == "__main__":
    # Set the number of patient data samples to generate (use the same batch size as in the GAN model training)
    num_samples_to_generate = 1000

    # Generate patient data using the generator model
    generated_patient_data = generate_patient_data(num_samples_to_generate)
    print (generated_patient_data)
    # Process and print the generated patient data
    for idx, data in enumerate(generated_patient_data):
        print(f"Generated Patient Data {idx + 1}:")
        for i, value in enumerate(data):
            # Handle the date value
            if i == 0:
                date = torch.tensor([value], device=device)  # Create tensor with unsqueezed value
                date_str = date_to_one_hot(date)
                print(f"Date: {date_str}")
            else:
                # Fetch value from the FHIR ValueSet for categorical values
                concept_index = int(value)
                concept_code = fhir_value_set['entry'][concept_index]['resource']['concept'][0]['code']
                print(f"Concept Index {i}: {concept_index}, Concept Code: {concept_code}")
        print("-----")
