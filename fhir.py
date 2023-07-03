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
    print ("date inside date_to_one_hot", date)
    # Split the date tensor into year, month, and day components
    year, month, day = date[:100].argmax().item(), date[100:112].argmax().item(), date[112:].argmax().item()

    # Define the possible values for year, month, and day
    years = [str(i) for i in range(1900, 2101)]  # You can adjust the range of years as needed
    months = [str(i).zfill(2) for i in range(1, 13)]
    days_in_month = [str(i).zfill(2) for i in range(1, 32)]

    # Create the one-hot encoded vectors for year, month, and day
    year_vector = [1 if year == int(y) else 0 for y in years]
    month_vector = [1 if month == int(m) else 0 for m in months]
    day_vector = [1 if day == int(d) else 0 for d in days_in_month]

    # Combine the one-hot encoded vectors into a single vector
    one_hot_vector = year_vector + month_vector + day_vector

    return one_hot_vector

if __name__ == "__main__":
    # Set the number of patient data samples to generate (use the same batch size as in the GAN model training)
    num_samples_to_generate = 1000

    # Generate patient data using the generator model
    generated_patient_data = generate_patient_data(num_samples_to_generate)

    # Process and print the generated patient data
    for idx, data in enumerate(generated_patient_data):
        print(f"Generated Patient Data {idx + 1}:")
        for i, value in enumerate(data):
            # Handle one-hot encoded date values
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
