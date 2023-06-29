import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from fhir_types import FHIR_Patient

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
        return fhir_patient_to_tensor(self.data[index], fhir_structure_definition_json, fhir_value_set)  # Assuming each line is a tensor

def fhir_patient_to_tensor(fhir_patient_json, fhir_structure_definition_json, fhir_value_set):
    # Parse the FHIR Patient resource
    fhir_patient = FHIR_Patient (fhir_patient_json)

    # Load the FHIR StructureDefinition
    fhir_structure_definition = json.loads(fhir_structure_definition_json, strict=False)

    # Get the list of elements from the StructureDefinition
    elements = fhir_structure_definition['differential']['element'][1:]
        # Create an empty tensor with the shape of the elements
    tensor_shape = (1001, len(elements)-1)
    tensor = torch.empty(tensor_shape)      
    # Iterate through the elements and populate the tensor
    for i, element in enumerate(elements):
        if (i>0 and i<26):
            value = fhir_patient.get(element['id'].split('.')[1],None)
            if value is not None:
                if isinstance(value, list):
                    tensor[0,i] = torch.tensor(len(value))
                elif element.get('type')[0].get('code') == 'date':
                    tensor[0,i] = torch.tensor(len(date_to_one_hot(value)))
                elif element.get('type')[0].get('code') == 'CodeableConcept':
                    tensor[0,i] = torch.tensor(len(value))
                else:
                    tensor[0,i] = torch.tensor(get_concept_index_from_codesystem(fhir_value_set, element['binding'].get('valueSet').split('|')[0], value))
            else:
                tensor[0,i] = -1

    return tensor
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



fhir_structure_definition_json = r"""
    {
        "abstract": false,
        "baseDefinition": "http://hl7.org/fhir/StructureDefinition/DomainResource",
        "contact": [
            {
                "telecom": [
                    {
                        "system": "url",
                        "value": "http://hl7.org/fhir"
                    }
                ]
            },
            {
                "telecom": [
                    {
                        "system": "url",
                        "value": "http://www.hl7.org/Special/committees/pafm/index.cfm"
                    }
                ]
            }
        ],
        "date": "2023-03-26T15:21:02+11:00",
        "derivation": "specialization",
        "description": "Demographics and other administrative information about an individual or animal receiving care or other health-related services.",
        "differential": {
            "element": [
                {
                    "alias": [
                        "SubjectOfCare Client Resident"
                    ],
                    "definition": "Demographics and other administrative information about an individual or animal receiving care or other health-related services.",
                    "id": "Patient",
                    "isModifier": false,
                    "mapping": [
                        {
                            "identity": "w5",
                            "map": "administrative.individual"
                        },
                        {
                            "identity": "rim",
                            "map": "Patient[classCode=PAT]"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving"
                        },
                        {
                            "identity": "cda",
                            "map": "ClinicalDocument.recordTarget.patientRole"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient",
                    "short": "Information about an individual or animal receiving health care services"
                },
                {
                    "definition": "An identifier for this patient.",
                    "id": "Patient.identifier",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "w5",
                            "map": "FiveWs.identifier"
                        },
                        {
                            "identity": "v2",
                            "map": "PID-3"
                        },
                        {
                            "identity": "rim",
                            "map": "id"
                        },
                        {
                            "identity": "interface",
                            "map": "Participant.identifier"
                        },
                        {
                            "identity": "cda",
                            "map": ".id"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.identifier",
                    "requirements": "Patients are almost always assigned specific numerical identifiers.",
                    "short": "An identifier for this patient",
                    "type": [
                        {
                            "code": "Identifier"
                        }
                    ]
                },
                {
                    "comment": "If a record is inactive, and linked to an active record, then future patient/record updates should occur on the other patient.",
                    "definition": "Whether this patient record is in active use. \nMany systems use this property to mark as non-current patients, such as those that have not been seen for a period of time based on an organization's business rules.\n\nIt is often used to filter patient lists to exclude inactive patients\n\nDeceased patients may also be marked as inactive for the same reasons, but may be active for some time after death.",
                    "id": "Patient.active",
                    "isModifier": true,
                    "isModifierReason": "This element is labelled as a modifier because it is a status element that can indicate that a record should not be treated as valid",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "w5",
                            "map": "FiveWs.status"
                        },
                        {
                            "identity": "rim",
                            "map": "statusCode"
                        },
                        {
                            "identity": "interface",
                            "map": "Participant.active"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "meaningWhenMissing": "This resource is generally assumed to be active if no value is provided for the active element",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.active",
                    "requirements": "Need to be able to mark a patient record as not to be used because it was created in error.",
                    "short": "Whether this patient's record is in active use",
                    "type": [
                        {
                            "code": "boolean"
                        }
                    ]
                },
                {
                    "comment": "A patient may have multiple names with different uses or applicable periods. For animals, the name is a \"HumanName\" in the sense that is assigned and used by humans and has the same patterns. Animal names may be communicated as given names, and optionally may include a family name.",
                    "definition": "A name associated with the individual.",
                    "id": "Patient.name",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-5, PID-9"
                        },
                        {
                            "identity": "rim",
                            "map": "name"
                        },
                        {
                            "identity": "interface",
                            "map": "Participant.name"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.name"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.name",
                    "requirements": "Need to be able to track the patient by multiple names. Examples are your official name and a partner name.",
                    "short": "A name associated with the patient",
                    "type": [
                        {
                            "code": "HumanName"
                        }
                    ]
                },
                {
                    "comment": "A Patient may have multiple ways to be contacted with different uses or applicable periods.  May need to have options for contacting the person urgently and also to help with identification. The address might not go directly to the individual, but may reach another party that is able to proxy for the patient (i.e. home phone, or pet owner's phone).",
                    "definition": "A contact detail (e.g. a telephone number or an email address) by which the individual may be contacted.",
                    "id": "Patient.telecom",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-13, PID-14, PID-40"
                        },
                        {
                            "identity": "rim",
                            "map": "telecom"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantContactable.telecom"
                        },
                        {
                            "identity": "cda",
                            "map": ".telecom"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.telecom",
                    "requirements": "People have (primary) ways to contact them in some way such as phone, email.",
                    "short": "A contact detail for the individual",
                    "type": [
                        {
                            "code": "ContactPoint"
                        }
                    ]
                },
                {
                    "binding": {
                        "description": "The gender of a person used for administrative purposes.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "AdministrativeGender"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/administrative-gender|5.0.0"
                    },
                    "comment": "The gender might not match the biological sex as determined by genetics or the individual's preferred identification. Note that for both humans and particularly animals, there are other legitimate possibilities than male and female, though the vast majority of systems and contexts only support male and female.  Systems providing decision support or enforcing business rules should ideally do this on the basis of Observations dealing with the specific sex or gender aspect of interest (anatomical, chromosomal, social, etc.)  However, because these observations are infrequently recorded, defaulting to the administrative gender is common practice.  Where such defaulting occurs, rule enforcement should allow for the variation between administrative and biological, chromosomal and other gender aspects.  For example, an alert about a hysterectomy on a male should be handled as a warning or overridable error, not a \"hard\" error.  See the Patient Gender and Sex section for additional information about communicating patient gender and sex.",
                    "definition": "Administrative Gender - the gender that the patient is considered to have for administration and record keeping purposes.",
                    "id": "Patient.gender",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-8"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/administrativeGender"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.gender"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.administrativeGenderCode"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.gender",
                    "requirements": "Needed for identification of the individual, in combination with (at least) name and birth date.",
                    "short": "male | female | other | unknown",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                },
                {
                    "comment": "Partial dates are allowed if the specific date of birth is unknown. There is a standard extension \"patient-birthTime\" available that should be used where Time is required (such as in maternity/infant care systems).",
                    "definition": "The date of birth for the individual.",
                    "id": "Patient.birthDate",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-7"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/birthTime"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.birthDate"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.birthTime"
                        },
                        {
                            "identity": "loinc",
                            "map": "21112-8"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.birthDate",
                    "requirements": "Age of the individual drives many clinical processes.",
                    "short": "The date of birth for the individual",
                    "type": [
                        {
                            "code": "date"
                        }
                    ]
                },
                {
                    "comment": "If there's no value in the instance, it means there is no statement on whether or not the individual is deceased. Most systems will interpret the absence of a value as a sign of the person being alive.",
                    "definition": "Indicates if the individual is deceased or not.",
                    "id": "Patient.deceased[x]",
                    "isModifier": true,
                    "isModifierReason": "This element is labeled as a modifier because once a patient is marked as deceased, the actions that are appropriate to perform on the patient may be significantly different.",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-30  (bool) and PID-29 (datetime)"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/deceasedInd, player[classCode=PSN|ANM and determinerCode=INSTANCE]/deceasedTime"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.deceased[x]",
                    "requirements": "The fact that a patient is deceased influences the clinical process. Also, in human communication and relation management it is necessary to know whether the person is alive.",
                    "short": "Indicates if the individual is deceased or not",
                    "type": [
                        {
                            "code": "boolean"
                        },
                        {
                            "code": "dateTime"
                        }
                    ]
                },
                {
                    "comment": "Patient may have multiple addresses with different uses or applicable periods.",
                    "definition": "An address for the individual.",
                    "id": "Patient.address",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-11"
                        },
                        {
                            "identity": "rim",
                            "map": "addr"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantContactable.address"
                        },
                        {
                            "identity": "cda",
                            "map": ".addr"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.address",
                    "requirements": "May need to keep track of patient addresses for contacting, billing or reporting requirements and also to help with identification.",
                    "short": "An address for the individual",
                    "type": [
                        {
                            "code": "Address"
                        }
                    ]
                },
                {
                    "binding": {
                        "description": "The domestic partnership status of a person.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "MaritalStatus"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "extensible",
                        "valueSet": "http://hl7.org/fhir/ValueSet/marital-status"
                    },
                    "definition": "This field contains a patient's most recent marital (civil) status.",
                    "id": "Patient.maritalStatus",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-16"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN]/maritalStatusCode"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.maritalStatusCode"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.maritalStatus",
                    "requirements": "Most, if not all systems capture it.",
                    "short": "Marital (civil) status of a patient",
                    "type": [
                        {
                            "code": "CodeableConcept"
                        }
                    ]
                },
                {
                    "comment": "Where the valueInteger is provided, the number is the birth number in the sequence. E.g. The middle birth in triplets would be valueInteger=2 and the third born would have valueInteger=3 If a boolean value was provided for this triplets example, then all 3 patient records would have valueBoolean=true (the ordering is not indicated).",
                    "definition": "Indicates whether the patient is part of a multiple (boolean) or indicates the actual birth order (integer).",
                    "id": "Patient.multipleBirth[x]",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-24 (bool), PID-25 (integer)"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/multipleBirthInd,  player[classCode=PSN|ANM and determinerCode=INSTANCE]/multipleBirthOrderNumber"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.multipleBirth[x]",
                    "requirements": "For disambiguation of multiple-birth children, especially relevant where the care provider doesn't meet the patient, such as labs.",
                    "short": "Whether patient is part of a multiple birth",
                    "type": [
                        {
                            "code": "boolean"
                        },
                        {
                            "code": "integer"
                        }
                    ]
                },
                {
                    "comment": "Guidelines:\n* Use id photos, not clinical photos.\n* Limit dimensions to thumbnail.\n* Keep byte count low to ease resource updates.",
                    "definition": "Image of the patient.",
                    "id": "Patient.photo",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "OBX-5 - needs a profile"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/desc"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.photo"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.photo",
                    "requirements": "Many EHR systems have the capability to capture an image of the patient. Fits with newer social media usage too.",
                    "short": "Image of the patient",
                    "type": [
                        {
                            "code": "Attachment"
                        }
                    ]
                },
                {
                    "comment": "Contact covers all kinds of contact parties: family members, business contacts, guardians, caregivers. Not applicable to register pedigree and family ties beyond use of having contact.",
                    "constraint": [
                        {
                            "expression": "name.exists() or telecom.exists() or address.exists() or organization.exists()",
                            "human": "SHALL at least contain a contact's details or a reference to an organization",
                            "key": "pat-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Patient"
                        }
                    ],
                    "definition": "A contact party (e.g. guardian, partner, friend) for the patient.",
                    "extension": [
                        {
                            "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-explicit-type-name",
                            "valueString": "Contact"
                        }
                    ],
                    "id": "Patient.contact",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/scopedRole[classCode=CON]"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact",
                    "requirements": "Need to track people you can contact about the patient.",
                    "short": "A contact party (e.g. guardian, partner, friend) for the patient",
                    "type": [
                        {
                            "code": "BackboneElement"
                        }
                    ]
                },
                {
                    "binding": {
                        "description": "The nature of the relationship between a patient and a contact person for that patient.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "ContactRelationship"
                            }
                        ],
                        "strength": "extensible",
                        "valueSet": "http://hl7.org/fhir/ValueSet/patient-contactrelationship"
                    },
                    "definition": "The nature of the relationship between the patient and the contact person.",
                    "id": "Patient.contact.relationship",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-7, NK1-3"
                        },
                        {
                            "identity": "rim",
                            "map": "code"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.relationship",
                    "requirements": "Used to determine which contact person is the most relevant to approach, depending on circumstances.",
                    "short": "The kind of relationship",
                    "type": [
                        {
                            "code": "CodeableConcept"
                        }
                    ]
                },
                {
                    "condition": [
                        "pat-1"
                    ],
                    "definition": "A name associated with the contact person.",
                    "id": "Patient.contact.name",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-2"
                        },
                        {
                            "identity": "rim",
                            "map": "name"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.name",
                    "requirements": "Contact persons need to be identified by name, but it is uncommon to need details about multiple other names for that contact person.",
                    "short": "A name associated with the contact person",
                    "type": [
                        {
                            "code": "HumanName"
                        }
                    ]
                },
                {
                    "comment": "Contact may have multiple ways to be contacted with different uses or applicable periods.  May need to have options for contacting the person urgently, and also to help with identification.",
                    "condition": [
                        "pat-1"
                    ],
                    "definition": "A contact detail for the person, e.g. a telephone number or an email address.",
                    "id": "Patient.contact.telecom",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-5, NK1-6, NK1-40"
                        },
                        {
                            "identity": "rim",
                            "map": "telecom"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.telecom",
                    "requirements": "People have (primary) ways to contact them in some way such as phone, email.",
                    "short": "A contact detail for the person",
                    "type": [
                        {
                            "code": "ContactPoint"
                        }
                    ]
                },
                {
                    "condition": [
                        "pat-1"
                    ],
                    "definition": "Address for the contact person.",
                    "id": "Patient.contact.address",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-4"
                        },
                        {
                            "identity": "rim",
                            "map": "addr"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.address",
                    "requirements": "Need to keep track where the contact person can be contacted per postal mail or visited.",
                    "short": "Address for the contact person",
                    "type": [
                        {
                            "code": "Address"
                        }
                    ]
                },
                {
                    "binding": {
                        "description": "The gender of a person used for administrative purposes.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "AdministrativeGender"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/administrative-gender|5.0.0"
                    },
                    "definition": "Administrative Gender - the gender that the contact person is considered to have for administration and record keeping purposes.",
                    "id": "Patient.contact.gender",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-15"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/administrativeGender"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.gender",
                    "requirements": "Needed to address the person correctly.",
                    "short": "male | female | other | unknown",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                },
                {
                    "condition": [
                        "pat-1"
                    ],
                    "definition": "Organization on behalf of which the contact is acting or for which the contact is working.",
                    "id": "Patient.contact.organization",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-13, NK1-30, NK1-31, NK1-32, NK1-41"
                        },
                        {
                            "identity": "rim",
                            "map": "scoper"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.organization",
                    "requirements": "For guardians or business related contacts, the organization is relevant.",
                    "short": "Organization that is associated with the contact",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Organization"
                            ]
                        }
                    ]
                },
                {
                    "definition": "The period during which this contact person or organization is valid to be contacted relating to this patient.",
                    "id": "Patient.contact.period",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "effectiveTime"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.period",
                    "short": "The period during which this contact person or organization is valid to be contacted relating to this patient",
                    "type": [
                        {
                            "code": "Period"
                        }
                    ]
                },
                {
                    "comment": "If no language is specified, this *implies* that the default local language is spoken.  If you need to convey proficiency for multiple modes, then you need multiple Patient.Communication associations.   For animals, language is not a relevant field, and should be absent from the instance. If the Patient does not speak the default local language, then the Interpreter Required Standard can be used to explicitly declare that an interpreter is required.",
                    "definition": "A language which may be used to communicate with the patient about his or her health.",
                    "id": "Patient.communication",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "LanguageCommunication"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.communication"
                        },
                        {
                            "identity": "cda",
                            "map": "patient.languageCommunication"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.communication",
                    "requirements": "If a patient does not speak the local language, interpreters may be required, so languages spoken and proficiency are important things to keep track of both for patient and other persons of interest.",
                    "short": "A language which may be used to communicate with the patient about his or her health",
                    "type": [
                        {
                            "code": "BackboneElement"
                        }
                    ]
                },
                {
                    "binding": {
                        "additional": [
                            {
                                "purpose": "starter",
                                "valueSet": "http://hl7.org/fhir/ValueSet/languages"
                            }
                        ],
                        "description": "IETF language tag for a human language",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "Language"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/all-languages|5.0.0"
                    },
                    "comment": "The structure aa-BB with this exact casing is one the most widely used notations for locale. However not all systems actually code this but instead have it as free text. Hence CodeableConcept instead of code as the data type.",
                    "definition": "The ISO-639-1 alpha 2 code in lower case for the language, optionally followed by a hyphen and the ISO-3166-1 alpha 2 code for the region in upper case; e.g. \"en\" for English, or \"en-US\" for American English versus \"en-AU\" for Australian English.",
                    "id": "Patient.communication.language",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-15, LAN-2"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/languageCommunication/code"
                        },
                        {
                            "identity": "cda",
                            "map": ".languageCode"
                        }
                    ],
                    "max": "1",
                    "min": 1,
                    "mustSupport": false,
                    "path": "Patient.communication.language",
                    "requirements": "Most systems in multilingual countries will want to convey language. Not all systems actually need the regional dialect.",
                    "short": "The language which can be used to communicate with the patient about his or her health",
                    "type": [
                        {
                            "code": "CodeableConcept"
                        }
                    ]
                },
                {
                    "comment": "This language is specifically identified for communicating healthcare information.",
                    "definition": "Indicates whether or not the patient prefers this language (over other languages he masters up a certain level).",
                    "id": "Patient.communication.preferred",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-15"
                        },
                        {
                            "identity": "rim",
                            "map": "preferenceInd"
                        },
                        {
                            "identity": "cda",
                            "map": ".preferenceInd"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.communication.preferred",
                    "requirements": "People that master multiple languages up to certain level may prefer one or more, i.e. feel more confident in communicating in a particular language making other languages sort of a fall back method.",
                    "short": "Language preference indicator",
                    "type": [
                        {
                            "code": "boolean"
                        }
                    ]
                },
                {
                    "alias": [
                        "careProvider"
                    ],
                    "comment": "This may be the primary care provider (in a GP context), or it may be a patient nominated care manager in a community/disability setting, or even organization that will provide people to perform the care provider roles.  It is not to be used to record Care Teams, these should be in a CareTeam resource that may be linked to the CarePlan or EpisodeOfCare resources.\nMultiple GPs may be recorded against the patient for various reasons, such as a student that has his home GP listed along with the GP at university during the school semesters, or a \"fly-in/fly-out\" worker that has the onsite GP also included with his home GP to remain aware of medical issues.\n\nJurisdictions may decide that they can profile this down to 1 if desired, or 1 per type.",
                    "definition": "Patient's nominated care provider.",
                    "id": "Patient.generalPractitioner",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PD1-4"
                        },
                        {
                            "identity": "rim",
                            "map": "subjectOf.CareEvent.performer.AssignedEntity"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.generalPractitioner",
                    "short": "Patient's nominated primary care provider",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Organization",
                                "http://hl7.org/fhir/StructureDefinition/Practitioner",
                                "http://hl7.org/fhir/StructureDefinition/PractitionerRole"
                            ]
                        }
                    ]
                },
                {
                    "comment": "There is only one managing organization for a specific patient record. Other organizations will have their own Patient record, and may use the Link property to join the records together (or a Person resource which can include confidence ratings for the association).",
                    "definition": "Organization that is the custodian of the patient record.",
                    "id": "Patient.managingOrganization",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "scoper"
                        },
                        {
                            "identity": "cda",
                            "map": ".providerOrganization"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.managingOrganization",
                    "requirements": "Need to know who recognizes this patient record, manages and updates it.",
                    "short": "Organization that is the custodian of the patient record",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Organization"
                            ]
                        }
                    ]
                },
                {
                    "comment": "There is no assumption that linked patient records have mutual links.",
                    "definition": "Link to a Patient or RelatedPerson resource that concerns the same actual individual.",
                    "id": "Patient.link",
                    "isModifier": true,
                    "isModifierReason": "This element is labeled as a modifier because it might not be the main Patient resource, and the referenced patient should be used instead of this Patient record. This is when the link.type value is 'replaced-by'",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "outboundLink"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.link",
                    "requirements": "There are multiple use cases:   \n\n* Duplicate patient records due to the clerical errors associated with the difficulties of identifying humans consistently, and \n* Distribution of patient information across multiple servers.",
                    "short": "Link to a Patient or RelatedPerson resource that concerns the same actual individual",
                    "type": [
                        {
                            "code": "BackboneElement"
                        }
                    ]
                },
                {
                    "comment": "Referencing a RelatedPerson here removes the need to use a Person record to associate a Patient and RelatedPerson as the same individual.",
                    "definition": "Link to a Patient or RelatedPerson resource that concerns the same actual individual.",
                    "id": "Patient.link.other",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-3, MRG-1"
                        },
                        {
                            "identity": "rim",
                            "map": "id"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 1,
                    "mustSupport": false,
                    "path": "Patient.link.other",
                    "short": "The other patient or related person resource that the link refers to",
                    "type": [
                        {
                            "code": "Reference",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-hierarchy",
                                    "valueBoolean": false
                                }
                            ],
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Patient",
                                "http://hl7.org/fhir/StructureDefinition/RelatedPerson"
                            ]
                        }
                    ]
                },
                {
                    "binding": {
                        "description": "The type of link between this patient resource and another Patient resource, or Patient/RelatedPerson when using the `seealso` code",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "LinkType"
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/link-type|5.0.0"
                    },
                    "definition": "The type of link between this patient resource and another patient resource.",
                    "id": "Patient.link.type",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "typeCode"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 1,
                    "mustSupport": false,
                    "path": "Patient.link.type",
                    "short": "replaced-by | replaces | refer | seealso",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                }
            ]
        },
        "experimental": false,
        "extension": [
            {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-category",
                "valueString": "Base.Individuals"
            },
            {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-standards-status",
                "valueCode": "normative"
            },
            {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-normative-version",
                "valueCode": "4.0.0"
            },
            {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-fmm",
                "valueInteger": 5
            },
            {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-security-category",
                "valueCode": "patient"
            },
            {
                "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-wg",
                "valueCode": "pa"
            }
        ],
        "fhirVersion": "5.0.0",
        "id": "Patient",
        "jurisdiction": [
            {
                "coding": [
                    {
                        "code": "001",
                        "display": "World",
                        "system": "http://unstats.un.org/unsd/methods/m49/m49.htm"
                    }
                ]
            }
        ],
        "kind": "resource",
        "mapping": [
            {
                "identity": "w5",
                "name": "FiveWs Pattern Mapping",
                "uri": "http://hl7.org/fhir/fivews"
            },
            {
                "identity": "rim",
                "name": "RIM Mapping",
                "uri": "http://hl7.org/v3"
            },
            {
                "identity": "interface",
                "name": "Interface Pattern",
                "uri": "http://hl7.org/fhir/interface"
            },
            {
                "identity": "cda",
                "name": "CDA (R2)",
                "uri": "http://hl7.org/v3/cda"
            },
            {
                "identity": "v2",
                "name": "HL7 V2 Mapping",
                "uri": "http://hl7.org/v2"
            },
            {
                "identity": "loinc",
                "name": "LOINC code for the element",
                "uri": "http://loinc.org"
            }
        ],
        "meta": {
            "lastUpdated": "2023-03-26T15:21:02.749+11:00"
        },
        "name": "Patient",
        "publisher": "Health Level Seven International (Patient Administration)",
        "purpose": "Tracking patient is the center of the healthcare process.",
        "resourceType": "StructureDefinition",
        "snapshot": {
            "element": [
                {
                    "alias": [
                        "SubjectOfCare Client Resident"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient"
                    },
                    "constraint": [
                        {
                            "expression": "contained.contained.empty()",
                            "human": "If the resource is contained in another resource, it SHALL NOT contain nested Resources",
                            "key": "dom-2",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/DomainResource"
                        },
                        {
                            "expression": "contained.where((('#'+id in (%resource.descendants().reference | %resource.descendants().ofType(canonical) | %resource.descendants().ofType(uri) | %resource.descendants().ofType(url))) or descendants().where(reference = '#').exists() or descendants().where(ofType(canonical) = '#').exists() or descendants().where(ofType(canonical) = '#').exists()).not()).trace('unmatched', id).empty()",
                            "human": "If the resource is contained in another resource, it SHALL be referred to from elsewhere in the resource or SHALL refer to the containing resource",
                            "key": "dom-3",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/DomainResource"
                        },
                        {
                            "expression": "contained.meta.versionId.empty() and contained.meta.lastUpdated.empty()",
                            "human": "If a resource is contained in another resource, it SHALL NOT have a meta.versionId or a meta.lastUpdated",
                            "key": "dom-4",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/DomainResource"
                        },
                        {
                            "expression": "contained.meta.security.empty()",
                            "human": "If a resource is contained in another resource, it SHALL NOT have a security label",
                            "key": "dom-5",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/DomainResource"
                        },
                        {
                            "expression": "text.`div`.exists()",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bestpractice",
                                    "valueBoolean": true
                                },
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bestpractice-explanation",
                                    "valueMarkdown": "When a resource has no narrative, only systems that fully understand the data can display the resource to a human safely. Including a human readable representation in the resource makes for a much more robust eco-system and cheaper handling of resources by intermediary systems. Some ecosystems restrict distribution of resources to only those systems that do fully understand the resources, and as a consequence implementers may believe that the narrative is superfluous. However experience shows that such eco-systems often open up to new participants over time."
                                }
                            ],
                            "human": "A resource should have narrative for robust management",
                            "key": "dom-6",
                            "severity": "warning",
                            "source": "http://hl7.org/fhir/StructureDefinition/DomainResource"
                        }
                    ],
                    "definition": "Demographics and other administrative information about an individual or animal receiving care or other health-related services.",
                    "id": "Patient",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "Entity, Role, or Act,Patient[classCode=PAT]"
                        },
                        {
                            "identity": "w5",
                            "map": "administrative.individual"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving"
                        },
                        {
                            "identity": "cda",
                            "map": "ClinicalDocument.recordTarget.patientRole"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient",
                    "short": "Information about an individual or animal receiving health care services"
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Resource.id"
                    },
                    "comment": "Within the context of the FHIR RESTful interactions, the resource has an id except for cases like the create and conditional update. Otherwise, the use of the resouce id depends on the given use case.",
                    "definition": "The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.",
                    "id": "Patient.id",
                    "isModifier": false,
                    "isSummary": true,
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.id",
                    "short": "Logical id of this artifact",
                    "type": [
                        {
                            "code": "http://hl7.org/fhirpath/System.String",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-fhir-type",
                                    "valueUrl": "id"
                                }
                            ]
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Resource.meta"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The metadata about the resource. This is content that is maintained by the infrastructure. Changes to the content might not always be associated with version changes to the resource.",
                    "id": "Patient.meta",
                    "isModifier": false,
                    "isSummary": true,
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.meta",
                    "short": "Metadata about the resource",
                    "type": [
                        {
                            "code": "Meta"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Resource.implicitRules"
                    },
                    "comment": "Asserting this rule set restricts the content to be only understood by a limited set of trading partners. This inherently limits the usefulness of the data in the long term. However, the existing health eco-system is highly fractured, and not yet ready to define, collect, and exchange data in a generally computable sense. Wherever possible, implementers and/or specification writers should avoid using this element. Often, when used, the URL is a reference to an implementation guide that defines these special rules as part of its narrative along with other profiles, value sets, etc.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.",
                    "id": "Patient.implicitRules",
                    "isModifier": true,
                    "isModifierReason": "This element is labeled as a modifier because the implicit rules may provide additional knowledge about the resource that modifies its meaning or interpretation",
                    "isSummary": true,
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.implicitRules",
                    "short": "A set of rules under which this content was created",
                    "type": [
                        {
                            "code": "uri"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Resource.language"
                    },
                    "binding": {
                        "additional": [
                            {
                                "purpose": "starter",
                                "valueSet": "http://hl7.org/fhir/ValueSet/languages"
                            }
                        ],
                        "description": "IETF language tag for a human language",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "Language"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/all-languages|5.0.0"
                    },
                    "comment": "Language is provided to support indexing and accessibility (typically, services such as text to speech use the language tag). The html language tag in the narrative applies  to the narrative. The language tag on the resource may be used to specify the language of other presentations generated from the data in the resource. Not all the content has to be in the base language. The Resource.language should not be assumed to apply to the narrative automatically. If a language is specified, it should it also be specified on the div element in the html (see rules in HTML5 for information about the relationship between xml:lang and the html lang attribute).",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The base language in which the resource is written.",
                    "id": "Patient.language",
                    "isModifier": false,
                    "isSummary": false,
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.language",
                    "short": "Language of the resource content",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                },
                {
                    "alias": [
                        "narrative",
                        "html",
                        "xhtml",
                        "display"
                    ],
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "DomainResource.text"
                    },
                    "comment": "Contained resources do not have a narrative. Resources that are not contained SHOULD have a narrative. In some cases, a resource may only have text with little or no additional discrete data (as long as all minOccurs=1 elements are satisfied).  This may be necessary for data from legacy systems where information is captured as a \"text blob\" or where text is additionally entered raw or narrated and encoded information is added later.",
                    "condition": [
                        "dom-6"
                    ],
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human. The narrative need not encode all the structured data, but is required to contain sufficient detail to make it \"clinically safe\" for a human to just read the narrative. Resource definitions may define what content should be represented in the narrative to ensure clinical safety.",
                    "id": "Patient.text",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "Act.text?"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.text",
                    "short": "Text summary of the resource, for human interpretation",
                    "type": [
                        {
                            "code": "Narrative"
                        }
                    ]
                },
                {
                    "alias": [
                        "inline resources",
                        "anonymous resources",
                        "contained resources"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "DomainResource.contained"
                    },
                    "comment": "This should never be done when the content can be identified properly, as once identification is lost, it is extremely difficult (and context dependent) to restore it again. Contained resources may have profiles and tags in their meta elements, but SHALL NOT have security labels.",
                    "condition": [
                        "dom-2",
                        "dom-4",
                        "dom-3",
                        "dom-5"
                    ],
                    "definition": "These resources do not have an independent existence apart from the resource that contains them - they cannot be identified independently, nor can they have their own independent transaction scope. This is allowed to be a Parameters resource if and only if it is referenced by a resource that provides context/meaning.",
                    "id": "Patient.contained",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "N/A"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contained",
                    "short": "Contained, inline Resources",
                    "type": [
                        {
                            "code": "Resource"
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "DomainResource.extension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the resource. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.",
                    "id": "Patient.extension",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "N/A"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.extension",
                    "short": "Additional content defined by implementations",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "DomainResource.modifierExtension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the resource and that modifies the understanding of the element that contains it and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer is allowed to define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.\n\nModifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
                    "id": "Patient.modifierExtension",
                    "isModifier": true,
                    "isModifierReason": "Modifier extensions are expected to modify the meaning or interpretation of the resource that contains them",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "N/A"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.modifierExtension",
                    "requirements": "Modifier extensions allow for extensions that *cannot* be safely ignored to be clearly distinguished from the vast majority of extensions which can be safely ignored.  This promotes interoperability by eliminating the need for implementers to prohibit the presence of extensions. For further information, see the [definition of modifier extensions](extensibility.html#modifierExtension).",
                    "short": "Extensions that cannot be ignored",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.identifier"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "An identifier for this patient.",
                    "id": "Patient.identifier",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "w5",
                            "map": "FiveWs.identifier"
                        },
                        {
                            "identity": "v2",
                            "map": "PID-3"
                        },
                        {
                            "identity": "rim",
                            "map": "id"
                        },
                        {
                            "identity": "interface",
                            "map": "Participant.identifier"
                        },
                        {
                            "identity": "cda",
                            "map": ".id"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.identifier",
                    "requirements": "Patients are almost always assigned specific numerical identifiers.",
                    "short": "An identifier for this patient",
                    "type": [
                        {
                            "code": "Identifier"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.active"
                    },
                    "comment": "If a record is inactive, and linked to an active record, then future patient/record updates should occur on the other patient.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Whether this patient record is in active use. \nMany systems use this property to mark as non-current patients, such as those that have not been seen for a period of time based on an organization's business rules.\n\nIt is often used to filter patient lists to exclude inactive patients\n\nDeceased patients may also be marked as inactive for the same reasons, but may be active for some time after death.",
                    "id": "Patient.active",
                    "isModifier": true,
                    "isModifierReason": "This element is labelled as a modifier because it is a status element that can indicate that a record should not be treated as valid",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "w5",
                            "map": "FiveWs.status"
                        },
                        {
                            "identity": "rim",
                            "map": "statusCode"
                        },
                        {
                            "identity": "interface",
                            "map": "Participant.active"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "meaningWhenMissing": "This resource is generally assumed to be active if no value is provided for the active element",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.active",
                    "requirements": "Need to be able to mark a patient record as not to be used because it was created in error.",
                    "short": "Whether this patient's record is in active use",
                    "type": [
                        {
                            "code": "boolean"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.name"
                    },
                    "comment": "A patient may have multiple names with different uses or applicable periods. For animals, the name is a \"HumanName\" in the sense that is assigned and used by humans and has the same patterns. Animal names may be communicated as given names, and optionally may include a family name.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A name associated with the individual.",
                    "id": "Patient.name",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-5, PID-9"
                        },
                        {
                            "identity": "rim",
                            "map": "name"
                        },
                        {
                            "identity": "interface",
                            "map": "Participant.name"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.name"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.name",
                    "requirements": "Need to be able to track the patient by multiple names. Examples are your official name and a partner name.",
                    "short": "A name associated with the patient",
                    "type": [
                        {
                            "code": "HumanName"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.telecom"
                    },
                    "comment": "A Patient may have multiple ways to be contacted with different uses or applicable periods.  May need to have options for contacting the person urgently and also to help with identification. The address might not go directly to the individual, but may reach another party that is able to proxy for the patient (i.e. home phone, or pet owner's phone).",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A contact detail (e.g. a telephone number or an email address) by which the individual may be contacted.",
                    "id": "Patient.telecom",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-13, PID-14, PID-40"
                        },
                        {
                            "identity": "rim",
                            "map": "telecom"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantContactable.telecom"
                        },
                        {
                            "identity": "cda",
                            "map": ".telecom"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.telecom",
                    "requirements": "People have (primary) ways to contact them in some way such as phone, email.",
                    "short": "A contact detail for the individual",
                    "type": [
                        {
                            "code": "ContactPoint"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.gender"
                    },
                    "binding": {
                        "description": "The gender of a person used for administrative purposes.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "AdministrativeGender"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/administrative-gender|5.0.0"
                    },
                    "comment": "The gender might not match the biological sex as determined by genetics or the individual's preferred identification. Note that for both humans and particularly animals, there are other legitimate possibilities than male and female, though the vast majority of systems and contexts only support male and female.  Systems providing decision support or enforcing business rules should ideally do this on the basis of Observations dealing with the specific sex or gender aspect of interest (anatomical, chromosomal, social, etc.)  However, because these observations are infrequently recorded, defaulting to the administrative gender is common practice.  Where such defaulting occurs, rule enforcement should allow for the variation between administrative and biological, chromosomal and other gender aspects.  For example, an alert about a hysterectomy on a male should be handled as a warning or overridable error, not a \"hard\" error.  See the Patient Gender and Sex section for additional information about communicating patient gender and sex.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Administrative Gender - the gender that the patient is considered to have for administration and record keeping purposes.",
                    "id": "Patient.gender",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-8"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/administrativeGender"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.gender"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.administrativeGenderCode"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.gender",
                    "requirements": "Needed for identification of the individual, in combination with (at least) name and birth date.",
                    "short": "male | female | other | unknown",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.birthDate"
                    },
                    "comment": "Partial dates are allowed if the specific date of birth is unknown. There is a standard extension \"patient-birthTime\" available that should be used where Time is required (such as in maternity/infant care systems).",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The date of birth for the individual.",
                    "id": "Patient.birthDate",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-7"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/birthTime"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.birthDate"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.birthTime"
                        },
                        {
                            "identity": "loinc",
                            "map": "21112-8"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.birthDate",
                    "requirements": "Age of the individual drives many clinical processes.",
                    "short": "The date of birth for the individual",
                    "type": [
                        {
                            "code": "date"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.deceased[x]"
                    },
                    "comment": "If there's no value in the instance, it means there is no statement on whether or not the individual is deceased. Most systems will interpret the absence of a value as a sign of the person being alive.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Indicates if the individual is deceased or not.",
                    "id": "Patient.deceased[x]",
                    "isModifier": true,
                    "isModifierReason": "This element is labeled as a modifier because once a patient is marked as deceased, the actions that are appropriate to perform on the patient may be significantly different.",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-30  (bool) and PID-29 (datetime)"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/deceasedInd, player[classCode=PSN|ANM and determinerCode=INSTANCE]/deceasedTime"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.deceased[x]",
                    "requirements": "The fact that a patient is deceased influences the clinical process. Also, in human communication and relation management it is necessary to know whether the person is alive.",
                    "short": "Indicates if the individual is deceased or not",
                    "type": [
                        {
                            "code": "boolean"
                        },
                        {
                            "code": "dateTime"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.address"
                    },
                    "comment": "Patient may have multiple addresses with different uses or applicable periods.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "An address for the individual.",
                    "id": "Patient.address",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-11"
                        },
                        {
                            "identity": "rim",
                            "map": "addr"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantContactable.address"
                        },
                        {
                            "identity": "cda",
                            "map": ".addr"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.address",
                    "requirements": "May need to keep track of patient addresses for contacting, billing or reporting requirements and also to help with identification.",
                    "short": "An address for the individual",
                    "type": [
                        {
                            "code": "Address"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.maritalStatus"
                    },
                    "binding": {
                        "description": "The domestic partnership status of a person.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "MaritalStatus"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "extensible",
                        "valueSet": "http://hl7.org/fhir/ValueSet/marital-status"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "This field contains a patient's most recent marital (civil) status.",
                    "id": "Patient.maritalStatus",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-16"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN]/maritalStatusCode"
                        },
                        {
                            "identity": "cda",
                            "map": ".patient.maritalStatusCode"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.maritalStatus",
                    "requirements": "Most, if not all systems capture it.",
                    "short": "Marital (civil) status of a patient",
                    "type": [
                        {
                            "code": "CodeableConcept"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.multipleBirth[x]"
                    },
                    "comment": "Where the valueInteger is provided, the number is the birth number in the sequence. E.g. The middle birth in triplets would be valueInteger=2 and the third born would have valueInteger=3 If a boolean value was provided for this triplets example, then all 3 patient records would have valueBoolean=true (the ordering is not indicated).",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Indicates whether the patient is part of a multiple (boolean) or indicates the actual birth order (integer).",
                    "id": "Patient.multipleBirth[x]",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-24 (bool), PID-25 (integer)"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/multipleBirthInd,  player[classCode=PSN|ANM and determinerCode=INSTANCE]/multipleBirthOrderNumber"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.multipleBirth[x]",
                    "requirements": "For disambiguation of multiple-birth children, especially relevant where the care provider doesn't meet the patient, such as labs.",
                    "short": "Whether patient is part of a multiple birth",
                    "type": [
                        {
                            "code": "boolean"
                        },
                        {
                            "code": "integer"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.photo"
                    },
                    "comment": "Guidelines:\n* Use id photos, not clinical photos.\n* Limit dimensions to thumbnail.\n* Keep byte count low to ease resource updates.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Image of the patient.",
                    "id": "Patient.photo",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "OBX-5 - needs a profile"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/desc"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.photo"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.photo",
                    "requirements": "Many EHR systems have the capability to capture an image of the patient. Fits with newer social media usage too.",
                    "short": "Image of the patient",
                    "type": [
                        {
                            "code": "Attachment"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.contact"
                    },
                    "comment": "Contact covers all kinds of contact parties: family members, business contacts, guardians, caregivers. Not applicable to register pedigree and family ties beyond use of having contact.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "name.exists() or telecom.exists() or address.exists() or organization.exists()",
                            "human": "SHALL at least contain a contact's details or a reference to an organization",
                            "key": "pat-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Patient"
                        }
                    ],
                    "definition": "A contact party (e.g. guardian, partner, friend) for the patient.",
                    "extension": [
                        {
                            "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-explicit-type-name",
                            "valueString": "Contact"
                        }
                    ],
                    "id": "Patient.contact",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/scopedRole[classCode=CON]"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact",
                    "requirements": "Need to track people you can contact about the patient.",
                    "short": "A contact party (e.g. guardian, partner, friend) for the patient",
                    "type": [
                        {
                            "code": "BackboneElement"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Element.id"
                    },
                    "condition": [
                        "ele-1"
                    ],
                    "definition": "Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.",
                    "id": "Patient.contact.id",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "path": "Patient.contact.id",
                    "representation": [
                        "xmlAttr"
                    ],
                    "short": "Unique id for inter-element referencing",
                    "type": [
                        {
                            "code": "http://hl7.org/fhirpath/System.String",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-fhir-type",
                                    "valueUrl": "string"
                                }
                            ]
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Element.extension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.",
                    "id": "Patient.contact.extension",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "path": "Patient.contact.extension",
                    "short": "Additional content defined by implementations",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content",
                        "modifiers"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "BackboneElement.modifierExtension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the element and that modifies the understanding of the element in which it is contained and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.\n\nModifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
                    "id": "Patient.contact.modifierExtension",
                    "isModifier": true,
                    "isModifierReason": "Modifier extensions are expected to modify the meaning or interpretation of the element that contains them",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "N/A"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "path": "Patient.contact.modifierExtension",
                    "requirements": "Modifier extensions allow for extensions that *cannot* be safely ignored to be clearly distinguished from the vast majority of extensions which can be safely ignored.  This promotes interoperability by eliminating the need for implementers to prohibit the presence of extensions. For further information, see the [definition of modifier extensions](extensibility.html#modifierExtension).",
                    "short": "Extensions that cannot be ignored even if unrecognized",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.contact.relationship"
                    },
                    "binding": {
                        "description": "The nature of the relationship between a patient and a contact person for that patient.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "ContactRelationship"
                            }
                        ],
                        "strength": "extensible",
                        "valueSet": "http://hl7.org/fhir/ValueSet/patient-contactrelationship"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The nature of the relationship between the patient and the contact person.",
                    "id": "Patient.contact.relationship",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-7, NK1-3"
                        },
                        {
                            "identity": "rim",
                            "map": "code"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.relationship",
                    "requirements": "Used to determine which contact person is the most relevant to approach, depending on circumstances.",
                    "short": "The kind of relationship",
                    "type": [
                        {
                            "code": "CodeableConcept"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.contact.name"
                    },
                    "condition": [
                        "pat-1"
                    ],
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A name associated with the contact person.",
                    "id": "Patient.contact.name",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-2"
                        },
                        {
                            "identity": "rim",
                            "map": "name"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.name",
                    "requirements": "Contact persons need to be identified by name, but it is uncommon to need details about multiple other names for that contact person.",
                    "short": "A name associated with the contact person",
                    "type": [
                        {
                            "code": "HumanName"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.contact.telecom"
                    },
                    "comment": "Contact may have multiple ways to be contacted with different uses or applicable periods.  May need to have options for contacting the person urgently, and also to help with identification.",
                    "condition": [
                        "pat-1"
                    ],
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A contact detail for the person, e.g. a telephone number or an email address.",
                    "id": "Patient.contact.telecom",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-5, NK1-6, NK1-40"
                        },
                        {
                            "identity": "rim",
                            "map": "telecom"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.telecom",
                    "requirements": "People have (primary) ways to contact them in some way such as phone, email.",
                    "short": "A contact detail for the person",
                    "type": [
                        {
                            "code": "ContactPoint"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.contact.address"
                    },
                    "condition": [
                        "pat-1"
                    ],
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Address for the contact person.",
                    "id": "Patient.contact.address",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-4"
                        },
                        {
                            "identity": "rim",
                            "map": "addr"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.address",
                    "requirements": "Need to keep track where the contact person can be contacted per postal mail or visited.",
                    "short": "Address for the contact person",
                    "type": [
                        {
                            "code": "Address"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.contact.gender"
                    },
                    "binding": {
                        "description": "The gender of a person used for administrative purposes.",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "AdministrativeGender"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/administrative-gender|5.0.0"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Administrative Gender - the gender that the contact person is considered to have for administration and record keeping purposes.",
                    "id": "Patient.contact.gender",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-15"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/administrativeGender"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.gender",
                    "requirements": "Needed to address the person correctly.",
                    "short": "male | female | other | unknown",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.contact.organization"
                    },
                    "condition": [
                        "pat-1"
                    ],
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Organization on behalf of which the contact is acting or for which the contact is working.",
                    "id": "Patient.contact.organization",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "NK1-13, NK1-30, NK1-31, NK1-32, NK1-41"
                        },
                        {
                            "identity": "rim",
                            "map": "scoper"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.organization",
                    "requirements": "For guardians or business related contacts, the organization is relevant.",
                    "short": "Organization that is associated with the contact",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Organization"
                            ]
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.contact.period"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The period during which this contact person or organization is valid to be contacted relating to this patient.",
                    "id": "Patient.contact.period",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "effectiveTime"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.contact.period",
                    "short": "The period during which this contact person or organization is valid to be contacted relating to this patient",
                    "type": [
                        {
                            "code": "Period"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.communication"
                    },
                    "comment": "If no language is specified, this *implies* that the default local language is spoken.  If you need to convey proficiency for multiple modes, then you need multiple Patient.Communication associations.   For animals, language is not a relevant field, and should be absent from the instance. If the Patient does not speak the default local language, then the Interpreter Required Standard can be used to explicitly declare that an interpreter is required.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "A language which may be used to communicate with the patient about his or her health.",
                    "id": "Patient.communication",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "LanguageCommunication"
                        },
                        {
                            "identity": "interface",
                            "map": "ParticipantLiving.communication"
                        },
                        {
                            "identity": "cda",
                            "map": "patient.languageCommunication"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.communication",
                    "requirements": "If a patient does not speak the local language, interpreters may be required, so languages spoken and proficiency are important things to keep track of both for patient and other persons of interest.",
                    "short": "A language which may be used to communicate with the patient about his or her health",
                    "type": [
                        {
                            "code": "BackboneElement"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Element.id"
                    },
                    "condition": [
                        "ele-1"
                    ],
                    "definition": "Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.",
                    "id": "Patient.communication.id",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "path": "Patient.communication.id",
                    "representation": [
                        "xmlAttr"
                    ],
                    "short": "Unique id for inter-element referencing",
                    "type": [
                        {
                            "code": "http://hl7.org/fhirpath/System.String",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-fhir-type",
                                    "valueUrl": "string"
                                }
                            ]
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Element.extension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.",
                    "id": "Patient.communication.extension",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "path": "Patient.communication.extension",
                    "short": "Additional content defined by implementations",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content",
                        "modifiers"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "BackboneElement.modifierExtension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the element and that modifies the understanding of the element in which it is contained and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.\n\nModifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
                    "id": "Patient.communication.modifierExtension",
                    "isModifier": true,
                    "isModifierReason": "Modifier extensions are expected to modify the meaning or interpretation of the element that contains them",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "N/A"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "path": "Patient.communication.modifierExtension",
                    "requirements": "Modifier extensions allow for extensions that *cannot* be safely ignored to be clearly distinguished from the vast majority of extensions which can be safely ignored.  This promotes interoperability by eliminating the need for implementers to prohibit the presence of extensions. For further information, see the [definition of modifier extensions](extensibility.html#modifierExtension).",
                    "short": "Extensions that cannot be ignored even if unrecognized",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 1,
                        "path": "Patient.communication.language"
                    },
                    "binding": {
                        "additional": [
                            {
                                "purpose": "starter",
                                "valueSet": "http://hl7.org/fhir/ValueSet/languages"
                            }
                        ],
                        "description": "IETF language tag for a human language",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "Language"
                            },
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-isCommonBinding",
                                "valueBoolean": true
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/all-languages|5.0.0"
                    },
                    "comment": "The structure aa-BB with this exact casing is one the most widely used notations for locale. However not all systems actually code this but instead have it as free text. Hence CodeableConcept instead of code as the data type.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The ISO-639-1 alpha 2 code in lower case for the language, optionally followed by a hyphen and the ISO-3166-1 alpha 2 code for the region in upper case; e.g. \"en\" for English, or \"en-US\" for American English versus \"en-AU\" for Australian English.",
                    "id": "Patient.communication.language",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-15, LAN-2"
                        },
                        {
                            "identity": "rim",
                            "map": "player[classCode=PSN|ANM and determinerCode=INSTANCE]/languageCommunication/code"
                        },
                        {
                            "identity": "cda",
                            "map": ".languageCode"
                        }
                    ],
                    "max": "1",
                    "min": 1,
                    "mustSupport": false,
                    "path": "Patient.communication.language",
                    "requirements": "Most systems in multilingual countries will want to convey language. Not all systems actually need the regional dialect.",
                    "short": "The language which can be used to communicate with the patient about his or her health",
                    "type": [
                        {
                            "code": "CodeableConcept"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.communication.preferred"
                    },
                    "comment": "This language is specifically identified for communicating healthcare information.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Indicates whether or not the patient prefers this language (over other languages he masters up a certain level).",
                    "id": "Patient.communication.preferred",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-15"
                        },
                        {
                            "identity": "rim",
                            "map": "preferenceInd"
                        },
                        {
                            "identity": "cda",
                            "map": ".preferenceInd"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.communication.preferred",
                    "requirements": "People that master multiple languages up to certain level may prefer one or more, i.e. feel more confident in communicating in a particular language making other languages sort of a fall back method.",
                    "short": "Language preference indicator",
                    "type": [
                        {
                            "code": "boolean"
                        }
                    ]
                },
                {
                    "alias": [
                        "careProvider"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.generalPractitioner"
                    },
                    "comment": "This may be the primary care provider (in a GP context), or it may be a patient nominated care manager in a community/disability setting, or even organization that will provide people to perform the care provider roles.  It is not to be used to record Care Teams, these should be in a CareTeam resource that may be linked to the CarePlan or EpisodeOfCare resources.\nMultiple GPs may be recorded against the patient for various reasons, such as a student that has his home GP listed along with the GP at university during the school semesters, or a \"fly-in/fly-out\" worker that has the onsite GP also included with his home GP to remain aware of medical issues.\n\nJurisdictions may decide that they can profile this down to 1 if desired, or 1 per type.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Patient's nominated care provider.",
                    "id": "Patient.generalPractitioner",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PD1-4"
                        },
                        {
                            "identity": "rim",
                            "map": "subjectOf.CareEvent.performer.AssignedEntity"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.generalPractitioner",
                    "short": "Patient's nominated primary care provider",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Organization",
                                "http://hl7.org/fhir/StructureDefinition/Practitioner",
                                "http://hl7.org/fhir/StructureDefinition/PractitionerRole"
                            ]
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Patient.managingOrganization"
                    },
                    "comment": "There is only one managing organization for a specific patient record. Other organizations will have their own Patient record, and may use the Link property to join the records together (or a Person resource which can include confidence ratings for the association).",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Organization that is the custodian of the patient record.",
                    "id": "Patient.managingOrganization",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "scoper"
                        },
                        {
                            "identity": "cda",
                            "map": ".providerOrganization"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.managingOrganization",
                    "requirements": "Need to know who recognizes this patient record, manages and updates it.",
                    "short": "Organization that is the custodian of the patient record",
                    "type": [
                        {
                            "code": "Reference",
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Organization"
                            ]
                        }
                    ]
                },
                {
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Patient.link"
                    },
                    "comment": "There is no assumption that linked patient records have mutual links.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Link to a Patient or RelatedPerson resource that concerns the same actual individual.",
                    "id": "Patient.link",
                    "isModifier": true,
                    "isModifierReason": "This element is labeled as a modifier because it might not be the main Patient resource, and the referenced patient should be used instead of this Patient record. This is when the link.type value is 'replaced-by'",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "outboundLink"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "mustSupport": false,
                    "path": "Patient.link",
                    "requirements": "There are multiple use cases:   \n\n* Duplicate patient records due to the clerical errors associated with the difficulties of identifying humans consistently, and \n* Distribution of patient information across multiple servers.",
                    "short": "Link to a Patient or RelatedPerson resource that concerns the same actual individual",
                    "type": [
                        {
                            "code": "BackboneElement"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 0,
                        "path": "Element.id"
                    },
                    "condition": [
                        "ele-1"
                    ],
                    "definition": "Unique id for the element within a resource (for internal references). This may be any string value that does not contain spaces.",
                    "id": "Patient.link.id",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 0,
                    "path": "Patient.link.id",
                    "representation": [
                        "xmlAttr"
                    ],
                    "short": "Unique id for inter-element referencing",
                    "type": [
                        {
                            "code": "http://hl7.org/fhirpath/System.String",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-fhir-type",
                                    "valueUrl": "string"
                                }
                            ]
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "Element.extension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the element. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.",
                    "id": "Patient.link.extension",
                    "isModifier": false,
                    "isSummary": false,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "n/a"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "path": "Patient.link.extension",
                    "short": "Additional content defined by implementations",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "alias": [
                        "extensions",
                        "user content",
                        "modifiers"
                    ],
                    "base": {
                        "max": "*",
                        "min": 0,
                        "path": "BackboneElement.modifierExtension"
                    },
                    "comment": "There can be no stigma associated with the use of extensions by any application, project, or standard - regardless of the institution or jurisdiction that uses or defines the extensions.  The use of extensions is what allows the FHIR specification to retain a core level of simplicity for everyone.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        },
                        {
                            "expression": "extension.exists() != value.exists()",
                            "human": "Must have either extensions or value[x], not both",
                            "key": "ext-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Extension"
                        }
                    ],
                    "definition": "May be used to represent additional information that is not part of the basic definition of the element and that modifies the understanding of the element in which it is contained and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.\n\nModifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).",
                    "id": "Patient.link.modifierExtension",
                    "isModifier": true,
                    "isModifierReason": "Modifier extensions are expected to modify the meaning or interpretation of the element that contains them",
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "N/A"
                        }
                    ],
                    "max": "*",
                    "min": 0,
                    "path": "Patient.link.modifierExtension",
                    "requirements": "Modifier extensions allow for extensions that *cannot* be safely ignored to be clearly distinguished from the vast majority of extensions which can be safely ignored.  This promotes interoperability by eliminating the need for implementers to prohibit the presence of extensions. For further information, see the [definition of modifier extensions](extensibility.html#modifierExtension).",
                    "short": "Extensions that cannot be ignored even if unrecognized",
                    "type": [
                        {
                            "code": "Extension"
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 1,
                        "path": "Patient.link.other"
                    },
                    "comment": "Referencing a RelatedPerson here removes the need to use a Person record to associate a Patient and RelatedPerson as the same individual.",
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "Link to a Patient or RelatedPerson resource that concerns the same actual individual.",
                    "id": "Patient.link.other",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "v2",
                            "map": "PID-3, MRG-1"
                        },
                        {
                            "identity": "rim",
                            "map": "id"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 1,
                    "mustSupport": false,
                    "path": "Patient.link.other",
                    "short": "The other patient or related person resource that the link refers to",
                    "type": [
                        {
                            "code": "Reference",
                            "extension": [
                                {
                                    "url": "http://hl7.org/fhir/StructureDefinition/structuredefinition-hierarchy",
                                    "valueBoolean": false
                                }
                            ],
                            "targetProfile": [
                                "http://hl7.org/fhir/StructureDefinition/Patient",
                                "http://hl7.org/fhir/StructureDefinition/RelatedPerson"
                            ]
                        }
                    ]
                },
                {
                    "base": {
                        "max": "1",
                        "min": 1,
                        "path": "Patient.link.type"
                    },
                    "binding": {
                        "description": "The type of link between this patient resource and another Patient resource, or Patient/RelatedPerson when using the `seealso` code",
                        "extension": [
                            {
                                "url": "http://hl7.org/fhir/StructureDefinition/elementdefinition-bindingName",
                                "valueString": "LinkType"
                            }
                        ],
                        "strength": "required",
                        "valueSet": "http://hl7.org/fhir/ValueSet/link-type|5.0.0"
                    },
                    "constraint": [
                        {
                            "expression": "hasValue() or (children().count() > id.count())",
                            "human": "All FHIR elements must have a @value or children",
                            "key": "ele-1",
                            "severity": "error",
                            "source": "http://hl7.org/fhir/StructureDefinition/Element"
                        }
                    ],
                    "definition": "The type of link between this patient resource and another patient resource.",
                    "id": "Patient.link.type",
                    "isModifier": false,
                    "isSummary": true,
                    "mapping": [
                        {
                            "identity": "rim",
                            "map": "typeCode"
                        },
                        {
                            "identity": "cda",
                            "map": "n/a"
                        }
                    ],
                    "max": "1",
                    "min": 1,
                    "mustSupport": false,
                    "path": "Patient.link.type",
                    "short": "replaced-by | replaces | refer | seealso",
                    "type": [
                        {
                            "code": "code"
                        }
                    ]
                }
            ]
        },
        "status": "active",
        "text": {
            "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\"><table border=\"0\" cellpadding=\"0\" cellspacing=\"0\" style=\"border: 0px #F0F0F0 solid; font-size: 11px; font-family: verdana; vertical-align: top;\"><tr style=\"border: 1px #F0F0F0 solid; font-size: 11px; font-family: verdana; vertical-align: top\"><th style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"The logical name of the element\">Name</a></th><th style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"Information about the use of the element\">Flags</a></th><th style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"Minimum and Maximum # of times the the element can appear in the instance\">Card.</a></th><th style=\"width: 100px\" class=\"hierarchy\"><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"Reference to the type of the element\">Type</a></th><th style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"Additional information about the element\">Description &amp; Constraints</a><span style=\"float: right\"><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"Legend for this format\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3goXBCwdPqAP0wAAAldJREFUOMuNk0tIlFEYhp9z/vE2jHkhxXA0zJCMitrUQlq4lnSltEqCFhFG2MJFhIvIFpkEWaTQqjaWZRkp0g26URZkTpbaaOJkDqk10szoODP//7XIMUe0elcfnPd9zsfLOYplGrpRwZaqTtw3K7PtGem7Q6FoidbGgqHVy/HRb669R+56zx7eRV1L31JGxYbBtjKK93cxeqfyQHbehkZbUkK20goELEuIzEd+dHS+qz/Y8PTSif0FnGkbiwcAjHaU1+QWOptFiyCLp/LnKptpqIuXHx6rbR26kJcBX3yLgBfnd7CxwJmflpP2wUg0HIAoUUpZBmKzELGWcN8nAr6Gpu7tLU/CkwAaoKTWRSQyt89Q8w6J+oVQkKnBoblH7V0PPvUOvDYXfopE/SJmALsxnVm6LbkotrUtNowMeIrVrBcBpaMmdS0j9df7abpSuy7HWehwJdt1lhVwi/J58U5beXGAF6c3UXLycw1wdFklArBn87xdh0ZsZtArghBdAA3+OEDVubG4UEzP6x1FOWneHh2VDAHBAt80IbdXDcesNoCvs3E5AFyNSU5nbrDPZpcUEQQTFZiEVx+51fxMhhyJEAgvlriadIJZZksRuwBYMOPBbO3hePVVqgEJhFeUuFLhIPkRP6BQLIBrmMenujm/3g4zc398awIe90Zb5A1vREALqneMcYgP/xVQWlG+Ncu5vgwwlaUNx+3799rfe96u9K0JSDXcOzOTJg4B6IgmXfsygc7/Bvg9g9E58/cDVmGIBOP/zT8Bz1zqWqpbXIsd0O9hajXfL6u4BaOS6SeWAAAAAElFTkSuQmCC\" alt=\"doco\" style=\"background-color: inherit\"/></a></span></th></tr><tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAJUlEQVR4Xu3IIQEAAAgDsHd9/w4EQIOamFnaBgAA4MMKAACAKwNp30CqZFfFmwAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAACCElEQVQ4y4XTv2sUQRTA8e9Mzt3kjoOLSXFgZ6GJQlALCysLC89OsLTXv0VFxE4stRAEQUghSWEXuM4qMZpATsUD70dyMdnduZ15z2IvMV5IfDDNm5nPm59GVTkpms1mTVXvhxDuichlEZn03m+KyJL3/mWj0fiKqp7YVlZWXrfbbR2PTqeji4uLn1WVEqdECKFRr9eP5WdnZ/HeXwROB0TEA3S7XarVKiLC1tYW8/PzeO/5LxBCUABrLXEc02q1KJfLB30F0P144dPU9LVL1kwcrU06WP0ewhML4JwDYDgcHo7I87wAjNq5ypU3Z8arT8F5u/xejw52zmGM+Rcg1wyIcc/BTYCdBlODyh3ElA1AHMekaUoURURRBECWZSNgaGzBxxAU9jfQ9jrJr2dcbbXobRYHlQAzo9X1gDR9+KUArE6CwLefZD9WCW6P0uRZKreXqADkHXZ3dshzjwRholJH397AOXcTwHTfzQ1n7q6NnYEAy+DWQVNwKWQJ6vcx557Se7HAzIN1M9rCwVteA/rAYDRRICQgSZEr7WLYO3bzJVJGQBu0D74PkoHkoBnIHvjfkO9AGABmDHCjFWgH8i7kPQh9yEeYH4DfLhBJgA2A7BBQJ9uwXWY3rhJqFo1AaiB1CBngwKZQcqAeSFSduL9Akj7qPF64jnALS5VTPwdgPwwJ+uog9Qcx4kRZiPKqxgAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Resource\" class=\"hierarchy\"/> <span title=\"Patient : Demographics and other administrative information about an individual or animal receiving care or other health-related services.\">Patient</span><a name=\"Patient\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; border: 1px grey solid; font-weight: bold; color: black; background-color: #e6ffe6\" href=\"versions.html#std-process\" title=\"Standards Status = Normative\">N</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"domainresource.html\">DomainResource</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Information about an individual or animal receiving health care services<br/><br/>Elements defined in Ancestors: <a href=\"resource.html#Resource\" title=\"The logical id of the resource, as used in the URL for the resource. Once assigned, this value never changes.\">id</a>, <a href=\"resource.html#Resource\" title=\"The metadata about the resource. This is content that is maintained by the infrastructure. Changes to the content might not always be associated with version changes to the resource.\">meta</a>, <a href=\"resource.html#Resource\" title=\"A reference to a set of rules that were followed when the resource was constructed, and which must be understood when processing the content. Often, this is a reference to an implementation guide that defines the special rules along with other profiles etc.\">implicitRules</a>, <a href=\"resource.html#Resource\" title=\"The base language in which the resource is written.\">language</a>, <a href=\"domainresource.html#DomainResource\" title=\"A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human. The narrative need not encode all the structured data, but is required to contain sufficient detail to make it &quot;clinically safe&quot; for a human to just read the narrative. Resource definitions may define what content should be represented in the narrative to ensure clinical safety.\">text</a>, <a href=\"domainresource.html#DomainResource\" title=\"These resources do not have an independent existence apart from the resource that contains them - they cannot be identified independently, nor can they have their own independent transaction scope. This is allowed to be a Parameters resource if and only if it is referenced by a resource that provides context/meaning.\">contained</a>, <a href=\"domainresource.html#DomainResource\" title=\"May be used to represent additional information that is not part of the basic definition of the resource. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer can define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension.\">extension</a>, <a href=\"domainresource.html#DomainResource\" title=\"May be used to represent additional information that is not part of the basic definition of the resource and that modifies the understanding of the element that contains it and/or the understanding of the containing element's descendants. Usually modifier elements provide negation or qualification. To make the use of extensions safe and managable, there is a strict set of governance applied to the definition and use of extensions. Though any implementer is allowed to define an extension, there is a set of requirements that SHALL be met as part of the definition of the extension. Applications processing a resource are required to check for modifier extensions.\n\nModifier extensions SHALL NOT change the meaning of any elements on Resource or DomainResource (including cannot change the meaning of modifierExtension itself).\">modifierExtension</a></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.identifier : An identifier for this patient.\">identifier</span><a name=\"Patient.identifier\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#Identifier\">Identifier</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">An identifier for this patient<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Patient.active : Whether this patient record is in active use. \nMany systems use this property to mark as non-current patients, such as those that have not been seen for a period of time based on an organization's business rules.\n\nIt is often used to filter patient lists to exclude inactive patients\n\nDeceased patients may also be marked as inactive for the same reasons, but may be active for some time after death.\">active</span><a name=\"Patient.active\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"conformance-rules.html#isModifier\" title=\"This element is a modifier element\">?!</a><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#boolean\">boolean</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Whether this patient's record is in active use<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.name : A name associated with the individual.\">name</span><a name=\"Patient.name\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#HumanName\">HumanName</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">A name associated with the patient<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.telecom : A contact detail (e.g. a telephone number or an email address) by which the individual may be contacted.\">telecom</span><a name=\"Patient.telecom\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#ContactPoint\">ContactPoint</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">A contact detail for the individual<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Patient.gender : Administrative Gender - the gender that the patient is considered to have for administration and record keeping purposes.\">gender</span><a name=\"Patient.gender\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#code\">code</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">male | female | other | unknown<br/>Binding: <a href=\"valueset-administrative-gender.html\">AdministrativeGender</a> (<a href=\"terminologies.html#required\" title=\"To be conformant, the concept in this element SHALL be from the specified value set.\">Required</a>)<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Patient.birthDate : The date of birth for the individual.\">birthDate</span><a name=\"Patient.birthDate\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#date\">date</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">The date of birth for the individual<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAMQfAGm6/idTd4yTmF+v8Xa37KvW+lyh3KHJ62aq41ee2bXZ98nm/2mt5W2Ck5XN/C1chEZieho8WXXA/2Gn4P39/W+y6V+l3qjP8Njt/lx2izxPYGyv51Oa1EJWZ////////yH5BAEAAB8ALAAAAAAQABAAAAWH4Cd+Xml6Y0pCQts0EKp6GbYshaM/skhjhCChUmFIeL4OsHIxXRAISQTl6SgIG8+FgfBMoh2qtbLZQr0TQJhk3TC4pYPBApiyFVDEwSOf18UFXxMWBoUJBn9sDgmDewcJCRyJJBoEkRyYmAABPZQEAAOhA5seFDMaDw8BAQ9TpiokJyWwtLUhADs=\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Choice of Types\" class=\"hierarchy\"/> <span title=\"Patient.deceased[x] : Indicates if the individual is deceased or not.\">deceased[x]</span><a name=\"Patient.deceased_x_\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"conformance-rules.html#isModifier\" title=\"This element is a modifier element\">?!</a><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Indicates if the individual is deceased or not<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Value of &quot;true&quot; or &quot;false&quot;\">deceasedBoolean</span></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#boolean\">boolean</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzME+lXFigAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+3OsRUAIAjEUOL+O8cJABttJM11/x1qZAGqRBEVcNIqdWj1efDqQbb3HwwwwEfABmQUHSPM9dtDAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"A date, date-time or partial date (e.g. just year or year + month).  If hours and minutes are specified, a UTC offset SHALL be populated. The format is a union of the schema types gYear, gYearMonth, date and dateTime. Seconds must be provided due to schema type constraints but may be zero-filled and may be ignored.                 Dates SHALL be valid dates.\">deceasedDateTime</span></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#dateTime\">dateTime</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.address : An address for the individual.\">address</span><a name=\"Patient.address\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#Address\">Address</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">An address for the individual<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.maritalStatus : This field contains a patient's most recent marital (civil) status.\">maritalStatus</span><a name=\"Patient.maritalStatus\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#CodeableConcept\">CodeableConcept</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Marital (civil) status of a patient<br/>Binding: <a href=\"valueset-marital-status.html\">Marital Status Codes</a> (<a href=\"terminologies.html#extensible\" title=\"To be conformant, the concept in this element SHALL be from the specified value set if any of the codes within the value set can apply to the concept being communicated.  If the value set does not cover the concept (based on human review), alternate codings (or, data type allowing, text) may be included instead.\">Extensible</a>)<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAMQfAGm6/idTd4yTmF+v8Xa37KvW+lyh3KHJ62aq41ee2bXZ98nm/2mt5W2Ck5XN/C1chEZieho8WXXA/2Gn4P39/W+y6V+l3qjP8Njt/lx2izxPYGyv51Oa1EJWZ////////yH5BAEAAB8ALAAAAAAQABAAAAWH4Cd+Xml6Y0pCQts0EKp6GbYshaM/skhjhCChUmFIeL4OsHIxXRAISQTl6SgIG8+FgfBMoh2qtbLZQr0TQJhk3TC4pYPBApiyFVDEwSOf18UFXxMWBoUJBn9sDgmDewcJCRyJJBoEkRyYmAABPZQEAAOhA5seFDMaDw8BAQ9TpiokJyWwtLUhADs=\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Choice of Types\" class=\"hierarchy\"/> <span title=\"Patient.multipleBirth[x] : Indicates whether the patient is part of a multiple (boolean) or indicates the actual birth order (integer).\">multipleBirth[x]</span><a name=\"Patient.multipleBirth_x_\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Whether patient is part of a multiple birth<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Value of &quot;true&quot; or &quot;false&quot;\">multipleBirthBoolean</span></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#boolean\">boolean</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzME+lXFigAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+3OsRUAIAjEUOL+O8cJABttJM11/x1qZAGqRBEVcNIqdWj1efDqQbb3HwwwwEfABmQUHSPM9dtDAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"A whole number\">multipleBirthInteger</span></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#integer\">integer</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.photo : Image of the patient.\">photo</span><a name=\"Patient.photo\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#Attachment\">Attachment</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Image of the patient<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPQfAOvGUf7ztuvPMf/78/fkl/Pbg+u8Rvjqteu2Pf3zxPz36Pz0z+vTmPzurPvuw/npofbjquvNefHVduuyN+uuMu3Oafbgjfnqvf/3zv/3xevPi+vRjP/20/bmsP///wD/ACH5BAEKAB8ALAAAAAAQABAAAAVl4CeOZGme5qCqqDg8jyVJaz1876DsmAQAgqDgltspMEhMJoMZ4iy6I1AooFCIv+wKybziALVAoAEjYLwDgGIpJhMslgxaLR4/3rMAWoBp32V5exg8Shl1ckRUQVaMVkQ2kCstKCEAOw==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Element\" class=\"hierarchy\"/> <span title=\"Patient.contact : A contact party (e.g. guardian, partner, friend) for the patient.\">contact</span><a name=\"Patient.contact\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; border: 1px maroon solid; font-weight: bold; color: #301212; background-color: #fdf4f4;\" href=\"conformance-rules.html#constraints\" title=\"This element has or is affected by some invariants\">C</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"types.html#BackBoneElement\">BackboneElement</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">A contact party (e.g. guardian, partner, friend) for the patient<br/><span style=\"font-style: italic\" title=\"pat-1\">+ Rule: SHALL at least contain a contact's details or a reference to an organization</span><br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.contact.relationship : The nature of the relationship between the patient and the contact person.\">relationship</span><a name=\"Patient.contact.relationship\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#CodeableConcept\">CodeableConcept</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">The kind of relationship<br/>Binding: <a href=\"valueset-patient-contactrelationship.html\">Patient Contact Relationship </a> (<a href=\"terminologies.html#extensible\" title=\"To be conformant, the concept in this element SHALL be from the specified value set if any of the codes within the value set can apply to the concept being communicated.  If the value set does not cover the concept (based on human review), alternate codings (or, data type allowing, text) may be included instead.\">Extensible</a>)<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.contact.name : A name associated with the contact person.\">name</span><a name=\"Patient.contact.name\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; border: 1px maroon solid; font-weight: bold; color: #301212; background-color: #fdf4f4;\" href=\"conformance-rules.html#constraints\" title=\"This element has or is affected by some invariants\">C</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#HumanName\">HumanName</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">A name associated with the contact person<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.contact.telecom : A contact detail for the person, e.g. a telephone number or an email address.\">telecom</span><a name=\"Patient.contact.telecom\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; border: 1px maroon solid; font-weight: bold; color: #301212; background-color: #fdf4f4;\" href=\"conformance-rules.html#constraints\" title=\"This element has or is affected by some invariants\">C</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#ContactPoint\">ContactPoint</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">A contact detail for the person<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.contact.address : Address for the contact person.\">address</span><a name=\"Patient.contact.address\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; border: 1px maroon solid; font-weight: bold; color: #301212; background-color: #fdf4f4;\" href=\"conformance-rules.html#constraints\" title=\"This element has or is affected by some invariants\">C</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#Address\">Address</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Address for the contact person<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Patient.contact.gender : Administrative Gender - the gender that the contact person is considered to have for administration and record keeping purposes.\">gender</span><a name=\"Patient.contact.gender\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#code\">code</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">male | female | other | unknown<br/>Binding: <a href=\"valueset-administrative-gender.html\">AdministrativeGender</a> (<a href=\"terminologies.html#required\" title=\"To be conformant, the concept in this element SHALL be from the specified value set.\">Required</a>)<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjEwMPRyoQAAAFxJREFUOE/NjEEOACEIA/0o/38GGw+agoXYeNnDJDCUDnd/gkoFKhWozJiZI3gLwY6rAgxhsPKTPUzycTl8lAryMyMsVQG6TFi6cHULyz8KOjC7OIQKlQpU3uPjAwhX2CCcGsgOAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Reference to another Resource\" class=\"hierarchy\"/> <span title=\"Patient.contact.organization : Organization on behalf of which the contact is acting or for which the contact is working.\">organization</span><a name=\"Patient.contact.organization\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; border: 1px maroon solid; font-weight: bold; color: #301212; background-color: #fdf4f4;\" href=\"conformance-rules.html#constraints\" title=\"This element has or is affected by some invariants\">C</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"references.html#Reference\">Reference</a>(<a href=\"organization.html\">Organization</a>)</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Organization that is associated with the contact<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzME+lXFigAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+3OsRUAIAjEUOL+O8cJABttJM11/x1qZAGqRBEVcNIqdWj1efDqQbb3HwwwwEfABmQUHSPM9dtDAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.contact.period : The period during which this contact person or organization is valid to be contacted relating to this patient.\">period</span><a name=\"Patient.contact.period\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#Period\">Period</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">The period during which this contact person or organization is valid to be contacted relating to this patient<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPQfAOvGUf7ztuvPMf/78/fkl/Pbg+u8Rvjqteu2Pf3zxPz36Pz0z+vTmPzurPvuw/npofbjquvNefHVduuyN+uuMu3Oafbgjfnqvf/3zv/3xevPi+vRjP/20/bmsP///wD/ACH5BAEKAB8ALAAAAAAQABAAAAVl4CeOZGme5qCqqDg8jyVJaz1876DsmAQAgqDgltspMEhMJoMZ4iy6I1AooFCIv+wKybziALVAoAEjYLwDgGIpJhMslgxaLR4/3rMAWoBp32V5exg8Shl1ckRUQVaMVkQ2kCstKCEAOw==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Element\" class=\"hierarchy\"/> <span title=\"Patient.communication : A language which may be used to communicate with the patient about his or her health.\">communication</span><a name=\"Patient.communication\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"types.html#BackBoneElement\">BackboneElement</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">A language which may be used to communicate with the patient about his or her health<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAMUlEQVR4Xu3LMQoAIBADwftr/v8GtdbqEAthAtMspJJUx9rYW8ftHwAA+NcRAAAAXplLq0BWj/rZigAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPZ/APrkusOiYvvfqbiXWaV2G+jGhdq1b8GgYf3v1frw3vTUlsWkZNewbcSjY/DQkad4Hb6dXv3u0f3v1ObEgfPTlerJiP3w1v79+e7OkPrfrfnjuNOtZPrpydaxa+/YrvvdpP779ZxvFPvnwKKBQaFyF/369M2vdaqHRPz58/HNh/vowufFhfroxO3OkPrluv779tK0e6JzGProwvrow9m4eOnIifPTlPDPkP78+Naxaf3v0/zowfXRi+bFhLWUVv379/rnwPvszv3rye3LiPvnv+3MjPDasKiIS/789/3x2f747eXDg+7Mifvu0tu7f+/QkfDTnPXWmPrjsvrjtPbPgrqZW+/QlPz48K2EMv36866OUPvowat8Ivvgq/Pbrvzgq/PguvrgrqN0Gda2evfYm9+7d/rpw9q6e/LSku/Rl/XVl/LSlfrkt+zVqe7Wqv3x1/bNffbOf59wFdS6if3u0vrqyP3owPvepfXQivDQkO/PkKh9K7STVf779P///wD/ACH5BAEKAH8ALAAAAAAQABAAAAemgH+CgxeFF4OIhBdKGwFChYl/hYwbdkoBPnaQkosbG3d3VEpSUlonUoY1Gzo6QkI8SrGxWBOFG4uySgY5ZWR3PFy2hnaWZXC/PHcPwkpJk1ShoHcxhQEXSUmtFy6+0iSFVResrjoTPDzdcoU+F65CduVU6KAhhQa3F8Tx8nchBoYuqoTLZoAKFRIhqGwqJAULFx0GYpBQeChRIR4TJm6KJMhQRUSBAAA7\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Data Type\" class=\"hierarchy\"/> <span title=\"Patient.communication.language : The ISO-639-1 alpha 2 code in lower case for the language, optionally followed by a hyphen and the ISO-3166-1 alpha 2 code for the region in upper case; e.g. &quot;en&quot; for English, or &quot;en-US&quot; for American English versus &quot;en-AU&quot; for Australian English.\">language</span><a name=\"Patient.communication.language\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">1..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#CodeableConcept\">CodeableConcept</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">The language which can be used to communicate with the patient about his or her health<br/>Binding: <a href=\"valueset-all-languages.html\">All Languages</a> (<a href=\"terminologies.html#required\" title=\"To be conformant, the concept in this element SHALL be from the specified value set.\">Required</a>)<table class=\"grid\"><tr><td style=\"font-size: 11px\"><b>Additional Bindings</b></td><td style=\"font-size: 11px\">Purpose</td></tr><tr><td style=\"font-size: 11px\"><a href=\"valueset-languages.html\" title=\"http://hl7.org/fhir/ValueSet/languages\">Common Languages</a></td><td style=\"font-size: 11px\"><a href=\"valueset-additional-binding-purpose.html#additional-binding-purpose-starter\" title=\"This value set is a good set of codes to start with when designing your system\">Starter Set</a></td></tr></table><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzMPbYccAgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAMElEQVQ4y+3OQREAIBDDwAv+PQcFFN5MIyCzqHMKUGVCpMFLK97heq+gggoq+EiwAVjvMhFGmlEUAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzME+lXFigAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+3OsRUAIAjEUOL+O8cJABttJM11/x1qZAGqRBEVcNIqdWj1efDqQbb3HwwwwEfABmQUHSPM9dtDAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Patient.communication.preferred : Indicates whether or not the patient prefers this language (over other languages he masters up a certain level).\">preferred</span><a name=\"Patient.communication.preferred\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#boolean\">boolean</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Language preference indicator<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjEwMPRyoQAAAFxJREFUOE/NjEEOACEIA/0o/38GGw+agoXYeNnDJDCUDnd/gkoFKhWozJiZI3gLwY6rAgxhsPKTPUzycTl8lAryMyMsVQG6TFi6cHULyz8KOjC7OIQKlQpU3uPjAwhX2CCcGsgOAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Reference to another Resource\" class=\"hierarchy\"/> <span title=\"Patient.generalPractitioner : Patient's nominated care provider.\">generalPractitioner</span><a name=\"Patient.generalPractitioner\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"/><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"references.html#Reference\">Reference</a>(<a href=\"organization.html\">Organization</a> | <a href=\"practitioner.html\">Practitioner</a> | <a href=\"practitionerrole.html\">PractitionerRole</a>)</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Patient's nominated primary care provider<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALUlEQVR4Xu3IoREAIAwEwfT6/ddA0GBAxO3NrLlKUj9263wAAAAvrgEAADClAVWFQIBRHMicAAAAAElFTkSuQmCC)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjEwMPRyoQAAAFxJREFUOE/NjEEOACEIA/0o/38GGw+agoXYeNnDJDCUDnd/gkoFKhWozJiZI3gLwY6rAgxhsPKTPUzycTl8lAryMyMsVQG6TFi6cHULyz8KOjC7OIQKlQpU3uPjAwhX2CCcGsgOAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Reference to another Resource\" class=\"hierarchy\"/> <span title=\"Patient.managingOrganization : Organization that is the custodian of the patient record.\">managingOrganization</span><a name=\"Patient.managingOrganization\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"references.html#Reference\">Reference</a>(<a href=\"organization.html\">Organization</a>)</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Organization that is the custodian of the patient record<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAJUlEQVR4Xu3IIQEAAAgDsHd9/w4EQIOamFnaBgAA4MMKAACAKwNp30CqZFfFmwAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzME+lXFigAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+3OsRUAIAjEUOL+O8cJABttJM11/x1qZAGqRBEVcNIqdWj1efDqQbb3HwwwwEfABmQUHSPM9dtDAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,R0lGODlhEAAQAPQfAOvGUf7ztuvPMf/78/fkl/Pbg+u8Rvjqteu2Pf3zxPz36Pz0z+vTmPzurPvuw/npofbjquvNefHVduuyN+uuMu3Oafbgjfnqvf/3zv/3xevPi+vRjP/20/bmsP///wD/ACH5BAEKAB8ALAAAAAAQABAAAAVl4CeOZGme5qCqqDg8jyVJaz1876DsmAQAgqDgltspMEhMJoMZ4iy6I1AooFCIv+wKybziALVAoAEjYLwDgGIpJhMslgxaLR4/3rMAWoBp32V5exg8Shl1ckRUQVaMVkQ2kCstKCEAOw==\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Element\" class=\"hierarchy\"/> <span title=\"Patient.link : Link to a Patient or RelatedPerson resource that concerns the same actual individual.\">link</span><a name=\"Patient.link\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"conformance-rules.html#isModifier\" title=\"This element is a modifier element\">?!</a><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">0..*</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"types.html#BackBoneElement\">BackboneElement</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">Link to a Patient or RelatedPerson resource that concerns the same actual individual<br/><br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: white\"><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAALElEQVR4Xu3IsQ0AIAwEsez6+89AqKGGJj7JzVWS+mm3zgcAAMxwDQAAgFcaYAVAgNGLTjgAAAAASUVORK5CYII=)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIZgEiYEgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAIElEQVQ4y2P8//8/AyWAiYFCMGrAqAGjBowaMGoAAgAALL0DKYQ0DPIAAAAASUVORK5CYII=\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzI3XJ6V3QAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+2RsQ0AIAzDav7/2VzQwoCY4iWbZSmo1QGoUgNMghvWaIejPQW/CrrNCylIwcOCDYfLNRcNer4SAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAadEVYdFNvZnR3YXJlAFBhaW50Lk5FVCB2My41LjEwMPRyoQAAAFxJREFUOE/NjEEOACEIA/0o/38GGw+agoXYeNnDJDCUDnd/gkoFKhWozJiZI3gLwY6rAgxhsPKTPUzycTl8lAryMyMsVQG6TFi6cHULyz8KOjC7OIQKlQpU3uPjAwhX2CCcGsgOAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: white; background-color: inherit\" title=\"Reference to another Resource\" class=\"hierarchy\"/> <span title=\"Patient.link.other : Link to a Patient or RelatedPerson resource that concerns the same actual individual.\">other</span><a name=\"Patient.link.other\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">1..1</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"references.html#Reference\">Reference</a>(<a href=\"patient.html\">Patient</a> | <a href=\"relatedperson.html\">RelatedPerson</a>)</td><td style=\"vertical-align: top; text-align : left; background-color: white; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">The other patient or related person resource that the link refers to<br/></td></tr>\r\n<tr style=\"border: 0px #F0F0F0 solid; padding:0px; vertical-align: top; background-color: #F7F7F7\"><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px; white-space: nowrap; background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAyAAAAACCAYAAACg/LjIAAAAJUlEQVR4Xu3IIQEAAAgDsHd9/w4EQIOamFnaBgAA4MMKAACAKwNp30CqZFfFmwAAAABJRU5ErkJggg==)\" class=\"hierarchy\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAWCAYAAAABxvaqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIs1vtcMQAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAE0lEQVQI12P4//8/AxMDAwNdCABMPwMo2ctnoQAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzIZgEiYEgAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAIElEQVQ4y2P8//8/AyWAiYFCMGrAqAGjBowaMGoAAgAALL0DKYQ0DPIAAAAASUVORK5CYII=\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAWCAYAAADJqhx8AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3wYeFzME+lXFigAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAANklEQVQ4y+3OsRUAIAjEUOL+O8cJABttJM11/x1qZAGqRBEVcNIqdWj1efDqQbb3HwwwwEfABmQUHSPM9dtDAAAAAElFTkSuQmCC\" alt=\".\" style=\"background-color: inherit\" class=\"hierarchy\"/><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAACXBIWXMAAAsTAAALEwEAmpwYAAAARklEQVQ4y2P8//8/AyWAhYFCMAgMuHjx4n+KXaCv+I0szW8WpCG8kFO1lGFKW/SIjAUYgxz/MzAwMDC+nqhDUTQyjuYFBgCNmhP4OvTRgwAAAABJRU5ErkJggg==\" alt=\".\" style=\"background-color: #F7F7F7; background-color: inherit\" title=\"Primitive Data Type\" class=\"hierarchy\"/> <span title=\"Patient.link.type : The type of link between this patient resource and another patient resource.\">type</span><a name=\"Patient.link.type\"> </a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a style=\"padding-left: 3px; padding-right: 3px; color: black; null\" href=\"elementdefinition-definitions.html#ElementDefinition.isSummary\" title=\"This element is included in summaries\">Σ</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">1..1</td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\"><a href=\"datatypes.html#code\">code</a></td><td style=\"vertical-align: top; text-align : left; background-color: #F7F7F7; border: 0px #F0F0F0 solid; padding:0px 4px 0px 4px\" class=\"hierarchy\">replaced-by | replaces | refer | seealso<br/>Binding: <a href=\"valueset-link-type.html\">Link Type</a> (<a href=\"terminologies.html#required\" title=\"To be conformant, the concept in this element SHALL be from the specified value set.\">Required</a>)<br/></td></tr>\r\n<tr><td colspan=\"5\" class=\"hierarchy\"><br/><a href=\"https://build.fhir.org/ig/FHIR/ig-guidance/readingIgs.html#table-views\" title=\"Legend for this format\"><img src=\"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3goXBCwdPqAP0wAAAldJREFUOMuNk0tIlFEYhp9z/vE2jHkhxXA0zJCMitrUQlq4lnSltEqCFhFG2MJFhIvIFpkEWaTQqjaWZRkp0g26URZkTpbaaOJkDqk10szoODP//7XIMUe0elcfnPd9zsfLOYplGrpRwZaqTtw3K7PtGem7Q6FoidbGgqHVy/HRb669R+56zx7eRV1L31JGxYbBtjKK93cxeqfyQHbehkZbUkK20goELEuIzEd+dHS+qz/Y8PTSif0FnGkbiwcAjHaU1+QWOptFiyCLp/LnKptpqIuXHx6rbR26kJcBX3yLgBfnd7CxwJmflpP2wUg0HIAoUUpZBmKzELGWcN8nAr6Gpu7tLU/CkwAaoKTWRSQyt89Q8w6J+oVQkKnBoblH7V0PPvUOvDYXfopE/SJmALsxnVm6LbkotrUtNowMeIrVrBcBpaMmdS0j9df7abpSuy7HWehwJdt1lhVwi/J58U5beXGAF6c3UXLycw1wdFklArBn87xdh0ZsZtArghBdAA3+OEDVubG4UEzP6x1FOWneHh2VDAHBAt80IbdXDcesNoCvs3E5AFyNSU5nbrDPZpcUEQQTFZiEVx+51fxMhhyJEAgvlriadIJZZksRuwBYMOPBbO3hePVVqgEJhFeUuFLhIPkRP6BQLIBrmMenujm/3g4zc398awIe90Zb5A1vREALqneMcYgP/xVQWlG+Ncu5vgwwlaUNx+3799rfe96u9K0JSDXcOzOTJg4B6IgmXfsygc7/Bvg9g9E58/cDVmGIBOP/zT8Bz1zqWqpbXIsd0O9hajXfL6u4BaOS6SeWAAAAAElFTkSuQmCC\" alt=\"doco\" style=\"background-color: inherit\"/> Documentation for this format</a></td></tr></table></div>",
            "status": "generated"
        },
        "type": "Patient",
        "url": "http://hl7.org/fhir/StructureDefinition/Patient",
        "version": "5.0.0"
    }
"""

#with open('fhir/Patient-example.json', encoding='utf8', mode='r') as f: 
#    fhir_patient_json = json.load(f)

with open('fhir/valuesets.json', encoding='utf8', mode='r') as f: 
    fhir_value_set = json.load(f)

#tensor = fhir_patient_to_tensor(fhir_patient_json, fhir_structure_definition_json, fhir_value_set)
#print(tensor)




# Set hyperparameters
input_dim = 100  # Dimension of the random noise input for the generator
output_dim = 100  # Dimension of the generated output
lr = 0.0002  # Learning rate
batch_size = 64  # Batch size for training

device = torch.device("cuda:0")

# Initialize generator and discriminator
generator = Generator(input_dim, output_dim)#.to(device)
discriminator = Discriminator(output_dim)#.to(device)

# Define loss function and optimizers
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

# Load the FHIR dataset
dataset = FHIRDataset('Patient.ndjson')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_idx, real_data in enumerate(dataloader):
        batch_size = real_data.size(0)

        # Train discriminator with real data
        discriminator.zero_grad()
        real_labels = torch.ones(batch_size, 1)
        real_output = discriminator(real_data)
        real_loss = criterion(real_output, real_labels)
        real_loss.backward()

        # Train discriminator with generated data
        noise = torch.randn(batch_size, input_dim)
        fake_data = generator(noise).detach()
        fake_labels = torch.zeros(batch_size, 1)
        fake_output = discriminator(fake_data)
        fake_loss = criterion(fake_output, fake_labels)
        fake_loss.backward()
        discriminator_loss = real_loss + fake_loss
        discriminator_optimizer.step()

        # Train generator
        generator.zero_grad()
        fake_labels.fill_(1)
        fake_output = discriminator(fake_data)
        generator_loss = criterion(fake_output, fake_labels)
        generator_loss.backward()
        generator_optimizer.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Batch [{batch_idx}/{len(dataloader)}], "
                f"Discriminator Loss: {discriminator_loss.item():.4f}, "
                f"Generator Loss: {generator_loss.item():.4f}"
            )

# Save trained models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')
