# Load required packages
import os
import pandas as pd
import pickle

import lipids_utils 
from llama_index.core import Settings # type: ignore
from llama_index.core.output_parsers import LangchainOutputParser # type: ignore
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # type: ignore
from langchain.prompts import PromptTemplate # type: ignore

def query_main(patient_data_filename: str, prompt_filename: str, 
               confidence_guidelines: str = None, expert_tips: str = None,
               save_embeddings: bool = False, use_gpt4: bool = False, 
               temperature: float = 0, save_responses: bool = True, 
               training_data: bool = True) -> tuple:
    
    # Open pkl file 
    with open(patient_data_filename, 'rb') as file: 
        patient_data_dict = pickle.load(file)
    
    # Load patient data
    patient_data = lipids_utils.load_patient(patient_data_dict)
    
    # Define response schema (uncomment based on which prompt is being used)

    # CAD Diagnoses
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="CAD", description="Whether notes indicate the patient has coronary artery disease (CAD): 'Yes' or 'No'."),
    # ResponseSchema(name="CAD Reasoning", description="A brief explanation of how CAD status was determined, citing specific findings including CAD diagnoses and procedures from the notes.")
    # ]

    # External LDL-C Values
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="LDL-C Value", description="The most recent LDL-C value mentioned in the notes. If no LDL-C value is found, record 'NA'."),
    # ResponseSchema(name="LDL-C Date", description="The date of the most recent LDL-C value mentioned in the notes, in MM/DD/YYYY format. For month/year only, use MM/01/YYYY. For relative dates, estimate based on the note date. If no date is provided, record 'NA'.")
    # ]


    # LLT Utilization
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="Statin Use", description="Whether notes indicate the patient was taking a statin during the most recent encounter where a statin was mentioned: 'Yes' or 'No'. If a statin is never mentioned, record 'No'."),
    # ResponseSchema(name="High Intensity Statin Use", description="Whether notes indicate the patient was taking a high-intensity statin (atorvastatin ≥40 mg daily or rosuvastatin ≥20 mg daily) during the most recent encounter where a high-intensity statin was mentioned: 'Yes' or 'No'. If the patient is not taking a statin or if high-intensity statin use is never mentioned, record 'No'."),
    # ResponseSchema(name="Ezetimibe Use", description="Whether notes indicate the patient was taking ezetimibe during the most recent encounter where ezetimibe was mentioned: 'Yes' or 'No'. If ezetimibe is never mentioned, record 'No'."),
    # ResponseSchema(name="Bempedoic Acid Use", description="Whether notes indicate the patient was taking bempedoic acid during the most recent encounter where bempedoic acid was mentioned: 'Yes' or 'No'. If bempedoic acid is never mentioned, record 'No'."),
    # ResponseSchema(name="PCSK9 Inhibitor Use", description="Whether notes indicate the patient was taking a PCSK9 inhibitor (inclisiran, evolocumab, or alirocumab) during the most recent encounter where a PCSK9 inhibitor was mentioned: 'Yes' or 'No'. If a PCSK9 inhibitor is never mentioned, record 'No'."),
    # ResponseSchema(name="LLT Use Reasoning", description="A brief explanation of how statin, high-intensity statin, ezetimibe, bempedoic acid, and PCSK9 inhibitor utilization were determined, citing specific findings from the notes.")
    # ]


    # Statin Intolerance
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="Statin Intolerance", description="Whether notes indicate the patient has statin intolerance: 'Yes' or 'No'."),
    # ResponseSchema(name="Statin Intolerance Reasoning", description="A brief explanation of how statin intolerance status was determined, citing specific findings from the notes.")
    # ]

    # Trial of ≥2 Statins
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="Two Statins Tried", description="Whether notes indicate the patient has taken or tried two or more distinct statin medications, including current and prior use: 'Yes' or 'No'. Record 'Yes' if notes state 'multiple' or 'several' statins even if specific names are not provided."),
    # ResponseSchema(name="Statins Tried Reasoning", description="A brief explanation of how current or prior use of at least two distinct statins was determined, citing specific findings from the notes and the names of specific statins if available.")
    # ]

    # Reasons for Statin Nonuse
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="Statin Nonuse Reason", description="The first applicable reason for statin nonuse, chosen from: 'Intolerance', 'Clinician decision', 'Patient preference', or 'Not mentioned'."),
    # ResponseSchema(name="Statin Nonuse Reasoning", description="A brief explanation of how the reason for statin nonuse was determined, citing specific findings from the notes.")
    # ]

    # Reasons for High-Intensity Statin Nonuse
    # response_schema = [
    # ResponseSchema(name="MRN", description="The medical record number (MRN) of the patient being reviewed. The MRN can be found in the note metadata."),
    # ResponseSchema(name="High-Intensity Statin Nonuse Reason", description="The first applicable reason for high-intensity statin nonuse, chosen from: 'Intolerance', 'Clinician decision', 'Patient preference', or 'Not mentioned'."),
    # ResponseSchema(name="High-Intensity Statin Nonuse Reasoning", description="A brief explanation of how the reason for high-intensity statin nonuse was determined, citing specific findings from the notes.")
    # ]

    # Define output parser 
    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schema) 
    output_parser = LangchainOutputParser(lc_output_parser) 
    
    # Generate and assign settings
    settings = lipids_utils.generate_settings_basic(use_gpt4=use_gpt4, temperature=temperature, output_parser=output_parser)
    Settings.prompt_helper = settings['prompt_helper']
    Settings.embed_model = settings['embed_model']
    Settings.node_parser = settings['node_parser']
    Settings.llm = settings['llm']

    # Parse patient data into node representation
    doc_nodes = lipids_utils.parse_patient_data_nodes(patient_data, settings['node_parser'])
    print(patient_data)

    # Read in prompt 
    prompt_template = lipids_utils.read_text(prompt_filename)

    # Set up prompt template 
    prompt_template = PromptTemplate(
        template=prompt_template,  # Prompt template from `read_text`
        input_variables=[]
    )

    # Format prompt
    prompt_text = prompt_template.format()
    
    # Establish response synthesizer 
    response_synthesizer = lipids_utils.create_response_synth(settings)
    
    # Initialize lists and dictionaries to store results
    structured_responses = []
    problem_keys = []
    source_nodes = {}
    
    # Create patient embeddings and query
    for key in doc_nodes.keys():
        try:
            index = lipids_utils.create_vector_index(doc_nodes=doc_nodes[key], settings=settings, save_embedding=save_embeddings)  # Index current MRN
            
            query_engine = lipids_utils.create_query_engine(index=index, settings=settings, response_synthesizer=response_synthesizer, k=5)  # Query engine on index
            
            response = query_engine.query(prompt_text)  # Prompt query engine and store formatted response
            
            response_dict = lipids_utils.response_to_dict(response.response)  # Convert formatted response to dictionary object
            
            structured_responses.append(response_dict)  # Append to list
            
            source_nodes[key] = response.source_nodes  # Track source nodes
            
        except Exception as e:
            print(f"Error processing key {key}: {e}")
            problem_keys.append(key)
            continue  # Continue processing other keys even if one fails
              
    if save_responses:
    
        # Convert structured responses to a DataFrame
        df = pd.DataFrame(structured_responses)

        data_type = "training" if training_data else "test"
        prompt_version = os.path.splitext(os.path.basename(patient_data_filename))[0]

        # Define the directory path
        dir_path = f'results_{data_type}_{prompt_version}'

        # Check if the directory exists, create it if it doesn't
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # Save the DataFrame to a CSV file 
        file_path = f'{dir_path}/structured_responses.csv'
        df.to_csv(file_path, index=False)
        
        # Define the sources subdirectory path
        sources_path = os.path.join(dir_path, "sources")

        # Check if the sources subdirectory exists, create it if it doesn't
        if not os.path.exists(sources_path):
            os.makedirs(sources_path)

        # Iterate through each key's list of objects and write 'text' and 'metadata' attributes to a file
        for mrn, obj_list in source_nodes.items():
            with open(os.path.join(sources_path, f"{mrn}.txt"), 'w') as file:
                for obj in obj_list:
                    file.write(f"Metadata:\n\t{str(obj.metadata)}\n\n")
                    file.write(f"Text:\n\t{str(obj.text)}\n\n")
                    file.write("\n\n")

        print(f"DataFrame saved to {file_path}")
        print(f"Source nodes saved to {sources_path}")
    
    # Return structured responses and the main directory path
    return structured_responses, dir_path