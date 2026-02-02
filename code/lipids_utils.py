# Load required packages
import datetime
import configparser
import ast
from llama_index.core import Document # type: ignore
from llama_index.core import Settings, PromptHelper, VectorStoreIndex # type: ignore
from llama_index.core.node_parser import SentenceSplitter # type: ignore
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding # type: ignore
from llama_index.llms.azure_openai import AzureOpenAI # type: ignore
from llama_index.core.output_parsers import LangchainOutputParser # type: ignore
from langchain.output_parsers import StructuredOutputParser, ResponseSchema # type: ignore
from llama_index.core.query_engine import BaseQueryEngine # type: ignore
from llama_index.core import get_response_synthesizer  # type: ignore
from llama_index.core.response_synthesizers import CompactAndRefine # type: ignore

# Function to ingest patient data into Document objects 
# Stored in a dictionary with MRN and note/narrative keys 
# Function to ingest patient data into Document objects 
# Stored in dictionary with MRN and note/narrative keys 
def load_patient(pat_data: dict) -> dict:
    """  
    Purpose:
    Load patient data into a dictionary of lists of Document objects for LLM processing.

    Parameters:
    pat_data (dict): The patient data dictionary containing notes and narratives.

    Returns:
    dict: A dictionary with MRN as keys and lists of Document objects categorized by type (e.g., notes and narratives) as values.
    """

    # Initialize dictionary to store Document objects for each MRN
    docs = {}

    # Iterate over each MRN in the patient data
    for mrn in pat_data.keys():
        # Initialize lists to store Document objects for different types of notes
        encounter_docs = []
        latest_date = None  # Variable to track the latest date of service

        # Iterate over each note/procedure ID for the MRN
        for nkey in pat_data[mrn].keys():
            try:
                # Attempt to parse the date of service from the note/narrative
                record_date = datetime.datetime.strptime(pat_data[mrn][nkey]['date'], "%Y-%m-%d %H:%M:%S")
            except:
                # If parsing fails, use the current date
                record_date = datetime.datetime.today()
            
            # Update the latest date if the current record's date is later
            if (latest_date is None) or (record_date > latest_date):
                latest_date = record_date

            # Check if 'type' key exists in the current record
            if 'type' in pat_data[mrn][nkey]:
                note_type = pat_data[mrn][nkey]['type']
                
                # Determine the type and create the appropriate Document object with metadata
                if note_type == 'encounter_notes':
                    doc = Document(
                        text=pat_data[mrn][nkey]['note'],
                        metadata={
                            'mrn': pat_data[mrn][nkey]['mrn'],
                            'date': pat_data[mrn][nkey]['date']
                        }
                    )
                    encounter_docs.append(doc)  # Append to encounter documents list
            else:
                print(f"Type not found for MRN: {mrn}, Note ID: {nkey}, skipping entry.")

        # Combine the Document objects into a dictionary for each MRN
        docs[mrn] = {
            'encounter': encounter_docs
        }

    return docs  # Return the dictionary of Document objects

# Helper function to read config file containing API keys, endpoints, etc.
# This is useful to keep API keys from being displayed if using version control like Git
def generate_config(config_filename):
    """   
    Purpose:
    Read and parse a configuration file to retrieve settings for a specified section.

    Parameters:
    config_filename (str): The path to the configuration file.
    
    Returns:
    configparser.ConfigParser: A ConfigParser object containing the parsed configuration settings.
    """
    
    config = configparser.ConfigParser() # create a ConfigParser object to read and parse the config file
    try:
        config.read([config_filename]) # read the specified configuration file
    except Exception as e: 
        raise ValueError(f"Error reading the configuration file: {e}")
    
    return config # return the settings for the config file


# ServiceContext is deprecated, so we need to use Settings
# Function to generate Azure embedding and LLM parameters
def azure_settings(use_gpt4: bool = False) -> tuple:
    """
    
    Purpose:
    Generate settings for Azure embedding and LLM based on the specified GPT version.

    Parameters:
    use_gpt4 (bool, optional): Flag to indicate whether to use GPT-4 settings. Default is False.

    Returns:
    tuple: A tuple containing dictionaries for Azure embedding settings and Azure LLM settings.
    """
    
    # Select the appropriate embedding and LLM model configuration based on GPT version
    config_file = '../config_files/azure_config_4.cfg' if use_gpt4 else '../config_files/azure_config.cfg'
    
    # Read the configuration file
    try:
        config = generate_config(config_file)
    except Exception as e:
        raise ValueError(f"Error generating configuration: {e}")
    
    # Read embedding and LLM sections
    try:
        azure_embed_settings = config['embedding']
        azure_llm_settings = config['llm']
    except KeyError as e:
        raise KeyError(f"Missing required section in the configuration file: {e}")
        
    # Return settings
    return azure_embed_settings, azure_llm_settings

# Code to generate base model settings
def generate_settings_basic(prompt_helper_dict: dict = None, 
                            use_gpt4: bool = False, 
                            temperature: float = 0.0, 
                            output_parser: LangchainOutputParser = None) -> dict:
    """
    
    Purpose: 
    Generate a basic settings configuration for LLM operations, configure embedding models, 
    node parsers, prompt helpers, and language models. This communicates with our Azure OpenAI account. 
    
    Parameters: 
    prompt_helper_dict (dict, optional): Dictionary containing prompt helper settings.
    use_gpt4 (bool, optional): Flag to indicate whether to use GPT-4 settings. Default is False. 
    temperature (float, optional): The temperature setting for the language model. Default is 0.0. 
    output_parser (LangchainOutputParser, optional): How to structure query response. Default is None, but one is recommended.
    
    Returns: 
    dict: A dictionary containing configured settings for LLM operations within Azure OpenAI.
    """

    # Select the appropriate embedding model configuration based on GPT version
    azure_embed_settings, azure_llm_settings = azure_settings(use_gpt4)

    # Set prompt helper settings if none are provided 
    if prompt_helper_dict is None: 
        prompt_helper_dict = { 
            'chunk_overlap_ratio': 0.1,  # Percentage of token amount that each chunk should overlap
            'num_output': 1024,  # Amount of token space to leave in input for generation
            'context_window': 16000,  # Maximum context size that will get sent to the LLM
            'chunk_size': 1024,  # Maximum size of each chunk
            'chunk_overlap': 100,  # Overlap size between chunks
            'separator': ' '  # Separator when chunking tokens
        }
        
    # Initialize the PromptHelper with the provided or default settings
    prompt_helper = PromptHelper(
        context_window=prompt_helper_dict['context_window'],
        chunk_overlap_ratio=prompt_helper_dict['chunk_overlap_ratio'],
        num_output=prompt_helper_dict['num_output']
    )
    
    # Initialize embedding model with settings from the configuration file
    embed_model = AzureOpenAIEmbedding(
        mode='similarity',  # Mode for embedding
        deployment_name=azure_embed_settings.get('deployment_id_main'),  # Deployment name for embeddings
        model=azure_embed_settings.get('deployment_id_embed'),  # Embedding model
        api_key=azure_embed_settings.get('api_key'),  # API key for authentication
        azure_endpoint=azure_embed_settings.get('azure_endpoint'),  # Azure endpoint URL
        api_type='azure',  # API type (azure for Azure OpenAI)
        api_version=azure_embed_settings.get('api_version'),  # API version
        embed_batch_size=azure_embed_settings.get('embed_batch_size')  # Batch size for embedding
    )
    
    # Initialize node parser
    node_parser = SentenceSplitter(
        chunk_size=prompt_helper_dict['chunk_size'], 
        chunk_overlap=prompt_helper_dict['chunk_overlap']
    )  # Create SentenceSplitter for node parsing
    
    # Initialize LLM with settings from the configuration file
    openai_llm = AzureOpenAI(
        deployment_name=azure_llm_settings.get('deployment_id_main'),  # Deployment name for main LLM
        engine=azure_llm_settings.get('deployment_id_main'),  # Specify engine
        model=azure_llm_settings.get('deployment_id_llm'),  # Specify model
        api_key=azure_llm_settings.get('api_key'),  # API key for authentication
        azure_endpoint=azure_llm_settings.get('azure_endpoint'),  # Azure endpoint URL
        api_type='azure',  # API type (azure for Azure OpenAI)
        api_version=azure_llm_settings.get('api_version'),  # API version
        output_parser=output_parser,  # Structure of LLM query output
        temperature=temperature  # Temperature setting for the LLM
    )
    
    # Create and return settings
    settings = {
        'prompt_helper': prompt_helper,
        'embed_model': embed_model,
        'node_parser': node_parser,
        'llm': openai_llm
    }
    
    return settings

# Parses ALL unstructured data corresponding to each MRN into nodes
def parse_patient_data_nodes(patient_data: dict, node_parser) -> dict:
    """       
    Purpose:
    Parse patient data to create document nodes for encounters and narratives, organizing by MRN.

    Parameters:
    patient_data (dict): The patient data dictionary with MRNs as keys and nested dictionaries for encounter and narrative documents.
    node_parser: The node parser to convert documents into nodes.

    Returns:
    dict: A dictionary containing MRNs as keys, with nested dictionaries for 'encounter' and 'narrative' nodes.
    """
     
    # Initialize dictionary to store nodes, organizing by MRN
    doc_nodes = {}

    # Iterate through each MRN in the patient data
    for mrn, docs in patient_data.items():
        # Initialize nested dictionary for the current MRN
        doc_nodes[mrn] = {
            'encounter': []
            }

        # Check if 'encounter' documents exist for the current MRN
        if 'encounter' in docs:
            try:
                # Parse encounter documents into nodes and store under the current MRN
                encounter_nodes = node_parser.get_nodes_from_documents(docs['encounter'], show_progress=False)
                doc_nodes[mrn]['encounter'].extend(encounter_nodes)
            except Exception as e:
                print(f"Error parsing encounter documents for MRN {mrn}: {e}")

        # Check if 'narrative' documents exist for the current MRN
        if 'narrative' in docs:
            try:
                # Parse narrative documents into nodes and store under the current MRN
                narrative_nodes = node_parser.get_nodes_from_documents(docs['narrative'], show_progress=False)
                doc_nodes[mrn]['narrative'].extend(narrative_nodes)
            except Exception as e:
                print(f"Error parsing narrative documents for MRN {mrn}: {e}")

    return doc_nodes

# Create vector store index
def create_vector_index(doc_nodes: dict, settings: dict, save_embedding: bool = False) -> VectorStoreIndex:
    """
    
    Purpose:
    Create a vector store index from document nodes, with an option to store embeddings.

    Parameters:
    doc_nodes (dict): A dictionary containing 'encounter' and/or 'narrative' nodes.
    settings (dict): A dictionary of settings for RAG framework. Likely generated using generate_settings_basic().
    save_embedding (bool): Flag to determine whether to save embeddings. Default is False.

    Returns:
    VectorStoreIndex: The created vector store index for the specific patient.
    """
    
    # Extract the MRN from the metadata of the first node available
    if 'encounter' in doc_nodes and doc_nodes['encounter']:
        mrn = doc_nodes['encounter'][0].metadata['mrn']
    else:
        raise ValueError("No encounter or narrative nodes available to extract MRN.")
    
    # Flatten dictionary into a list of all node objects
    all_nodes = []
    if 'encounter' in doc_nodes:
        all_nodes.extend(doc_nodes['encounter'])


    # Create vector store index without saving embeddings
    vector_index = VectorStoreIndex(
        nodes=all_nodes, 
        embed_model=settings['embed_model'], 
        prompt_helper=settings['prompt_helper'], 
        show_progress=False
    )

    if save_embedding:
        # Get the current date in yyyymmdd format for the path
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # Check if the 'stored_embeddings' directory exists, create it if it doesn't
        if not os.path.exists('stored_embeddings'):
            os.mkdir('stored_embeddings')
        
        # Persist the storage context to save embeddings with MRN in the directory path
        vector_index.storage_context.persist(persist_dir=f'./stored_embeddings/{mrn}_embeddings_{current_date}')
        
    # Return the vector store index
    return vector_index

# Function to create Response Synthesis
def create_response_synth(settings: dict, response_mode: str = 'compact'):
    """   
    Purpose:
    Create a response synthesizer for generating structured answers from the language model. This function utilizes the 
    settings to configure the response synthesizer with specified options.

    Parameters:
    settings (dict): The settings containing configurations for the language model service.

    Returns:
    ResponseSynthesizer: The configured response synthesizer for generating structured responses.
    """
    
    # Initialize the response synthesizer with the given options.
    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode, # Specify response mode
        structured_answer_filtering=False,  # Do not filter answers to be structured
        prompt_helper=settings['prompt_helper'],  # Use the prompt helper from settings
        use_async=True,  # Use asynchronous processing
        verbose=False  # Disable verbose logging
    )

    return response_synthesizer  # Return the configured response synthesizer

# Function to generate Query Engine
def create_query_engine(index: VectorStoreIndex, settings: dict, 
                        response_synthesizer=CompactAndRefine, k: int = 5) -> BaseQueryEngine:
    """
    Purpose:
    Create a query engine for querying the vector store index with structured responses. This function configures
    the query engine with the specified response synthesizer and optional filters.

    Parameters:
    index (VectorStoreIndex): The vector store index to query.
    settings (dict): The settings containing configurations for the language model service, including the response synthesizer.
    filters (MetadataFilters, optional): Filters to apply for the query. Default is None.

    Returns:
    BaseQueryEngine: The configured query engine for querying the vector store index.
    """
    
    # Create the response synthesizer using the provided settings
    #response_synthesizer = create_response_synth(settings)
    
    # Initialize the query engine with the specified settings
    query_engine = index.as_query_engine(
        similarity_top_k=5, # Number of top-k similar documents
        response_synthesizer=response_synthesizer,  # Use the configured response synthesizer
        llm=settings['llm'], 
    )

    # Return the configured query engine
    return query_engine

# Helper functions for reading prompt txt files
def open_txt_file(path):
    """
    Purpose:
    Open a text file and read its contents.

    Parameters:
    path (str): The path to the text file to be read.

    Returns:
    str: The contents of the text file as a string.
    """
    
    with open(path, "r") as f:
        # Read and return the contents of the file
        return f.read()

def read_text(prompt_filename: str):
    """
    Purpose:
    Read and return the contents of a text file containing a prompt template.

    Parameters:
    prompt_filename (str): The path to the text file containing the prompt template.

    Returns:
    str: The contents of the prompt template file as a string.
    """
    
    # Read the contents of the prompt template file
    template = open_txt_file(prompt_filename)
    
    # Return the contents of the template
    return template

# Helper function for getting response object into a dictionary
def response_to_dict(response_str):
    """    
    Purpose:
    Convert a response string to a dictionary.
    
    Parameters:
    response_str (str): The response string to convert.
    NOTE: must be response.response to get proper string object
    
    Returns:
    dict: The converted dictionary.
    """
    return ast.literal_eval(response_str)
