# Required Libraries
import os
import pandas as pd

# Function to process encounter notes
def process_notes(path):
    """
    Purpose:
    Process unstructured EHR encounter notes from a pandas DataFrame, aggregate by MRN and Note ID,
    sort and compile note lines, and remove unprocessed text data.

    Parameters:
    path: The path to a file containing EHR notes. Supported formats are .csv, .tsv, and .xlsx. Expected columns:
        - 'mrn': Medical Record Number
        - 'note_id': Note identifier
        - 'line': Line number of the note
        - 'note_text': Text of the note
        - 'contact_date': Date of the note
        - 'no_statin': Patient on statin (1 = yes)
        - 'no_high_intensity': Patient on high intensity statin (1 = yes)
        - ldl_ge_70: Patient LDL greater than or equal to 70

    Returns:
    dict: A nested dictionary structured as follows:
        {
            MRN: {
                NoteID: {
                    'mrn': str,        # Patient MRN for querying
                    'note_type': str,  # Type of note
                    'note': str,       # Compiled and cleaned text of the note
                    'date': str,       # Date of the note
                    'statin': int     # Patient on Statin
                    'high_intensity_statin': int    # Patient on high intensity statin 
                    'elevated_ldl': int    # Patient LDL >= 70 
                }
            }
        }
    """
    
    def read_file_with_encodings(file_path, delimiter=','):
        encodings = [None, 'utf-8', 'latin1']
        for encoding in encodings:
            try:
                return pd.read_csv(file_path, dtype=str, parse_dates=True, encoding=encoding, delimiter=delimiter)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("The file encoding is not UTF-8 or Latin-1. Please ensure the file is properly encoded or specify the correct encoding.")
    
    # Determine file extension and read the file into a DataFrame
    if path.endswith('.csv'):
        df = read_file_with_encodings(path)
    elif path.endswith('.tsv'):
        df = read_file_with_encodings(path, delimiter='\t')
    elif path.endswith('.xlsx'):
        df = pd.read_excel(path, dtype=str, parse_dates=True)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv, .tsv, or .xlsx file.")
    
    # Ensure required columns are present
    required_columns = ['mrn', 'note_id', 'line', 'note_text', 'contact_date']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Initialize empty dictionary to store processed data 
    data = {} 
    
    # Group DataFrame by MRN and NOTE_ID 
    grouped = df.groupby(['mrn', 'note_text']) 
    
    # Iterate through each (mrn, note_id) group to extract features and process text 
    for (mrn, nid), group in grouped: 
        if mrn not in data:  # Check if MRN is already a key in data
            data[mrn] = {}  # If not, initialize nested dictionary for MRN 
            
        if nid not in data[mrn]:  # Check if Note ID is a key within the MRNs dictionary 
            # If not, initialize a nested dictionary for Note ID containing features of interest 
            data[mrn][nid] = {
                'text': {},  # Initialize dictionary to store raw note text
                'mrn': str(mrn),  # Store MRN
                'date': str(group.iloc[0]['contact_date']),  # Store date of note 
                # 'bucket': str(group.iloc[0]['bucket']), # Store statin use 
                # 'lipids_checked': str(group.iloc[0]['lipids_checked']), # Store high intensity usage 
            }
            
        # Iterate through each row in a group 
        for _, row in group.iterrows(): 
            data[mrn][nid]['text'][int(row['line'])] = ' '.join(str(row['note_text']).split())  # Store note text by line number
            
    # Sort and compile notes 
    for mrn, notes in data.items():  # Iterate over key-value pairs for dictionary nested by MRN
        for nid, details in notes.items():  # Iterate over key-value pairs in sub-dictionary nested by Note ID
            details['text'] = dict(sorted(details['text'].items()))  # Ensure lines are sorted within a note
            note = "\n".join([' '.join(details['text'][line].split()) for line in details['text']])  # Concatenate lines
            details['note'] = note  # Store concatenated note
            del details['text']  # Delete uncompiled text 
            
    return data  # Return compiled data


def read_all_data(data_dir):
    """
    Purpose:
    Reads and processes all EHR data files from the specified directory and combines them into a master dictionary.

    Parameters:
    data_dir (str): The directory path where the data files are located.

    Returns:
    dict: A master dictionary containing all processed EHR data aggregated by MRN and Note/Procedure IDs.
    """
    
    master_data = {}  # Initialize an empty dictionary to store the combined data

    # List of expected file names and corresponding processing functions
    file_processors = {
        'encounter_notes': process_notes
    }

    # Supported file extensions
    extensions = ['.csv', '.tsv', '.xlsx']

    for file_prefix, processor in file_processors.items():
        file_found = False
        for ext in extensions:
            file_path = os.path.join(data_dir, f'{file_prefix}{ext}')
            if os.path.exists(file_path):
                file_found = True
                # Process the data using the appropriate function
                processed_data = processor(file_path)

                # Combine the processed data into the master dictionary
                for key in processed_data.keys():
                    if key not in master_data:
                        master_data[key] = {}
                    for nkey in processed_data[key].keys():
                        master_data[key][nkey] = processed_data[key][nkey]
                        master_data[key][nkey]['type'] = file_prefix
                break  # Exit the loop once the file is found and processed
        if not file_found:
            print(f'{file_prefix} file not found in the directory.')

    return master_data  # Return the combined master dictionary
