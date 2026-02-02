#!/usr/bin/env python
# coding: utf-8

# # Data Splitting

# ## Input: 
# - encounter_notes.csv (EHR data)
# 
# ## Output: 
# - **./train_test_dictionaries** directory (load these for LLM)
#     - Contents: 
#         1. train_patient_data.pkl 

# Load required libraries and modules 
import os 
import pickle
import numpy as np # type: ignore
import pandas as pd  # type: ignore
import input_parse_lipids
from sklearn.model_selection import StratifiedShuffleSplit # type: ignore

# Load and parse raw EHR unstructured data (e.g., notes)
data = input_parse_lipids.read_all_data("../raw_notes/CAD_Diagnoses/")

# Extract MRNs to subset registry data
lipids_llm_keys = pd.Series(list(data.keys()))

# Check numbers
print("Perform Data Check:")
print("\tNumber of MRNs in Queryable Lipids: {}".format(len(lipids_llm_keys)))
print("\t\tNumber of Unique MRNs: {}".format(lipids_llm_keys.nunique()))

# Check if the directory exists, if not, create it
directory = "../train_test_dictionaries_CAD_Diagnoses"
if not os.path.exists(directory):
    os.mkdir(directory)

with open(os.path.join(directory, "CAD_Diagnoses.pkl"), 'wb') as file:
    pickle.dump(data, file)

