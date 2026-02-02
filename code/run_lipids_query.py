#!/usr/bin/env python
# coding: utf-8

# Load required libraries 
import lipids_query as query


# Run query
responses, dir_path = query.query_main(
    patient_data_filename="../train_test_dictionaries_CAD_Diagnoses/CAD_Diagnoses.pkl", 
    prompt_filename="../prompts/prompt_1_CAD_Diagnoses.txt",
    save_embeddings=False, 
    use_gpt4=True,
    temperature=0, 
    save_responses=True, 
    training_data=False
)
