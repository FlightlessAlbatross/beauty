#!/bin/bash

# Loop through each line in parameters.txt
while IFS= read -r params; do
    # Execute the Python script with the parameters
    python3 notebooks/04_training.py $params
done < data/models/model_parameters.txt
