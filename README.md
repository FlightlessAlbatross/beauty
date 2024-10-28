This repository contains the codes to replicate the "Bewertungsmodell zum Landschaftsbild zur Stromnetzausbau". 
We want to use this to create a beauty or attractiveness score for all of Europe in the context of the GRANULAR project. 

Run the notebooks of the notebook folder in order. For the most part they call scripts in the scripts folder or call functions in the python module. 
The rural_beauty python module is mostly there to contain the data paths in the config file and only some custom functions.

to be able to use the imports go to the folder containing the setup.py file and run this line 
pip install -e .

to install all the needed R libraries you can run the line below. Notice it takes the default path for the libraries.
Rscript requirements_R.R
