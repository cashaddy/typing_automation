# Typing Behavior is About More than Speed: Users' Strategies for Choosing Word Suggestions Despite Slower Typing Rates
We analyze a dataset of tens of thousands of mobile users in order to discover and solve issues with mobile keyboards.

Use Python 3.10 to execute the python files.

## Data processing steps

You can download the CSV from our OSF repo. You can take a shortcut and use these files for analysis by putting them 
into the data folder.

If you prefer to start from scratch, follow the instructions below to generate the CSV files.

1. Download the processed data from Palin et al., see https://userinterfaces.aalto.fi/typing37k/ "SQL file with processed 
data as a .zip file (0.8 GB zipped, 5.2 GB unzipped)."
2. Set up a MySQL database locally, e.g. by using the docker-compose file in this repo. Just execute 'docker-compose up' 
from the command shell and connect to your database with an administration tool such as DataGrip. Restore the Database 
you have downloaded before.
3. Rename the tables by running *rename_tables.sql*
4. Mark invalid test sections by running *mark_invalid.sql*
5. Get a sample of participants by running *sample.sql*
6. Export log_sample to csv (tab delimiter)
7. Clean the csv of the sample by running *clean_sample.py*
8. Export participants to csv (tab delimiter)
9. Parse the user agent of the participants of the participants by running *parse_user_agent.py*