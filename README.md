# README

## Data processing
1. Download data from Kseniia
1. Rename the tables by running *rename_tables.sql*
1. Mark invalid test sections by running *mark_invalid.sql*
1. Get a sample of participants by running *sample.sql*
1. Export log_sample to csv (tab delimiter)
1. Clean the csv of the sample by running *sample_clean.py*
1. Export participants to csv (tab delimiter)
1. Parse the user agent of the participants of the participants by running *parse_user_agent.py*


## Roadmap
### ITE detection
1. Use consistent key inference for all test sections. Currently the ones that are undefined are inferred while the ones that are already defined are not inferred. If we infer the key for everything, we can sidestep some of the awkward key input we currently see (e.g. one letter input is registered as a continuous key).
1. Use two entry designations: "entry" and "action". This allows us to differentiate between the individual key presses leading up to a prediction, while at the same time accounting for the fact that they belong to the same entry. This also allows us to group the multiple actions registered for some gesture entries, because we would group them all as one action.
1. After we do this, we can label each entry based on the video annotations. This allows us to automatically create confusion matrices and to train ML algorithms.
1. Once we have the labeled data, we can train and evaluate ITE classification methods. Because we already developed a manual classification method, we know the key parameters to look at (e.g. IKI, leading spaces, number of characters)
