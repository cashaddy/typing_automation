# README

## Data processing
1. Download data from Palin et al.
1. Rename the tables by running *rename_tables.sql*
1. Mark invalid test sections by running *mark_invalid.sql*
1. Get a sample of participants by running *sample.sql*
1. Export log_sample to csv (tab delimiter)
1. Clean the csv of the sample by running *clean_sample.py*
1. Export participants to csv (tab delimiter)
1. Parse the user agent of the participants of the participants by running *parse_user_agent.py*


## Future improvements
1. Use a more complete word frequency list (or derive one manually from a word corpus). Currently, some words (e.g. "Azerbaijan") are so uncommon that they are not listed in our word frequency list. We currently assign these words an artificially low frequency of 1, but this is a completely arbitrary value. A more complete word frequency list would allow us to characterize the frequency of more words.
1. Incorrect levenshtein distance calculation.
1. Normalize the leadup and base speeds in the selection model. Since we know that suggestion users naturally type slower, we should normalize the typing speed according to the user's natural speed.
1. Handle multi-word suggestions. This usually occurs for common word combinations (e.g. user types "I am going", and selects "to the" from the suggestion list). We do not currently handle these cases explicitly when classifying ITE's, and therefore the behaviour is undefined.
1. Decide how to handle multiple suggestions for one word. It rarely happens, but sometimes users will use the suggestion list more than once for the same word. Sometimes it occurs consecutively, while on other occasions the user inserts characters in between using the suggestion list. Currently we only register only the last suggestion, but other methods include registering only the first suggestion, or completely removing these cases from the analysis. 
1. Localize middle-of-string inputs. Currently, we can only handle user inputs that correspond to the end of the existing string. Users inserting letters, spaces, and backspaces in any location other than the end of the string (i.e. by manually moving their cursor) are marked as undefined because we do not know where in the strong they were inserted. If we are able to pinpoint where in the string these inputs occur (e.g. during the edit distance calculation), then we can include these keystrokes in our analysis.


## Limitations of the dataset
* Given that this is a transcription task, we miss out on some strategies of suggestion usage, such as using the suggestion list as a "dictionary" in order to spell a word that the user does not know how to spell. Since the sentences are transcribed, the user can always consult the template sentence in order to confirm the spelling, thereby eliminating this strategy.
* We only focus on end-of-sentence keystrokes. That is, we only consider keystrokes that modify the end of a sentence. This includes typing letters, backspacing, and predicting words. However, changes that were made in the middle of the sentece (i.e. by moving the cursor manually) are removed from the analysis.
* It is difficult to get statistically significant suggestion usage patterns for individual participants, since suggestion is not used often enough for there to be large enough sample size. For example, if a user uses suggestions 5 times, and all of those occur on words of length 5 or less, does that mean that they prefer to use it on shorter words? Or is it just too small of a sample?
