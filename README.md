"# Natural-Language-Processing" 


Natural Language Processing(NLP):
These files contain an implementation of the baum welch algorithm. The algorithm is used to recognize patterns in sentences such as
grammar by tagging a sequence of words with a sequence of hidden Markova model states. The formal problem is usually referred to as parts
of speech tagging. This algorithm also has many other uses in recognizing time series such as predicting stock values, speech recognition, 
genetics and even body frame recognition. 

The algorithm is trained inside nlp_training.py where it is feed a .dat file containing the brown corpus and a training file with any
English text. The algorithm will recognize the patterns in the training file and use these label words with it's states these states can
then be statistically compared against words labeled with English grammar symbols. The brown_words.dat file contains a corpus that is 
labeled with correct English grammar symbols. However, in practice one might often see the labels left as raw states.
