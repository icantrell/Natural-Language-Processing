"# Natural-Language-Processing" 


Natural Language Processing(NLP):
These files contain an implementation of the baum welch algorithm. The algorithm is used to recognize patterns in sentences such as
grammar by tagging a sequence of words with a sequence of Hidden Markov Model states. The formal problem is usually referred to as parts of speech tagging. This algorithm also has many other uses in recognizing time series such as predicting stock values, speech recognition, genetics and even body frame recognition. Note: this is an old algorithm and not very effective for NLP so neural nets will be much better choice. This project was mostly done in order to test the effectiveness of this method and to work deeper with machine learning models and mathematics.

# training
The algorithm is trained inside nlp_training.py where it is feed a .dat file containing the brown corpus and a training file with any English text. 
<br>
The algorithm will recognize the patterns in the training file and use these label words with it's states these states can then be statistically compared against words labeled with English grammar symbols. The brown_words.dat file contains a corpus that is  labeled with correct English grammar symbols. However, in practice one might often see the labels left as raw states.

<br><br>
To start the training file just use.<br>
python nlp_training.py <br>
this will train a model on the asop.txt file and save it into hmm2.dat(which should have a pre-trained model).

<br><br>
# running example
To run the main file enter <br>
python nlp_main.py<br>
this will start a loop which will allow you to enter sentences. Note: it will complain if the word is not in the training set and some words might have a statistical value of zero(which is not an error in the algorithm and has a simple yet annoying to implement fix that i haven't bothered with since I will probably never use this source code except for reference.).
