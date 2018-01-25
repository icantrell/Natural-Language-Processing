import  nlp
import pickle


f = open('asop.txt','r')
corpus = f.read()
f.close()

parsed_corpus = nlp.parse(corpus)

#parsed_corpus = pickle.load(open('brown_words.dat','rb'))

h = nlp.hmm()
h.randomize(set(parsed_corpus),13)

sentences = nlp.get_sentences(parsed_corpus,1)

h.baum_welch(list(filter(lambda x: len(x) < 17,sentences)))


#h.map_internal_states(pickle.load(open('brown_tags.dat','rb')))

h.save_to_file('hmm2.dat')
