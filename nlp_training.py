import  nlp
import pickle

'''
f = open('asop.txt','r')
corpus = f.read()
f.close()

parsed_corpus = nlp.parse(corpus)

parsed_corpus = pickle.load(open('brown_words.dat','rb'))
'''
a =[['A','B','B','A']]+[['B','A','B']]
h = nlp.hmm()
#h.randomize(set(parsed_corpus),85)

#sentences = nlp.get_sentences(parsed_corpus,1)

h.randomize(['A','B'],3)
print(((h._forward_algorithm(['A','B','B','A']) * h._backward_algorithm(['A','B','B','A']))[0] + (h._forward_algorithm(['B','A','B']) * h._backward_algorithm(['B','A','B']))[0]).sum(0))
print('\n')
h.baum_welch(a)
print('\n')
print(((h._forward_algorithm(['A','B','B','A']) * h._backward_algorithm(['A','B','B','A']))[0]+ (h._forward_algorithm(['B','A','B']) * h._backward_algorithm(['B','A','B']))[0]).sum(0))


#h.map_internal_states(pickle.load(open('brown_tags.dat','rb')))

h.save_to_file('hmm2.dat')
