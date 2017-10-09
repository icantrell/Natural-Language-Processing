import nlp
import matplotlib.pylab as plb
import numpy as np
import pickle
from functools import reduce

tagged_sents = pickle.load(open('brown_tags.dat','rb'))

h= nlp.hmm()

h.load_from_file('hmm2.dat')

tagged_words = reduce((lambda x,y:x+y),tagged_sents)
tags = list(set([t for w,t in tagged_words]))
tag_hit_freqs = np.zeros(len(tags),dtype=np.float32)
tag_total_freqs = np.zeros(len(tags),dtype=np.float32)

for sent in tagged_sents:
    our_tags = h.viterbi([w for w,t in sent])
    for i,wt in enumerate(sent):
        if wt[1] == our_tags[i]:
            tag_hit_freqs[tags.index(wt[1])] +=1
        tag_total_freqs[tags.index(wt[1])] +=1

output_array = np.zeros(len(tag_total_freqs)+1)
output_array[:-1] = tag_hit_freqs/tag_total_freqs
output_array[-1] = (tag_hit_freqs/tag_total_freqs).mean()

plb.bar(np.arange(len(tags)+1),output_array)
plb.xticks(np.arange(len(tags)+1),tags + ['average'],rotation='vertical')


plb.xlabel('tags (from brown corpus)(https://en.wikipedia.org/wiki/Brown_Corpus)')
plb.ylabel('frequency of tag correctly labeled')


plb.show()
