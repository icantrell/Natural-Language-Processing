import nlp

h = nlp.hmm()
h.load_from_file('hmm2.dat')


while(1):
    h.talk()
    s = nlp.parse(input('sentence: '))
    ts = h.viterbi(s)
    
    print(' '.join([str((p[0],str(p[1]))) for p in zip(s,ts)]))
    
    