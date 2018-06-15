import numpy as np
from copy import deepcopy
import re
import pickle
from functools import reduce
from scipy import random

_PARSE_REGEX = 'Mrs\.|Mr\.|Ms\.|\.|;|:|\?|!|\'|\w+|\d+|,|"'


def parse(corpus):
    '''Parse the document into its words.'''
    return [w.lower() for w in re.findall(_PARSE_REGEX,corpus)]    

def get_sentences(corpus,n=1):
    '''Split parsed text into groups of n sentences.'''
    #corpus = parse(corpus)
    paragraph = []
    sentences = []
    sentence = []
    regex = re.compile('\.|!|\?')
    i=0
    for word in corpus:
        sentence.append(word)
        if regex.match(word):
            sentences+=sentence.copy()
            sentence = []
            i+=1
            if i%n==0:
                paragraph.append(sentences.copy())
                sentences= []
            
    return paragraph

class hmm:
    '''class hmm
        This class is for a hidden markov model.

        --------------------------------------
        DOC
        --------------------------------------

        numpy.array
        

        2x2 numpy.dict
        emisson matrix
                    columns are states
                    0,1,2,3 ... COLUMNS ARE numpy.array int16
                    ---------
        a0='word0' |
       a1 ='word1' |
                   |
               ... |
                   |
     ROWS ARE DICT ENTRIES 

        2x2 numpy.array
        transition matrix
                    Columns are the outgoing states.
        ,  ....      0 1 2 3 ... COLUMNS ARE numpy.array dtype=int16            
                    -------------
                0  |
                1  |
                2  |
              ...  |
                
     ROWS ARE numpy.array dtype=int16
     Rows are the incoming states.
        
    '''
    def __init__(self):
        '''
        #Test parameters from paper.
        self.transition_matrix = np.array([[.3,.7],[.1,.9]])
        self.initial_matrix = np.array([.85,.15])
        self.emission_matrix = {'A':np.array([.4,.5]),'B':np.array([.6,.5])}
        self.number_of_states = 2
        '''
        
        #initialize empty containers
        self.transition_matrix = np.empty([0])
        self.initial_matrix = np.empty([0])
        self.emission_matrix = {}
        self.number_of_states = 0
        self.state_tag_map = np.array([])

    
    def randomize(self,observations,num_states):
        '''
        initialize an hmm with n hidden states and m observations for each state
        and andomize all transitions.
        '''
        
        self.number_of_states = num_states
        #The probabilites for each state's outoging transitions sum to 1.
        self.transition_matrix = np.matrix(random.dirichlet(np.ones((num_states)),size=num_states))
        self.transition_matrix.dtype = np.float64
        
        #The initial probabilities of starting with a state all sum to 1.
        self.initial_matrix = np.matrix(random.dirichlet(np.ones((num_states))))
        self.initial_matrix.dtype = np.float64
        
        
        #The emission probabilities for each state sum to 1.
        emission_probs = np.matrix(random.dirichlet(np.ones(len(observations)), num_states))
        
        for o in observations:
            self.emission_matrix[o] = np.matrix(np.zeros(num_states))
            self.emission_matrix[o].dtype = np.float64
            
        #set values of emission matrix   
        for i,o in enumerate(observations):
            for x in range(num_states):
                self.emission_matrix[o][0,x] =  emission_probs[x,i]
                
                
    
    def _forward_algorithm(self,arr):
        #test if prefix of the array was already computed?
        forward_array = np.matrix(np.zeros([len(arr),len(self.transition_matrix)],dtype=np.float64))

        #set intial probablities
        #for each state, x, multiply the the probability of the first word being emitted times the probability of x being the initial state.
        t0 = self.emission_matrix[arr[0]]
        t1 = self.initial_matrix
        forward_array[0]=np.multiply(t0, t1)
                                 
        #starting at the second word in the array
        for index,entry in enumerate(arr[1:]):                     
            #for each state, x, take the summation of the fi*ex*tix i=[0:n] where fi is the ith value computed last iteration
            #for the ith state and ex is probability of the word("entry") is emitted for the xth state and tix is the incoming
            #transition from the ith state to the xth.
            #numpy's will broadcast for each of the states.
            forward_array[index + 1] = np.multiply(self.emission_matrix[entry].T, self.transition_matrix.T * forward_array[index].T).T
            #forward_array[index+1] = (forward_array[index] * self.emission_matrix[entry].reshape((self.number_of_states,1)) * self.transition_matrix.T).sum(1)
        
        #return the matrix
        return forward_array
            
               
            
                                     
    def _backward_algorithm(self,arr):
        #reverse the array
        arr = arr[::-1]
                                 
        #set the initial probabilities to 1. For the transitions to the final states.
        backward_array = np.matrix(np.full([len(arr), len(self.transition_matrix)],1,dtype=np.float64))

        for index,entry in enumerate(arr[:-1]):
            #for each state, x, take the summation of the fi*ei*tix i=[0:n] where fi is the ith value computed last iteration
            #for the ith state and ei is probability of the word("entry") is emitted for the ith state and tix is the outgoing
            #transition from the xth state to the ith. This part is done by going backward through the directed graph.
            #numpy's will broadcast for each of the states.
            backward_array[index+1] = (self.transition_matrix* np.multiply(self.emission_matrix[entry].T , backward_array[index].T)).T

        #return the matrix   
        return backward_array[::-1]

    def viterbi(self,arr,testing=False):
        #USE ADDITIVE LOG.
        #set aside space for a transition matrix to hold the values for the
        transition_array = np.empty((len(arr),len(self.transition_matrix)),dtype=np.float64)
        backtrack_array = np.empty((len(arr) - 1,len(self.transition_matrix)),dtype=np.float64)
        
        #set the initial transiton matrix
        if arr[0] in self.emission_matrix:
            transition_array[0] = np.array(self.initial_matrix) * np.array(self.emission_matrix[arr[0]])
        else:
            print('"' + arr[0] +'" is not recognized.')
            return []          
        
        #for each element in the input find and record the last most likely state.
        for index,entry in list(enumerate(arr))[1:]:
            if arr[index] in self.emission_matrix:
                for n_index in range(len((self.transition_matrix))):
                    transition_array[index-1] /= np.sum(transition_array[index-1])
                    #get the incoming values by multiplying by their transition probablities.
                    incoming_values = transition_array[index-1]*np.array(self.transition_matrix).T[n_index]
                    #find the index of the most likely.
                    l_index = np.argmax(incoming_values)
                    #get the probability this state output the current element.
                    transition_array[index][n_index] = np.array(self.emission_matrix[arr[index]])[0][n_index] * incoming_values[l_index]
                    #record last value.
                    backtrack_array[index-1][n_index] = l_index
            else:
                print('"' + arr[index] +'" is not recognized.')
                return []
        
        #set aside space to build the output sequence.
        output_sequence = np.empty((len(arr)),dtype=np.int64)   
        #find the most likely final state.
        output_sequence[0] = np.argmax(transition_array[-1])
        print(transition_array)
        for i,x in zip(range(1,len(output_sequence)),range(len(backtrack_array))[::-1]):
            #go backwards down the backtrack getting each value.
            output_sequence[i] = backtrack_array[x][output_sequence[i-1]]
        
        if testing:
            return transition_array 
        
        #return the sequence.
        if len(self.state_tag_map) == len(self.transition_matrix):
            return self.state_tag_map[output_sequence[::-1]]

        #return the reverse of the array
        return output_sequence[::-1]
                            
    def _probability_of_transitions(self, f_array, b_array, probability_of_arr, arr):
        #allocate space for the probability of taking a transition from i at at time t to j at time t for j,i =[0,num_states],[0,num_states-1].
        prob_of_transitions = np.zeros((len(arr),len(self.transition_matrix),len(self.transition_matrix)),dtype = np.float64)
        
        for t in range(len(arr)-1):
            prob_of_transitions[t] = np.multiply(np.multiply(np.multiply( f_array[t].T , self.transition_matrix).T , self.emission_matrix[arr[t+1]].T), b_array[t+1].T).T / probability_of_arr 
        
        return prob_of_transitions
    
    def _probability_of_states(self, prob_of_transitions, f_array, probability_of_arr,arr):
        #using the transition probabilities sum the columns along the j axis for each row to make the probabilities for state i
        #allocate space for the probability of being at state i at time t.
        prob_of_states = np.zeros([len(arr),len(self.transition_matrix)],dtype = np.float64)
    
        prob_of_states = prob_of_transitions.sum(2)
        
        #set state probabilities at last time  to the results of the forward array divided by the words current probability
        #Fixed: makes a prob dist.
        prob_of_states[-1] = f_array[-1] / probability_of_arr 
        
        
        return prob_of_states
    
    def __epoch():
        pass
    
    def baum_welch(self,arrs):
        ''' input: arr (type: numpy.array)
            optimizes hidden markov model to recognize a given sequence.
        '''
        #set the current and last probabilities for first iteration(do while loop).
        current_probs = np.full((len(arrs)),0.0000000001,dtype=np.float64)
        last_probs = np.zeros([len(arrs)],dtype=np.float64)
        
    
        #while probabilities of observations are still rising.
        while current_probs.sum() > last_probs.sum():
            #copy last probability
            last_probs = current_probs.copy()
            not_used = 0
            
            
            #initialize temporary matricies. Set all to zero for summations amongst different observations sequences.
            initial_m = np.matrix(np.zeros(self.initial_matrix.shape,dtype=np.float64))
            prob_of_transitions_m = np.matrix(np.zeros(self.transition_matrix.shape,dtype=np.float64))
            emission_m = deepcopy(self.emission_matrix)
            prob_of_states_m = np.matrix(np.zeros(len(self.transition_matrix),dtype=np.float64))
            prob_of_states_except_last_m = np.matrix(np.zeros(len(self.transition_matrix),dtype=np.float64))
            
            
            for e in emission_m:
                emission_m[e][:] = 0
            
            #for each observation sequence
            for i,arr in enumerate(arrs):
                if int(i%(len(arrs)*0.1)) == 0:
                    print(i/float(len(arrs)))
                #run forward and backward algorithm and store results.
                f_array = self._forward_algorithm(arr)
                b_array = self._backward_algorithm(arr)
                
                c = np.multiply(f_array,b_array).sum(1)
                
                #get the probability of this observation sequence occuring.
                probability_of_arr = f_array[-1].sum()
                
                
                #set this words current probability
                current_probs[i] = probability_of_arr               
                
                if probability_of_arr != 0.0:
                    
                    prob_of_transitions = self._probability_of_transitions(f_array, b_array, probability_of_arr, arr)
                    
                    prob_of_states = self._probability_of_states(prob_of_transitions, f_array, probability_of_arr, arr)

                    
                    #set initial probabilities
                    initial_m += prob_of_states[0]
                    
                    #sum the states probs. along the time axis.(To get the average)
                    prob_of_states_m  += prob_of_states.sum(0)
                    
                    prob_of_states_except_last_m += prob_of_states[:-1].sum(0)
                    
                    
                    #sum the transitions probs. along the time axis.(To get the average)
                    prob_of_transitions_m += prob_of_transitions.sum(0) 
                    #!
                    t = len(prob_of_transitions)
                    s= prob_of_transitions.sum(2).sum(1)
                    d = prob_of_transitions[-1]
                    a= prob_of_transitions.sum(2)
                    b=a.sum()
                    c=b/len(self.transition_matrix)
                    
                    
                    
                    
                    #must be a slow spot at around 5000 words and 1000 sentences.
                    #for each possible observation by the hmm.
                    for entry in self.emission_matrix:
                        #works as indicator function
                        boolean_array = np.equal(np.array(arr,dtype=np.object),entry)
                        #broadcast over time axis
                        emission_m[entry] += (prob_of_states*boolean_array.reshape(len(arr),1)).sum(0)
                    
                    
                else:
                    print('sentence '+str(i)+' with value \"' + ' '.join(arr) + '\" has probability of 0 and was not counted in this iteration.')
                    not_used += 1
                    
            print('current probs:')
            print(current_probs.sum()) 
            print(current_probs.mean())
            print('last probs:')
            print(last_probs.sum()) 
            print(last_probs.mean()) 
            
            
            #divide the initial probabilites by there sum so that the new sum is 1.
            initial_m = initial_m/(len(arrs ) - not_used)
            b=initial_m.sum()#ERROR ON LAST ITERATION?
            if not np.isclose(b, 1.0):
                print('here')
            #do the same for each state and it's outgoing probabilities.
            prob_of_transitions_m = prob_of_transitions_m/prob_of_states_except_last_m.T 
            a= prob_of_transitions_m.sum(1)
            c=np.matrix(np.zeros(len(self.transition_matrix)))
            for e in emission_m:
                emission_m[e] = emission_m[e]/prob_of_states_m 
                c += emission_m[e]
                
            
            #now update the hmm's parameters.
            self.initial_matrix = initial_m
            self.transition_matrix = prob_of_transitions_m
            self.emission_matrix = emission_m
        return
    
    def __roulette(self,vec):
        #vec is a probability distribution
        assert(np.isclose(vec.sum(), 1.0))
        r = random.random()
        i = 0
        t = 0
        while( t < r):
            t += vec[i]
            i+=1
            
        return i - 1
            
        
    def talk(self, n=1):
        words = [w for w in self.emission_matrix]
        state_dists = np.array([np.array(dist)[0] for dist in self.emission_matrix.values()]).T 
        
        for _ in range(n):
            state = self.__roulette(np.array(self.initial_matrix)[0])
            output =  words[self.__roulette(state_dists[state])]
            while(True):
                print((output, state))
                if output == '.' or output == '?' or output == '!':
                    break
                state = self.__roulette(np.array(self.transition_matrix[state])[0])
                output = words[self.__roulette(state_dists[state])]
            
            
    
    def save_to_file(self,filename):
        '''save hmm parameters to a file.'''
        f = open(filename,'wb')
        pickle.dump(self.__dict__,f,2)
        f.close()
        
    def load_from_file(self,filename):
        '''Load hmm parameters from a file and set this hmm's parameters to them.'''
        f = open(filename,'rb')
        self.__dict__.update(pickle.load(f))
        f.close()

            
    
    def map_internal_states(self,tagged_sequences):
        '''Maps the numeric hidden states to meaningful tags given a small amount of
        tagged data. 
        the key_tags parameter is used to force hidden states to be exclusive
        to tags in key_tags.Also states will only return those tags instead of
        multiple tags.'''
        
        tagged_words = reduce((lambda x,y:x + y),tagged_sequences)
        tag_indicies = list(set([t for w,t in tagged_words]))
        
        if len(tag_indicies) != len(self.transition_matrix):
            print('The number of tags should equal the number of states.')
            return []
        
        self.state_tag_map = []
        state_tag_matrix = np.zeros([len(self.transition_matrix)]*2, dtype = np.int32)
        
        
        for sequence in tagged_sequences:
            our_tags = self.viterbi([w for w,t in sequence])
            if np.any(our_tags):
                for i in range(len(sequence)):
                    tag = sequence[i][1]
                    
                    state_tag_matrix[our_tags[i],tag_indicies.index(tag)] +=  1
        
         
        state_tag = self._gale_shapley(np.argsort(-state_tag_matrix.T),np.argsort(-state_tag_matrix))
        
        
        self.state_tag_map = np.array([tag_indicies[tag_index] for tag_index in state_tag])
        
        
                
    def _gale_shapley(self,matp,mata):
        '''given two matrices this function solve the stable marriage problem.'''
        if matp.shape == mata.shape and matp.shape[0] == matp.shape[1]:
            #males who are not married
            proposers_not_married = np.full(len(matp),fill_value=True, dtype=np.bool)
            #array for acceptor's best proposers so far
            accepted_rank = np.full(len(matp),fill_value=len(matp),dtype = np.int32)
            proposed_rank = np.full(len(matp),fill_value=len(matp),dtype = np.int32)
            
            proposer_marriages = np.full(len(matp),fill_value=-1,dtype = np.int32)
            acceptor_marriages = np.full(len(matp),fill_value=-1,dtype = np.int32)
            
            
            while(proposers_not_married.any()):
                for p in proposers_not_married.nonzero()[0]:
                    
                    for current_acceptor_rank,a in enumerate(matp[p]):
                        if current_acceptor_rank > accepted_rank[p]:
                            break
                        for current_proposer_rank,desired_proposer in enumerate(mata[a]):
                            if current_proposer_rank > proposed_rank[a]:
                                break
                            if desired_proposer == p:
                                old_marriage = acceptor_marriages[a]
                                proposer_marriages[p] = a
                                acceptor_marriages[a] = p
                                
                                proposers_not_married[p] = False
                                accepted_rank[p] = current_acceptor_rank
                                proposed_rank[a] = current_proposer_rank
                                if old_marriage != -1:
                                    proposers_not_married[old_marriage] = True
                                    accepted_rank[old_marriage] = len(matp)
                                    proposer_marriages[old_marriage] = -1
                            
                        
                        
            return proposer_marriages
        
            
        return []
    
    
        
        
#references:
#http://www.indiana.edu/~iulg/moss/hmmcalculations.pdf
#https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm
#https://people.eecs.berkeley.edu/~stephentu/writeups/hmm-baum-welch-derivation.pdf


                      
                                 
        
        
        
