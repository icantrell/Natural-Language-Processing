import nlp
import numpy as np

def get_test_model0_str():
    return ['a','b','b','a']

def get_test_model0():
    A = nlp.hmm()
    A.initial_matrix = np.matrix([0.4,0.6])
    A.transition_matrix = np.matrix([[0.1,0.9],[0.5,0.5]])
    A.emission_matrix = {'a' : np.matrix([0.7,0.4]), 'b': np.matrix([0.3,0.6])}
    
    return A

def get_test_model0_forward_arr():
    return np.matrix([[0.28, 0.24], [0.0444, 0.2232], [0.034812, 0.090936], [0.03426444, 0.03071952]])

def get_test_model0_backward_arr():
    return np.matrix([[0.133227,0.115335], [0.3099, 0.2295], [0.43, 0.55], [1,1]])

def get_test_model0_transition_probs():
    return np.matrix([[[0.040058500589991754, 0.5339840785326102],[0.17167928824282172,0.25427813263457627]],[[0.00881386729894577, 0.20292392153386776],[0.22153774562215042,0.5667244655450361]], [[0.036909274227055415, 0.28927876971486505],[0.4897762463229387,0.279872140755965]], [[0,0],[0,0]]])

def get_test_model0_state_probs():
    return np.matrix([[ 0.57404258,  0.42595742],
       [ 0.21173779,  0.78826221],
       [ 0.32618804,  0.76964839],
       [ 0.52727535,  0.47272465]])


#random parameters tests

def test_random_mat():
    A = nlp.hmm()
    iters = 10000
    num_states = 5
    
    sum_mat = np.matrix(np.zeros((5,5)))
    sum_mat2 = np.matrix(np.zeros((2,5)))
    sum_mat3 = np.matrix(np.zeros((5)))
    
    for i in range(iters):
        A.randomize(['b','a'],5)
        sum_mat += A.transition_matrix
        
        t = sum(A.transition_matrix[0],0)
        sum_mat2[0] += A.emission_matrix['a']
        sum_mat2[1] += A.emission_matrix['b']
        
        sum_mat3+= A.initial_matrix
        
    assert(np.isclose(np.average(np.sum(sum_mat,1)/iters),1.0,atol = 0.01))
    assert(np.isclose(np.average(np.sum(sum_mat2,0)/iters),1.0,atol = 0.01))
    assert(np.isclose(np.average(np.sum(sum_mat3,1)/iters),1.0,atol = 0.01))
    
    assert(A.transition_matrix.shape==(5,5))
    assert(A.emission_matrix['a'].shape == (1,5))
    assert(A.emission_matrix['b'].shape == (1,5))
    assert(A.initial_matrix.shape == (1,5))

def test_forward_backward_alg():
    A = nlp.hmm()
    A.randomize(['a','b'],2)
    
    s = 'ab'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r0= np.multiply(b_arr  , f_arr).sum(1)
    s = 'aa'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r1= np.multiply(b_arr  , f_arr).sum(1)
    s = 'bb'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r2= np.multiply(b_arr  , f_arr).sum(1)
    s = 'ba'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r3= np.multiply(b_arr  , f_arr).sum(1)    
    
    r = r0 + r1 + r2 + r3
    assert(np.all(np.isclose(r,np.ones(r.shape),atol=0.001)))
   
    
#set parameters tests
def test_forward():
    A = get_test_model0()
    f_arr = A._forward_algorithm(['a','b','b','a'])
    assert(np.all(np.isclose(get_test_model0_forward_arr(), f_arr)))
    
#set parameters tests
def test_backward():
    A = get_test_model0()
    b_arr = A._backward_algorithm(['a','b','b','a'])
    assert(np.all(np.isclose(get_test_model0_backward_arr(), b_arr)))
    
def test_probability_of_transitions():
    arr = get_test_model0_str()
    A = get_test_model0()
    pa = A._forward_algorithm(arr)[-1].sum()
    pots = A._probability_of_transitions(A._forward_algorithm(arr), A._backward_algorithm(arr), pa, arr)
    print(pots)
    
    
    
#should've used inheritence to test private methods.
test_probability_of_transitions()



def test_viterbi():
    A = get_test_model0()
    
    seq = A.viterbi('this is a test sentence, which is used for testing long sentences.', testing=True)


