import nlp
import numpy as np

def get_test_model0():
    A = nlp.hmm()
    A.initial_matrix = np.matrix([0.4,0.6])
    A.transition_matrix = np.matrix([[0.1,0.9],[0.5,0.5]])
    A.emission_matrix = {'a' : np.matrix([0.7,0.4]), 'b': np.matrix([0.3,0.6])}
    
    return A

def get_test_model0_forward_arr():
    return np.matrix([[0.28, 0.24], [0.0444, 0.06696], [0.034812, 0.06696], [0.02587284, 0.02592432]])

#random parameters tests

def test_random_mat():
    A = nlp.hmm()
    iters = 10000
    num_states = 5
    
    sum_mat = np.zeros((5,5))
    sum_mat2 = np.zeros((2,5))
    sum_mat3 = np.zeros((5))
    
    for i in range(iters):
        A.randomize(['b','a'],5)
        sum_mat += A.transition_matrix
        t = sum(A.transition_matrix[0],0)
        sum_mat2[0] += A.emission_matrix['a']
        sum_mat2[1] += A.emission_matrix['b']
        sum_mat3+= A.initial_matrix
        
    assert(np.isclose(np.average(np.sum(sum_mat,1)/iters),1.0,atol = 0.01))
    assert(np.isclose(np.average(np.sum(sum_mat2,0)/iters),1.0,atol = 0.01))
    assert(np.isclose(np.average(np.sum(sum_mat3,0)/iters),1.0,atol = 0.01))
    
    assert(A.transition_matrix.shape==(5,5))
    assert(A.emission_matrix['a'].shape == (5,))
    assert(A.emission_matrix['b'].shape == (5,))
    assert(A.initial_matrix.shape == (5,))

def test_forward_backward_alg():
    A = nlp.hmm()
    A.randomize(['a','b'],2)
    
    s = 'ab'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r0= (b_arr  * f_arr).sum(1)
    s = 'aa'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r1= (b_arr  * f_arr).sum(1)
    s = 'bb'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r2= (b_arr  * f_arr).sum(1)
    s = 'ba'
    b_arr = A._backward_algorithm(list(s))
    f_arr = A._forward_algorithm(list(s))
    r3= (b_arr  * f_arr).sum(1)    
    
    r = r0 + r1 + r2 + r3
    assert(np.all(np.isclose(r,np.ones(r.shape),atol=0.001)))
    
    
#set parameters tests
def test_forward():
    A = get_test_model0()
    f_arr = A._forward_algorithm(['a','b','b','a'])
    assert(np.all(np.isclose(get_test_model0_forward_arr(), f_arr)))
    
    