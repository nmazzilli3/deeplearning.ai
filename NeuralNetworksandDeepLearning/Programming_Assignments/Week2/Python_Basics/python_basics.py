# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:55:21 2020

@author: nmazzilli24
"""

import numpy as np
import math


'''
1 - Building basic functions with numpy
Numpy is the main package for scientific computing in Python. It is maintained by a large community (www.numpy.org). In this exercise you will learn several key numpy functions such as np.exp, np.log, and np.reshape. You will need to know how to use these functions for future assignments.

1.1 - sigmoid function, np.exp()
Before using np.exp(), you will use math.exp() to implement the sigmoid function. You will then see why np.exp() is preferable to math.exp().

Exercise: Build a function that returns the sigmoid of a real number x. Use math.exp(x) for the exponential function.

Reminder:  sigmoid(x)=11+e−xsigmoid(x)=11+e−x  is sometimes also known as the logistic function. It is a non-linear function used not only in Machine Learning (Logistic Regression), but also in Deep Learning.
'''

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+math.exp(-x))
    ### END CODE HERE ###

    return s

'''
Any time you need more info on a numpy function, we encourage you to look at the official documentation.

You can also create a new cell in the notebook and write np.exp? (for example) to get quick access to the documentation.

Exercise: Implement the sigmoid function using numpy.

Instructions: x could now be either a real number, a vector, or a matrix. The data structures we use in numpy to represent these shapes (vectors, matrices...) are called numpy arrays. You don't need to know more for now.

'''

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    s = 1/(1+np.exp(-x))
    ### END CODE HERE ###
    
    return s

'''
1.2 - Sigmoid gradient
As you've seen in lecture, you will need to compute gradients to optimize loss functions using backpropagation. Let's code your first gradient function.

Exercise: Implement the function sigmoid_grad() to compute the gradient of the sigmoid function with respect to its input x. The formula is:
sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))(2)
(2)sigmoid_derivative(x)=σ′(x)=σ(x)(1−σ(x))
 
You often code this function in two steps:

Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
Compute  σ′(x)=s(1−s)σ′(x)=s(1−s)
'''

def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    s = sigmoid(x)
    ds = s*(1-s)
    ### END CODE HERE ###
    
    return ds
'''
1.3 - Reshaping arrays
Two common numpy functions used in deep learning are np.shape and np.reshape().

X.shape is used to get the shape (dimension) of a matrix/vector X.
X.reshape(...) is used to reshape X into some other dimension.
For example, in computer science, an image is represented by a 3D array of shape  (length,height,depth=3)(length,height,depth=3) . However, when you read an image as the input of an algorithm you convert it to a vector of shape  (length∗height∗3,1)(length∗height∗3,1) . In other words, you "unroll", or reshape, the 3D array into a 1D vector.

Exercise: Implement image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1). For example, if you would like to reshape an array v of shape (a, b, c) into a vector of shape (a*b,c) you would do:

v = v.reshape((v.shape[0]*v.shape[1], v.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
Please don't hardcode the dimensions of image as a constant. Instead look up the quantities you need with image.shape[0], etc.                                                                               
  
'''                                                                               

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    ### END CODE HERE ###
    
    return v

'''
1.4 - Normalizing rows
Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization. Here, by normalization we mean changing x to  x∥x∥x∥x∥  (dividing each row vector of x by its norm).

For example, if
x=[023644](3)
(3)x=[034264]
 
then
∥x∥=np.linalg.norm(x,axis=1,keepdims=True)=[556⎯⎯⎯⎯√](4)
(4)∥x∥=np.linalg.norm(x,axis=1,keepdims=True)=[556]
 
and
x_normalized=x∥x∥=0256√35656√45456√(5)
(5)x_normalized=x∥x∥=[03545256656456]
 
Note that you can divide matrices of different sizes and it works fine: this is called broadcasting and you're going to learn about it in part 5.

Exercise: Implement normalizeRows() to normalize the rows of a matrix. After applying this function to an input matrix x, each row of x should be a vector of unit length (meaning length 1).

'''

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x,axis =1, keepdims = True)
    
    # Divide x by its norm.
    x = x/x_norm
    ### END CODE HERE ###

    return x

'''
1.5 - Broadcasting and the softmax function¶
A very important concept to understand in numpy is "broadcasting". It is very useful for performing mathematical operations between arrays of different shapes. For the full details on broadcasting, you can read the official broadcasting documentation.

Exercise: Implement a softmax function using numpy. You can think of softmax as a normalizing function used when your algorithm needs to classify two or more classes. You will learn more about softmax in the second course of this specialization.

'''

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (m,n).

    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp,axis =1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum

    ### END CODE HERE ###
    
    return s

'''
2.1 Implement the L1 and L2 loss functions
Exercise: Implement the numpy vectorized version of the L1 loss. You may find the function abs(x) (absolute value of x) useful.

Reminder:

The loss is used to evaluate the performance of your model. The bigger your loss is, the more different your predictions (ŷ y^) are from the true values (yy). In deep learning, you use optimization algorithms like Gradient Descent to train your model and to minimize the cost.
L1 loss is defined as:

'''

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(abs(y-yhat))
    ### END CODE HERE ###
    
    return loss


#Testing functions
def main():
    print("Testing basic_sigmoid function")
    test_sig = basic_sigmoid(3)
    sig_sol = 0.952
    tol = 0.001
    assert(abs(test_sig-sig_sol) < tol)
    print("Passed the basic_sigmoid function!")
    print()
    
    #Testing the sigmoid with np apis 
    print("Testing sigmoid function")
    x = np.array([1, 2, 3])
    test_sig_np = sigmoid(x)
    sig_np_sol = [ 0.73105858, 0.88079708, 0.95257413]

    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    assert(np.allclose(test_sig_np,sig_np_sol, tol))
    print("Passed the basic_sigmoid function!")
    print()    
    
    #Sigmoid Derivative Test
    print("Testing sigmoid_derivative function")
    x = np.array([1, 2, 3])
    test_sig_deriv = sigmoid_derivative(x)
    test_sig_deriv_sol = [ 0.19661193, 0.10499359, 0.04517666]
    assert(np.allclose(test_sig_deriv,test_sig_deriv_sol, tol))
    print("Passed the sigmoid_derivative function!")
    print()     
    
    #Normalize Row Test
    print("Testing normalizeRows function")
    x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
    norm_row = normalizeRows(x)
    norm_row_sol = [[ 0., 0.6, 0.8 ], [ 0.13736056, 0.82416338, 0.54944226]]
    assert(np.allclose(norm_row,norm_row_sol, tol))
    print("Passed the normalizeRows function!")
    print() 
    
    #Testing Softmax Funciton
    print("Testing softmax function")
    x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
    softmax_res = softmax(x)
    softmax_sol = [[ 9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04], [ 8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]]
    np.testing.assert_allclose(softmax_res,softmax_sol,rtol=1e-5, atol=0)
    print("Passed the softmax function!")
    print() 
    
    #Testing L1 Function 
    print("Testing L1 function")
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    L1_test = L1(yhat,y)
    L1_sol = 1.1 
    assert(abs(L1_test-L1_sol) < tol)
    print("Passed the L1 function!")
    print()  
    
    ''' Framework for Computational Time
    import time

    x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
    x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
    
    ### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
    tic = time.process_time()
    dot = 0
    for i in range(len(x1)):
        dot+= x1[i]*x2[i]
    toc = time.process_time()
    print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    
    ### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
    tic = time.process_time()
    outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
    for i in range(len(x1)):
        for j in range(len(x2)):
            outer[i,j] = x1[i]*x2[j]
    toc = time.process_time()
    print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    
    ### CLASSIC ELEMENTWISE IMPLEMENTATION ###
    tic = time.process_time()
    mul = np.zeros(len(x1))
    for i in range(len(x1)):
        mul[i] = x1[i]*x2[i]
    toc = time.process_time()
    print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
    
    ### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
    W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
    tic = time.process_time()
    gdot = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        for j in range(len(x1)):
            gdot[i] += W[i,j]*x1[j]
    toc = time.process_time()
    print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
'''

    



    

    
    




if __name__ == '__main__':
    main()