import numpy as np
from cvxopt import matrix
from cvxopt import solvers


def rbf_kernel(X1, X2, sig):
    (datanum1,_) = X1.shape
    (datanum2,_) = X2.shape

    XXh1 = np.matmul(np.sum(X1**2,axis=1,keepdims=True),
                     np.ones([1,datanum2]))
    XXh2 = np.matmul(np.sum(X2**2,axis=1,keepdims=True),
                        np.ones([1,datanum1]))
    omega = XXh1+XXh2.T - 2*np.matmul(X1,X2.T)
    omega = np.exp(-omega/(2*sig*sig))
    return omega


#******************************************************
#                 NN forward pass
#******************************************************

def evalSVM(X,SV_X, SV_Y, SV_a,b_bound,sig):

    Y_out_tmp = np.matmul(rbf_kernel(X, SV_X, sig),(SV_a* SV_Y))
    Ysvm = Y_out_tmp + b_bound

    return Ysvm


#******************************************************
#                 NN backward pass
#******************************************************
def quadprogSVM(K, Y, C):

    # x = cvxopt.solvers.qp(H,f,A,b,Aeq,beq) returns a vector x that
    #
    # minimizes 1/2*x'*H*x + f'*x
    #       subject to
    #           A⋅x≤b,
    #           Aeq⋅x=beq,

    (datanum,_) = Y.shape
    H = matrix(np.matmul(Y,Y.T)*K, tc='d')
    f = -matrix(np.ones([datanum,1]), tc='d')
    A = matrix(-np.row_stack((np.eye(datanum), np.eye(datanum))), tc='d')
    b = matrix(np.row_stack((np.zeros([datanum, 1]), C * np.ones([datanum, 1]))), tc='d')
    Aeq = matrix(Y.T, tc='d')
    beq = matrix(0, tc='d')
    sol = solvers.qp(H, f, A, b, Aeq, beq)
    a_train = np.array(sol['x'])
       
    return a_train

