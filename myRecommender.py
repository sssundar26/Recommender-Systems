import numpy as np


def my_recommender(rate_mat, lr, with_reg):
    """

    :param rate_mat:
    :param lr:
    :param with_reg:
        boolean flag, set true for using regularization and false otherwise
    :return:
    """

    # TODO pick hyperparams
    max_iter = 500
    if(with_reg):
        learning_rate = 2e-4
        reg_coef = 0.9981
    else:
        learning_rate=0.0002
        reg_coef=0
    n_user, n_item = rate_mat.shape[0], rate_mat.shape[1]

    U = np.random.rand(n_user, lr) / lr
    V = np.random.rand(n_item, lr) / lr

    # TODO implement your code here
    Iter=0
    Mask=rate_mat>0
    
    while(Iter<max_iter):
        Iter=Iter+1
        U_new= U+ (2*learning_rate*(((rate_mat-U.dot(V.T))*Mask)@V))- (2*learning_rate*reg_coef*U)
        V_new= V+ (2*learning_rate*(((rate_mat-U.dot(V.T))*Mask).T@U))-(2*learning_rate*reg_coef*V)
        U=U_new
        V=V_new


    return U, V

    return U, V