from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    for i in range(X.shape[0]):
        
        scores  = X[i].dot(W)
        scores -= np.max(scores)
        sum_exp = np.sum( np.exp( scores ) )
        cor_exp = np.exp(scores[y[i]])
        loss   += -cor_exp + np.log( sum_exp )
     
    dW[:, y[i]] += (-1) * (sum_exp - cor_exp) / sum_exp * X[i]
    for j in xrange(W.shape[1]):
        
          # pass correct class gradient
        if j == y[i]:
            continue
            
          # for incorrect classes
            dW[:, j] += np.exp(scores[j]) / sum_exp * X[i]
    
    loss /= X.shape[0]
    loss += reg * np.sum( W * W )
    dW   /= X.shape[0]
    dW   += 2 * reg * W
       
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
#     num_classes = W.shape[1]
#     num_train = X.shape[0]

    
    #loss
#     scores  = X.dot(W)
#     scores -= np.max(scores)
#     sum_exp = np.sum( np.exp(scores), axis = 1)
#     cor_exp = np.exp(scores[range(X.shape[0]), y])
#     loss    = -np.sum( np.log( cor_exp / sum_exp ) )
#     loss   /= X.shape[0]
#     loss   += reg * np.sum( W * W )

#     scores = X.dot(W)
#     scores -= scores.max()
#     scores = np.exp(scores)
#     scores_sums = np.sum(scores, axis=1)
#     cors = scores[range(num_train), y]
#     loss = cors / scores_sums
#     loss = -np.sum(np.log(loss))/num_train + reg * np.sum(W * W)
    
#     grad 
#     s = np.divide(scores, sum_exp.reshape(X.shape[0], 1))
#     s[range(X.shape[0]), y] = - (sum_exp - cor_exp) / sum_exp
#     dW = X.T.dot(s)
#     dW /= X.shape[0]
#     dW += 2 * reg * W    

#     s = np.divide(scores, scores_sums.reshape(num_train, 1))
#     s[range(num_train), y] = - (scores_sums - cors) / scores_sums
#     dW = X.T.dot(s)
#     dW /= num_train
#     dW += 2 * reg * W


    num_train = X.shape[0]
    scores = X.dot(W)
  # Normalize the scores to avoid computational problems with the exponential
  # Normalize with max as zero
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    loss = -np.sum(np.log(probs[np.arange(num_train),y]))

  # Divide the loss by the number of trainig examples
    loss /= num_train
  # Add regularization
    loss += 0.5*reg*np.sum(W*W)

  # Compute the gradient
  # gradWy=X(Si-1)
  # gradWj=SjX
    dprobs = probs
    dprobs[np.arange(num_train),y] -= 1
    dW = X.T.dot(dprobs)
    dW /= num_train

  # Gradient regularization
    dW += reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
