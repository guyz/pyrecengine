'''
Created on Apr 10, 2015

@author: GuyZ
'''

import numpy as np

class MF(object):
    '''
    classdocs
    '''


    def __init__(self, n_users, n_items, n_feat=10, epsilon=50, momentum=0.8, lambda_pq=0.1, lambda_bw=None):
        '''
        Constructor
        '''
        
        if lambda_bw is None:
            lambda_bw = lambda_pq
        
        self._n_users = n_users
        self._n_items = n_items
        self._n_feat = n_feat
        self._epsilon = epsilon
        self._momentum = momentum
        self._lambda_pq = lambda_pq
        self._lambda_bw = lambda_bw
        
    
    def fit(self, X, max_epochs = 10, n_batches=10):
        '''
        Fits the matrix factorization and biases,
        see equation (4) 
        in Koren et al. (http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf)
        
            INPUT:
                X             (Nx3) ndarray or list of the same size.
                              Each row is a triplet (user, item, rating)
                max_epochs    Number of iterations over X (for SGD)
                n_batches     X is split to len(X)/n_batches mini-batches (for SGD)
            OUTPUT:
                None.
                The model is fitted and can be used for predictions.
                self._b_i, self._b_u, self._Q and self._P hold the learned
                factorization parameters.
        '''
    
        # initialization
        self._P   = np.random.randn(self._n_users, self._n_feat)
        self._Q   = np.random.randn(self._n_items, self._n_feat)
        self._b_u = np.random.randn(self._n_users)
        self._b_i = np.random.randn(self._n_items)
        
        P_inc     = np.zeros( (self._n_users, self._n_feat) )
        Q_inc     = np.zeros( (self._n_items, self._n_feat) )
        b_u_inc   = np.zeros(  self._n_users  )
        b_i_inc   = np.zeros(  self._n_items  )
        
        self._mu  = np.mean(X[:,2])
        
        N         = np.floor(X.shape[0]/n_batches) # TODO: TMP -- truncate last batch. Should change
        X_train   = X[0:n_batches*N,:] # TODO: TMP - truncate last batch. Need to change
        
        
        # start training
        err_train = []
        cost_func = np.inf
        for epoch in xrange(max_epochs):
            train_idx = np.random.permutation(np.arange(X_train.shape[0])).astype(int)
            X_train = X_train[train_idx,:]
            
            for batch in xrange(n_batches):
                print 'epoch %d batch %d' % (epoch, batch)
                
                u_idx   = X_train[batch*N:(batch+1)*N,0].astype(int)
                i_idx   = X_train[batch*N:(batch+1)*N,1].astype(int)
                rating  = X_train[batch*N:(batch+1)*N,2]
                
                ### compute cost and predictions
                pred    = np.sum(self._Q[i_idx,:]*self._P[u_idx,:],1) + self._mu + self._b_u[u_idx] + self._b_i[i_idx] # + sum(w1_Y1(aa_p,:).*Xu(aa_p,:),2);
                err_ui  = (pred - rating)
                
                cost_func = np.sum(err_ui**2 + 0.5*(self._lambda_pq*np.sum( (self._Q[i_idx,:]**2 + self._P[u_idx,:]**2),1) + 
                                             self._lambda_bw*(self._b_u[u_idx]**2 + self._b_i[i_idx]**2 ) ))
                
                ### compute gradients
                IO      = np.tile(2*err_ui,(self._n_feat,1)).T
                Ix_i    = IO*self._P[u_idx,:] + self._lambda_pq*self._Q[i_idx,:]
                Ix_u    = IO*self._Q[i_idx,:] + self._lambda_pq*self._P[u_idx,:]
                Ix_b_i  = 2*err_ui + self._lambda_bw*self._b_i[i_idx]
                Ix_b_u  = 2*err_ui + self._lambda_bw*self._b_u[u_idx]
                
                dQ      = np.zeros( (self._n_items, self._n_feat) )
                dP      = np.zeros( (self._n_users, self._n_feat) )
                db_i    = np.zeros(  self._n_items  )
                db_u    = np.zeros(  self._n_users  )
                
                # TODO: vectorize
                for ii in xrange(int(N)):
                    dQ[i_idx[ii],:]     += Ix_i[ii,:]
                    dP[u_idx[ii],:]     += Ix_u[ii,:]
                    db_i[i_idx[ii]]     += Ix_b_i[ii]
                    db_u[u_idx[ii]]     += Ix_b_u[ii]
                    
                ### update weights
                Q_inc    = self._momentum*Q_inc + self._epsilon*dQ/N
                self._Q -= Q_inc;
            
                P_inc    = self._momentum*P_inc + self._epsilon*dP/N
                self._P -= P_inc
                
                b_i_inc    = self._momentum*b_i_inc + self._epsilon*db_i/N
                self._b_i -= b_i_inc
                
                b_u_inc    = self._momentum*b_u_inc + self._epsilon*db_u/N
                self._b_u -= b_u_inc
                
            ### epoch complete
            curr_err = np.sqrt(cost_func/N)
            err_train.append(curr_err)
                    
            print 'epoch %d train error=%f' % (epoch, curr_err)
    
class ContentMF(MF):
    '''
    classdocs
    '''
    
    def fit(self, X, X_u=None, X_i=None, max_epochs = 10, n_batches=10):
        
        if X_u is None and X_i is None:
            # this is content-less CF
            super(ContentMF, self).fit(X, max_epochs, n_batches)
            
        # TODO: ...
        

# TODO: move tests to another file.
import cPickle
X = cPickle.load(open('/Users/GuyZ/Dropbox (MIT)/mit/grad/courses/cs181 machine learning/practicals/p3/data/train_set.pkl', 'rb'))
mf = MF(233286, 2000, lambda_bw=1.0)
X1 = X.copy().astype(float)
X1[:,2] = np.log(X1[:,2]*1.0)
mf.fit(X1)
        