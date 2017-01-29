# Copyright (c) 2016-2017, Zhenwen Dai.
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from GPy.models import SSGPLVM
import GPy

class SITAR(SSGPLVM):
    """
    Sparse latent varIable model of TrAnscritional Regulation (SITAR)
    """
    def __init__(self, data, connM):
        Y = data

        kern = GPy.kern.Linear(connM.shape[1],ARD=False,variances=1)
        kern.variances.fix(warning=False)
        X_prior = connM.copy()*0.9
        X_prior[connM==0] = 1e-9
        X_prior[connM>0] = 0.2*np.random.rand(connM.sum())+0.4

        X_init = Y.dot(np.random.randn(Y.shape[1],connM.shape[1])/np.sqrt(Y.shape[1]))

        Z = np.eye(connM.shape[1])


        super(SITAR,self).__init__(Y, Z.shape[1],X=X_init, Z=Z, Gamma=X_prior, SLVM=True, alpha=2., beta=2., connM=connM.copy(),
                       kernel=kern,num_inducing=connM.shape[1],group_spike=False)

        self.X.tau[:] = 2.
        self.X.gamma[connM==0].fix(warning=False)
        self.Z.fix(warning=False)
        self.likelihood.variance[:] = 0.01

    def optimize(self,max_iters=1000,verbose=True,bfgs_factor=10,**kw):

        self.likelihood.variance.fix(warning=False)
        super(SITAR,self).optimize(max_iters=max_iters/10,messages=verbose,bfgs_factor=bfgs_factor,**kw)
        self.likelihood.variance.constrain_positive(warning=False)
        super(SITAR,self).optimize(max_iters=max_iters,messages=verbose,bfgs_factor=bfgs_factor,**kw)
