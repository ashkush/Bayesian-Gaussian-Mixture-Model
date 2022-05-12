%pyspark
################ Distributed Mixture Models ###########################
from abc import abstractmethod
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from pyspark.ml.feature import StandardScaler
from collections import Counter
import numpy as np
from itertools import product,combinations,permutations
from scipy import linalg
from functools import partial
from scipy.special import logsumexp
from scipy.special import betaln, digamma, gammaln
import math
import datetime
from pyspark import StorageLevel
import random


def _compute_precision_cholesky(covariances):
    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError('estimate_precision_error_message')
        precisions_chol[k] = linalg.solve_triangular(cov_chol,np.eye(n_features),lower=True).T
    
    return precisions_chol



def _compute_log_det_cholesky(matrix_chol, n_features):
    n_components, _, _ = matrix_chol.shape
    log_det_chol = (np.sum(np.log(matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]), 1))
    return log_det_chol



def _estimate_gaussian_parameters(features_resp,n_samples,n_features,n_components,reg_covar):  
    nk=features_resp.reduce(lambda x,y:np.array(x)+np.array(y))[n_features:]         
    nk=np.array([x+10 * np.finfo(float).eps if x==0 else float(x) for x in nk])   
    
    def computeMean(iterator):
      arr=np.array(list(iterator))      
      to_Return=np.einsum('ij,ik->ikj',arr[:,0:n_features],arr[:,n_features:]) 
      return to_Return

    #Compute Cluster Means
    means=features_resp.mapPartitions(computeMean).reduce(lambda x,y:np.array(x)+np.array(y))/nk[:,np.newaxis]    
        
    #Compute Cluster covariances
    def computeCovs(iterator):
      arr=np.array(list(iterator))
      to_Return=np.empty((arr.shape[0],n_components,n_features,n_features))   
      for k in range(n_components):
        diff=arr[:,0:n_features]-means[k]
        temp=np.einsum('ab,ad->abd',diff,diff)
        to_Return[:,k]=temp*((arr[:,n_features+k])[:,None,None])        
      return to_Return

    covs=features_resp.mapPartitions(computeCovs).reduce(lambda x,y:np.array(x)+np.array(y))/nk[:,np.newaxis,np.newaxis] 
    for k in range(n_components):      
      covs[k].flat[::n_features + 1]+= reg_covar
    return nk,means,covs


def _log_wishart_norm(degrees_of_freedom, log_det_precisions_chol, n_features):
    
    return -(degrees_of_freedom * log_det_precisions_chol +
             degrees_of_freedom * n_features * .5 * math.log(2.) +
             np.sum(gammaln(.5 * (degrees_of_freedom -
                                  np.arange(n_features)[:, np.newaxis])), 0))


def _initialize_cluster_assignments(X,n_components,seed=1):
      #K-Means
      features=X.columns
      stages=[]
      assembler=VectorAssembler(inputCols=features,outputCol='assembledVectors')
      stages.append(assembler)
      #Scaling
      standardizer = StandardScaler(inputCol='assembledVectors',outputCol='features')
      stages.append(standardizer)
      kmeans = KMeans().setK(n_components).setSeed(seed)
      stages.append(kmeans)
      pipeline=Pipeline(stages=stages)
      model = pipeline.fit(X)
      cols_to_select=X.columns+['prediction']
      X=model.transform(X).select(*cols_to_select)
      return X


class BaseMixture:
    
    
    def __init__(self,n_components,tol,max_iter,reg_covar,n_inits):
        self.n_components=n_components
        self.tol=tol
        self.max_iter=max_iter
        self.reg_covar=reg_covar
        self.n_inits=n_inits
         
    @abstractmethod
    def _get_parameters(self):
        pass
    
    @abstractmethod
    def _set_parameters(self,best_params):
        pass
        
    @abstractmethod
    def _check_parameters(self,X):
        pass
    
    @abstractmethod
    def _compute_lower_bound(self,log_resp, log_prob_norm): 
        pass
    
    @abstractmethod
    def _initialize_params(self,features_resp):
        pass
    
    @abstractmethod
    def _estimate_log_weights(self):
        pass
    
    @abstractmethod
    def _compute_log_responsibility(self,features_resp,log_det,log_weights,sum_log_prob_norm):
        pass
        
    
    def _estimate_log_resp(self,features_resp,sum_log_prob_norm):
        log_det = _compute_log_det_cholesky(self.precisions_cholesky_, self.n_features)
        log_weights=self._estimate_log_weights()  
        return self._compute_log_responsibility(features_resp,log_det,log_weights,sum_log_prob_norm)
        
        
        
    def _e_step(self,features_resp,sum_log_prob_norm):        
        return self._estimate_log_resp(features_resp,sum_log_prob_norm)        
        
    
    @abstractmethod
    def _m_step(self,features_resp):
        pass
    
    def fit(self,X):
        self.fit_predict(X)
        return self
    
    def fit_predict(self,X):
                 
        n_samples=X.count()
        n_features=len(X.columns)
        n_components=self.n_components
        self.n_samples=n_samples
        self.n_features=n_features
        
        #Check Parameters
        self._check_parameters(X) 
        
        for i in range(self.n_inits):
          #Initialize the Cluster Assignments
          if self.n_inits>1:
            seed=random.randint(0,100)
            print("Trying K-Means Initialization with seed "+str(seed))
          else:
            seed=1
          
          labeled_X=_initialize_cluster_assignments(X,self.n_components,seed=seed)
          
          
          def merge_features_resp(iterator):
            arr=np.array(list(iterator))
            
            features=arr[:,0:-1]
            label=arr[:,-1].astype(int)  
            resp = np.zeros((arr.shape[0],n_components))
            resp[np.arange(arr.shape[0]), label] = 1
            
            return np.hstack((features, resp))

          features_resp=labeled_X.rdd.mapPartitions(merge_features_resp)
          
          #Initialize the Gaussians
          self._initialize_params(features_resp)  
          lower_bound=-np.infty 
          self.converged_ = False
          
          for k,n_iter in enumerate(range(1, self.max_iter + 1)):
            prev_lower_bound = lower_bound
            sum_log_prob_norm=sc.accumulator(0)
            
            #E Step  
            result_rdd,sum_log_prob_norm = self._e_step(features_resp,sum_log_prob_norm)
            result_rdd=result_rdd.persist(StorageLevel.MEMORY_AND_DISK)
            result_rdd.count()
            
            if k!=0:
              features_resp.unpersist()

            features_resp=result_rdd                    
                  
              
            #M Step
            self._m_step(features_resp)
            
            
            #Compute Lower Bound
            lower_bound=self._compute_lower_bound(features_resp,sum_log_prob_norm.value/self.n_samples)
            
      
            change = lower_bound - prev_lower_bound
            if abs(change) < self.tol:
              self.converged_ = True
              print('Model Converged!')
              print('Iteration: '+str(k+1))
              break
          
          if self.converged_==False:
            print('Model did not converge!')   
          
          else:
            #Final E Step
            log_resp,_ = self._e_step(features_resp,sc.accumulator(0))

            def predict(iterator):
                arr=np.array(list(iterator))
                features=arr[:,0:n_features]
                prediction=np.argmax(arr[:,n_features:],axis=1)
                
                return np.hstack((features, prediction[:,None]))

            return log_resp.mapPartitions(predict) 
        return None 
    
    
       
    
class GMM(BaseMixture):
    
    def __init__(self,n_components,max_iter=100,tol=1e-3,reg_covar=1e-06,n_inits=1):
        BaseMixture.__init__(self,n_components=n_components,max_iter=max_iter,
                             tol=tol,reg_covar=reg_covar,n_inits=n_inits)
        
    
    def _get_parameters(self):
        return (self.weights_,
                self.means_,
                self.covariances_)
    
    
    def _set_parameters(self,best_params):
        pass
        
    def _check_parameters(self,X):
        pass
    
    def _compute_lower_bound(self,log_resp, log_prob_norm): 
        return log_prob_norm
    
    def _initialize_params(self,features_resp):
        #Initialize the weights,means,covariances
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(features_resp,self.n_samples,self.n_features,self.n_components,self.reg_covar) 
         
        self.weights_=[x/float(self.n_samples) for x in self.weights_] 
        #Compute Precision Matrix
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_)
        

    
    def _estimate_log_weights(self):
        return np.log(self.weights_)
        
    
    def _compute_log_responsibility(self,features_resp,log_det,log_weights,sum_log_prob_norm):
        means=self.means_
        precisions_chol=self.precisions_cholesky_
        n_features=self.n_features
        
        def parse(iterator):
          arr=np.array(list(iterator))
          log_prob=np.empty((arr.shape[0],means.shape[0]))      
          for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):          
              y = np.dot(arr[:,0:n_features], prec_chol) - np.dot(mu, prec_chol)
              log_prob[:, k] = np.sum(np.square(y), axis=1)
              
          log_prob=-.5 * (n_features* np.log(2 * np.pi) + log_prob) + log_det 
          weighted_log_prob= log_prob + log_weights     
          log_sum_exp=logsumexp(weighted_log_prob,axis=1)
          sum_log_prob_norm.add(np.sum(log_sum_exp))
          with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_sum_exp[:, None]            
          #Update responsibility      
          arr[:,n_features:] = log_resp
          return arr
        result=features_resp.mapPartitions(parse)
        return result,sum_log_prob_norm
    
    def _m_step(self,features_resp):
        n_features=self.n_features
        def exp(iterator):
          arr=np.array(list(iterator))    
          arr[:,n_features:]=np.exp(arr[:,n_features:])
          return arr  
        self.weights_, self.means_, self.covariances_ = _estimate_gaussian_parameters(features_resp.mapPartitions(exp),self.n_samples,self.n_features,self.n_components,self.reg_covar)
        self.weights_=[x/float(self.n_samples) for x in self.weights_] 
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_)


class DPGMM(BaseMixture):
    
    def __init__(self,n_components,max_iter=100,tol=1e-3,reg_covar=1e-06,weight_concentration_prior=None,mean_precision_prior=None,covariance_prior=None,degrees_of_freedom_prior=None,n_inits=1):
        BaseMixture.__init__(self,n_components=n_components,max_iter=max_iter,
                             tol=tol,reg_covar=reg_covar,n_inits=n_inits)
        
        self.weight_concentration_prior=weight_concentration_prior
        self.mean_precision_prior=mean_precision_prior
        self.degrees_of_freedom_prior=degrees_of_freedom_prior
        self.covariance_prior=covariance_prior 
        
    
    def _get_parameters(self):
        weight_dirichlet_sum = self.weight_concentration_[0] + self.weight_concentration_[1]
        tmp = self.weight_concentration_[1] / weight_dirichlet_sum
        self.weights_ = (
            self.weight_concentration_[0] / weight_dirichlet_sum *
            np.hstack((1, np.cumprod(tmp[:-1]))))
        self.weights_ /= float(np.sum(self.weights_))
        return (self.weights_,self.means_,self.covariances_)
    
    
    def _set_parameters(self,best_params):
        pass
        
    def _check_parameters(self,X):
        #Check weight parameters
        if self.weight_concentration_prior!=None:
          self.weight_concentration_prior_=self.weight_concentration_prior
        else:
          self.weight_concentration_prior_=1. / float(self.n_components)
    
        #Check Mean Parameters    
        if self.mean_precision_prior!=None:
          self.mean_precision_prior_=self.mean_precision_prior
        else:
          self.mean_precision_prior_=1.
    
        #Check precison Parameters
        if self.degrees_of_freedom_prior!=None:
          self.degrees_of_freedom_prior_=self.degrees_of_freedom_prior
        else:
          self.degrees_of_freedom_prior_=self.n_features
    
        #Check covariances parameters
        if self.covariance_prior is None:
          raise Exception('Covariance Prior Required')      
        else:
          self.covariance_prior_=self.covariance_prior
        
        #Compute Mean Prior 
        self.mean_prior_= np.array([m for m in X.select([F.mean(c).alias(c) for c in X.columns]).collect()[0]]) 
        
    
    
    def _compute_lower_bound(self,log_resp, log_prob_norm): 
        log_det_precisions_chol = (_compute_log_det_cholesky(self.precisions_cholesky_, self.n_features) -
              .5 * self.n_features * np.log(self.degrees_of_freedom_))

        log_wishart = np.sum(_log_wishart_norm(
                      self.degrees_of_freedom_, log_det_precisions_chol, self.n_features))
        
        log_norm_weight = -np.sum(betaln(self.weight_concentration_[0],self.weight_concentration_[1]))
        n_features=self.n_features
        def parse(iterator):
          arr=np.array(list(iterator))
          return np.sum(np.exp(arr[:,n_features:])*arr[:,n_features:],axis=1)
        sm=log_resp.mapPartitions(parse).reduce(lambda x,y:x+y)       
        
        return -sm - log_wishart - log_norm_weight - 0.5 * self.n_features * np.sum(np.log(self.mean_precision_)) 
        
    
    
    def _estimate_wishart_full(self, nk, xk, sk):
        self.degrees_of_freedom_ = self.n_features + nk
        self.covariances_ = np.empty((self.n_components, self.n_features,self.n_features))
    
        for k in range(self.n_components):
            diff = xk[k] - self.mean_prior_
            self.covariances_[k] = (self.covariance_prior_ + nk[k] * sk[k] +
                                    nk[k] * self.mean_precision_prior_ /
                                    self.mean_precision_[k] * np.outer(diff,diff))
    
        
        self.covariances_ /= (self.degrees_of_freedom_[:, np.newaxis, np.newaxis])



    def _estimate_weights(self, nk):       
       self.weight_concentration_ = (1. + nk,(self.weight_concentration_prior_ +
            np.hstack((np.cumsum(nk[::-1])[-2::-1], 0)))) 
  
   
    def _estimate_means(self, nk, xk):  
        self.mean_precision_ = self.mean_precision_prior_ + nk
        self.means_ = ((self.mean_precision_prior_ * self.mean_prior_ +nk[:, np.newaxis] * xk) / self.mean_precision_[:, np.newaxis])
  
  
    def _estimate_precisions(self, nk, xk, sk): 
        self._estimate_wishart_full(nk,xk,sk)
        self.precisions_cholesky_ = _compute_precision_cholesky(self.covariances_) 
      
    
    def _initialize_params(self,features_resp):
        #Initialize the weights,means,covariances
        nk, xk, sk = _estimate_gaussian_parameters(features_resp,self.n_samples,self.n_features,self.n_components,self.reg_covar)  
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk) 
        

    
    def _estimate_log_weights(self):
        digamma_sum = digamma(self.weight_concentration_[0] + self.weight_concentration_[1])
        digamma_a = digamma(self.weight_concentration_[0])
        digamma_b = digamma(self.weight_concentration_[1])
        return (digamma_a - digamma_sum + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
            
    
    def _compute_log_responsibility(self,features_resp,log_det,log_weights,sum_log_prob_norm):
        means=self.means_
        precisions_chol=self.precisions_cholesky_
        n_features=self.n_features
        degrees_of_freedom=self.degrees_of_freedom_
        mean_precision=self.mean_precision_
        
        def parse(iterator):
          arr=np.array(list(iterator),ndmin=2)          
          log_prob=np.empty((arr.shape[0],means.shape[0]))      
          
          for k, (mu, prec_chol) in enumerate(zip(means,precisions_chol)):          
              y = np.dot(arr[:,0:n_features], prec_chol) - np.dot(mu, prec_chol)
              log_prob[:, k] = np.sum(np.square(y), axis=1)
              
          log_prob=-.5 * (n_features* np.log(2 * np.pi) + log_prob) + log_det 
          log_prob = log_prob - .5 * n_features * np.log(degrees_of_freedom)
          log_lambda = n_features * np.log(2.) + np.sum(digamma(.5 * (degrees_of_freedom -np.arange(0, n_features)[:, np.newaxis])), 0)
          weighted_log_prob=log_prob + .5 * (log_lambda - n_features / mean_precision) + log_weights

          log_sum_exp=logsumexp(weighted_log_prob,axis=1)
          sum_log_prob_norm.add(np.sum(log_sum_exp))
          with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_sum_exp[:, None]            
          #Update responsibility      
          arr[:,n_features:] = log_resp
          return arr
        
        
        result=features_resp.mapPartitions(parse)

        return result,sum_log_prob_norm
      
       
    
    def _m_step(self,features_resp):
        n_features=self.n_features
        def exp(iterator):
          arr=np.array(list(iterator))    
          arr[:,n_features:]=np.exp(arr[:,n_features:])
          return arr 
        nk, xk, sk = _estimate_gaussian_parameters(features_resp.mapPartitions(exp),self.n_samples,self.n_features,self.n_components,self.reg_covar)  
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)