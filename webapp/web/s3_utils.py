import numpy as np
import json

class S3Utils(object):
  def __init__(self):
    super(S3Utils, self).__init__()
  
  #should get file_name from database of models
  #class method
  @classmethod
  def fetch_model_params(cls, model):
    gen = model.gen
    s3_fn = model.url
    #s3.get(s3_fn) #store in cache on disk parse
    #hack, just load this from disk for now!
    if gen=="snado":
      return S3Utils.load_snado(s3_fn)
    elif gen=="legacy":
      
      nItems = 23033
      nUsers = 39387
      K=10
      item_bias, user_factors, item_factors = S3Utils.load_legacy(s3_fn, nItems, nUsers, K)
      return item_bias, user_factors, item_factors
    else:
      dump(gen)
      
    
    
  @classmethod
  def load_snado(cls, s3_fn):
    #ideally this would pull it from s3
    fn="tmp/%s"%s3_fn
    coeffs = np.load(fn)
    item_bias = coeffs['item_bias']
    user_factors = coeffs['user_factors']
    item_factors = coeffs['item_factors']
    
    return item_bias, user_factors, item_factors
    
  @classmethod
  def load_legacy(cls, path, nItems, nUsers, K):
    fn="tmp/%s"%path 
    item_bias = np.zeros(nItems)
    user_factors = np.zeros((nUsers, K))
    item_factors = np.zeros((nItems, K))
  
    idx=0
    with open(fn, 'rb') as param_file:
      dump = json.load(param_file)
      nw=dump["NW"] 
      w=np.array(dump["W"])
  
      item_bias = w[idx:nItems]
      idx+=nItems
  
      for u in range(nUsers):
        user_factors[u]=w[idx:idx+K]
        idx+=K
    
      for i in range(nItems):
        item_factors[i]=w[idx:idx+K]
        idx+=K
    
      return item_bias, user_factors, item_factors
    