import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_test_users = None
_test_items = None
_testNegatives = None
_K = None

def evaluate_model(model, test_users, test_items, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _test_users
    global _test_items
    global _testNegatives
    global _K
    _model = model
    _test_users = test_users
    _test_items = test_items
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [],[]
    if(num_thread > 1): # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_test_items)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_test_items)):
        (hr,ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)

def eval_one_rating(idx):

    items = _testNegatives[idx]
    u = _test_users[idx]
    gtItem = _test_items[idx]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = []
    for _ in range(len(items)):
        users.append(u)
    predictions = _model.predict([np.array(users), np.array(items)], 
                                   batch_size=100, verbose=0)
    for i in range(len(items)):
        map_item_score[i] = predictions[i]
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, (len(items)-1))
    ndcg = getNDCG(ranklist, (len(items)-1))
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
