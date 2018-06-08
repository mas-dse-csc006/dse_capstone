import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
import platform
import struct
import numpy as np
import numpy
import random
import tensorflow as tf
import os
import pickle as pickle
import matplotlib.pyplot as plt
import sys
from datetime import datetime
import time
from pathlib import Path

def format_time(t):
    return t.strftime("%Y-%m-%d %H:%M:%S")

#this function helps to visualize the dict
from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
# take(1, prod_desc.values())

# load data after creating features
def load_data_hybrid(data_path, min_items=2, min_users=2):
    user_ratings = defaultdict(set)
    item_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    user_count = 0
    item_count = 0
    reviews = 0
    users = {}  # aid to id LUT
    items = {}  # asid to id LUT
    records = {} # all records
    features = {}
    random.seed(0)
    columns = None
    offset_to_features = 3
    with open(data_path, 'r') as f:
        bad_actor = 0
        for line in f.readlines():
            record = {}
            split_line = line.split('\t')
            if columns is None:
                columns = [e.rstrip() for e in split_line]
                continue
            #if (sampling and random.random()>sample_size):
            #   continue
            reviews += 1
            
            if (len(split_line) != len(columns)):
                raise Exception ("Line %d isn't aligned. Found %d values on line. In %d column file" 
                                 % (reviews, len(split_line), len(columns)))
                #bad_actor = bad_actor + 1
                #continue
            else:
                auid, asid, _ = split_line[0:offset_to_features]
                record = {columns[i]:split_line[i].rstrip() for i in  range (offset_to_features, len(split_line))}

            u, i = None, None

            if auid in users:
                u = users[auid]
            else:
                user_count += 1  # new user so increment
                users[auid] = user_count
                u = user_count
            
            if asid in items:
                i = items[asid]
            else:
                item_count += 1  # new i so increment
                items[asid] = item_count
                i = item_count
                
                if 'cluster' in record:
                    record['cluster'] = float(record['cluster'])
                
                for c in ['price_delta_calc1','price_delta_calc2','price_delta_l4avg']:
                    if c in record:
                        record[c] = float(record[c])
                if 'price' in record:
                    if record['price'] == '':
                        record['price'] = 0
                    else:
                        record['price'] = float(record['price'])
                if 'level4_average' in record:
                    if record['level4_average'] == '':
                            record['level4_average'] = 0
                    else:
                        record['level4_average'] = float(record['level4_average'])
                        # Price ratio feature added
                        record['level4_ratio_price']= record['price']/record['level4_average']
                if 'polarity' in record:
                    record['polarity']= round((float(record['polarity'])),2)
                    
                if 'feature_vector' in record:
                    if len(record['feature_vector']) == 0:
                        record['feature_vector'] = list(np.zeros(4524))
                    else:
                        record['feature_vector'] = [int(el) for el in list(record['feature_vector'])[:-1][1:]]
    
                for c in ['top_categories','rating','percentile_hotcoded','season','level4','sentiment']:
                    if c in record:
                        record[c] = [int(el) for el in list(record[c])[:-2][1:]]
                records[i] = record
            
            user_ratings[u].add(i)
            item_ratings[i].add(u)
            max_u_id = max(u, max_u_id)
            max_i_id = max(i, max_i_id)
            
    print ("max_u_id: ", max_u_id)
    print ("max_i_id: ", max_i_id)
    print ("reviews : ", reviews)


    # filter out users w/ less than X reviews
    num_u_id = 0
    num_i_id = 0
    num_reviews = 0
    user_ratings_filtered = defaultdict(set)
    for u, ids in user_ratings.items():
        if len(ids) > min_items:
            user_ratings_filtered[u] = ids
            num_u_id += 1
            num_reviews += len(ids)
            
    item_ratings_filtered = defaultdict(set)
    for ids, u in item_ratings.items():
        if len(u) > min_users:
            # keep
            item_ratings_filtered[ids] = u
            num_i_id += 1
    
    feature_keys = records[1].keys() #should be same as columns[offset:]
    features = {k:{i:records[i][k] for i in range(1,len(records)+1)} for k in feature_keys}

    print ("u_id: ", num_u_id)
    print ("i_id: ", num_i_id)
    print ("reviews : ", num_reviews)
    #return max_u_id, max_i_id, users, items, user_ratings_filtered,\
    #            item_ratings_filtered, brands, prices, prod_desc, prod_cat,price_feature,season_feature
    return max_u_id, max_i_id, users, items, user_ratings_filtered,item_ratings_filtered, features

#load image features for the given asin collection into dictionary
def load_image_features(path, items):
    count=0
    image_features = {}
    f = open(path, 'rb')
    while True:
        asin = f.read(10)
        if asin == '': break
        features_bytes = f.read(16384) # 4 * 4096 = 16KB, fast read, don't unpack
  
        if asin in items: #only unpack 4096 bytes if w need it -- big speed up
            features = (np.fromstring(features_bytes, dtype=np.float32)/58.388599)
            iid=items[asin]
            if len(features)==0:
                image_features[iid] = np.zeros(4096)
            else:
                image_features[iid] = features
    
    return image_features

def uniform_sample_batch(train_ratings, test_ratings, item_count, advanced_features):
    neg_items = 2
    for u in train_ratings.keys():
        t = []
        iv = []
        jv = []
        for i in train_ratings[u]:
            if (u in test_ratings.keys()):
                if (i != test_ratings[u]):  # make sure it's not in the test set
                    for k in range(1,neg_items):
                        j = random.randint(1, item_count)
                        while j in train_ratings[u]:
                            j = random.randint(1, item_count)
                        # sometimes there will not be an image for given product
                        try:
                            advanced_features[i]
                            advanced_features[j]
                        except KeyError:
                            continue
                        iv.append(advanced_features[i])
                        jv.append(advanced_features[j])
                        t.append([u, i, j])
            else:
                for k in range(1,neg_items):
                    j = random.randint(1, item_count)
                    while j in train_ratings[u]:
                        j = random.randint(1, item_count)
                    # sometimes there will not be an image for given product
                    try:
                        advanced_features[i]
                        advanced_features[j]
                    except KeyError:
                        continue
                    iv.append(advanced_features[i])
                    jv.append(advanced_features[j])
                    t.append([u, i, j])

        # block if queue is full
        if len(iv)>1:
            yield numpy.asarray(t), numpy.vstack(tuple(iv)), numpy.vstack(tuple(jv))
        else:
            continue

def test_batch_generator_by_user(train_ratings, test_ratings, item_ratings, item_count, advanced_features, cold_start = False, cold_start_thresh=10):
    # using leave one cv
    for u in random.sample(test_ratings.keys(), min(len(test_ratings),4000)):
    #for u in test_ratings.keys():
        i = test_ratings[u]
        if (cold_start and len(item_ratings[i]) > cold_start_thresh-1):
            continue
        t = []
        ilist = []
        jlist = []
        count = 0
        for j in random.sample(range(item_count), 100):
            # find item not in test[u] and train[u]
            if j != test_ratings[u] and not (j in train_ratings[u]):
                try:
                    advanced_features[i]
                    advanced_features[j]
                except KeyError:
                    continue

                count += 1
                t.append([u, i, j])
                ilist.append(advanced_features[i])
                jlist.append(advanced_features[j])

        # print numpy.asarray(t).shape
        # print numpy.vstack(tuple(ilist)).shape
        # print numpy.vstack(tuple(jlist)).shape
        if (len(ilist) == 0):
            #print "could not find neg item for user, count: ", count, u
            continue
        yield numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))

def generate_test(user_ratings):
    '''
    for each user, random select one rating into test set
    '''
    user_test = dict()
    for u, i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    return user_test

def price_transform(value, levels, price_max):
    prices_vec = np.zeros(levels + 1, dtype = int)
    idx = int(numpy.ceil(float(value)/(price_max/levels)))
    prices_vec[idx] = 1
    level = prices_vec[1::]
    #level_return = ''.join(str(e) for e in level)
    #level_return = ''.join(str(e) for e in level)
    return level 

"""
    l4_avg_adjusted_min = df['l4_avg_adjusted'].min()
    df['l4_avg_adjusted_log'] = df['l4_avg_adjusted'].apply(lambda x: \
                                                                 np.log(x-l4_avg_adjusted_min+1))
    l4_avg_adjusted_max = df['l4_avg_adjusted'].max()
    df['l4_avg_adjusted_1hotencoded'] = df['l4_avg_adjusted'].apply(lambda x: \
                                                                  price_transform(x,20, l4_avg_adjusted_max)) 
    l4_avg_adjusted_log_max = df['l4_avg_adjusted_log'].max()
    df['l4_avg_adjusted_log_1hotencoded'] = df['l4_avg_adjusted_log'].apply(lambda x: \
                                                                  price_transform(x,6, l4_avg_adjusted_log_max))
"""    
#user_count, item_count, users, items, user_ratings, item_ratings, brands, prices, prod_desc = load_data_hybrid(data_path, min_items=4, min_users=0, sampling= True, sample_size = 0.8)
def transform_features (features):
    
    #
    # Code below could be cleaned up into functions
    #
    
    # Level 4 Ratio Price
    
    level4_ratio_price = features['level4_ratio_price']
    level4_ratio_price_max = max(level4_ratio_price.values())
    level4_ratio_price_min = min(level4_ratio_price.values())
    print("l4 ratio price max", level4_ratio_price_max)
    features['level4_ratio_price_1hotencoded'] = {k:price_transform(v,26,level4_ratio_price_max) 
                                            for k, v in level4_ratio_price.items()}
    
    
    # Level 4 log ratio price log transformed
    level4_ratio_price_log = {k:np.log(1+v) for k, v in level4_ratio_price.items()}
    level4_ratio_price_log_max = max(level4_ratio_price_log.values())
    level4_ratio_price_log_1hotencoded = {k:price_transform(v,12,level4_ratio_price_log_max) 
                             for k, v in level4_ratio_price_log.items()}
    features['level4_ratio_price_log_1hotencoded'] = level4_ratio_price_log_1hotencoded
    
    
    
    # Level 4 Average
    level4_average = features['level4_average']
    level4_average_max = max(level4_average.values())
    level4_average_min = min(level4_average.values())
    features['level4_average_1hotencoded'] = {k:price_transform(v,20,level4_average_max) 
                                            for k, v in level4_average.items()}
    # Level 4 Average log transformed
    level4_average_log = {k:np.log(1+v) for k, v in level4_average.items()}
    level4_average_log_max = max(level4_average_log.values())
    level4_average_log_1hotencoded = {k:price_transform(v,10,level4_average_log_max) 
                             for k, v in level4_average_log.items()}
    features['level4_average_log_1hotencoded'] = level4_average_log_1hotencoded
    
    # Standard Price 1-Hot Encoding
    price = features['price']
    price_max = max(price.values())
    price_min = min(price.values())
    features['price_1hotencoded'] = {k:price_transform(v,20,price_max) 
                                            for k, v in price.items()}
    # Price log transformed
    price_log = {k:np.log(1+v) for k, v in price.items()}
    price_log_max = max(price_log.values())
    price_log_1hotencoded = {k:price_transform(v,10,price_log_max) 
                             for k, v in price_log.items()}
    features['price_log_1hotencoded'] = price_log_1hotencoded

#     # This is the product's price - the average product price in the level 4 subcategory
#     l4_avg_adjusted = features['price_delta_l4avg']
#     l4_avg_adjusted_max = max(l4_avg_adjusted.values())
#     l4_avg_adjusted_min = min(l4_avg_adjusted.values())
#     l4_avg_adjusted_1hotencoded = {k:price_transform(v,30,l4_avg_adjusted_max) 
#                                        for k, v in l4_avg_adjusted.items()}
#     features['l4_avg_adjusted_1hotencoded'] = l4_avg_adjusted_1hotencoded

#     # Adjusted price log transformed
#     l4_avg_adjusted_log = {k:np.log(1.1-l4_avg_adjusted_min+v) 
#                            for k, v in l4_avg_adjusted.items()}
#     l4_avg_adjusted_log_max = max(l4_avg_adjusted_log.values())
#     l4_avg_adjusted_log_1hotencoded = {k:price_transform(v,30,l4_avg_adjusted_log_max) 
#                                        for k, v in l4_avg_adjusted_log.items()}
#     features['l4_avg_adjusted_log_1hotencoded'] = l4_avg_adjusted_log_1hotencoded
    
    # below could be modified as the hotencode item is the length of the number of items; 
    # it could be shortened to the unique number of brands
    if 'brand' in features:
        brands_features = {}
        brands = features['brand']
        brands_all = list(set(brands.values()))
        for key, value in brands.items():
            brands_vec = numpy.zeros(len(brands_all))
            brands_vec[brands_all.index(value)] = 1
            brands_features[key] = brands_vec
        features['brand_1hotencoded'] = brands_features
        
    if 'cluster' in features:
        cluster_features = {}
        cluster = features['cluster']
        cluster_all = sorted( list(set(cluster.values())) )
        for key, value in cluster.items():
            cluster_vec = numpy.zeros(len(cluster_all))
            cluster_vec[int(v)] = 1
            cluster_features[key] = cluster_vec
        features['cluster_1hotencoded'] = cluster_features
        
    return features

# list of features defined as dicts can be passed and they are combined, if none array of zeros are created

def feature_set(feature_dicts=None,item_count=None):
    #print(type(feature_dicts))
    #print(str(feature_dicts))
    if feature_dicts==None:
        return {n: [0] for n in range(1,item_count+1)} #return just zeros dummy advanced features for baseline BPR
    else:
        combined_features = defaultdict(list)
        for d in feature_dicts:
            for k, v in d.items():  
                combined_features[k].extend(v)

        return dict([(k,v) for k,v in combined_features.items()])


def abpr(user_count, item_count, advanced_features, hidden_dim=10, hidden_adv_dim=10,
         l2_regulization=0.1,
         bias_regulization=0.01,
         embed_regulization = 0,
         adv_feature_regulization =0.1,
         adv_feature_bias_regulization = 0.01):
    """
    user_count: total number of users
    item_count: total number of items
    hidden_dim: hidden feature size of MF
    hidden_adv_dim: hidden visual/non-visual feature size of MF
    P.S. advanced_features can be one or many features combined. it can only be image features, non-image features, or both
    """
    advanced_feat_dim = len(advanced_features[1])
    iv = tf.placeholder(tf.float32, [None, advanced_feat_dim])
    jv = tf.placeholder(tf.float32, [None, advanced_feat_dim])
    u = tf.placeholder(tf.int32, [None])
    i = tf.placeholder(tf.int32, [None])
    j = tf.placeholder(tf.int32, [None])
    
    # model parameters -- LEARN THESE
    # latent factors
    user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    # biases
    item_b = tf.get_variable("item_b", [item_count + 1, 1], initializer=tf.constant_initializer(0.0))

    # pull out the respective latent factor vectors for a given user u and items i & j
    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    
    # get the respective biases for items i & j
    i_b = tf.nn.embedding_lookup(item_b, i)
    j_b = tf.nn.embedding_lookup(item_b, j)


    # MF predict: u_i > u_j
   
    # UxD Advanced feature latent factors for users
    user_adv_w = tf.get_variable("user_adv_w", [user_count + 1, hidden_adv_dim],
                             initializer=tf.random_normal_initializer(0, 0.1))
    # this is E, the embedding matrix
    item_adv_w = tf.get_variable("item_adv_w", [hidden_adv_dim, advanced_feat_dim],
                            initializer=tf.random_normal_initializer(0, 0.1))

    theta_i = tf.matmul(iv, item_adv_w, transpose_b=True)  # (f_i * E), eq. 3
    theta_j = tf.matmul(jv, item_adv_w, transpose_b=True)  # (f_j * E), eq. 3

    adv_feature_bias = tf.get_variable("adv_feature_bias", [1, advanced_feat_dim], initializer=tf.random_normal_initializer(0, 0.1))
    # pull out the visual factor, 1 X D for user u

    u_img = tf.nn.embedding_lookup(user_adv_w, u)

    xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_i), 1, keep_dims=True) \
                                                                        + tf.reduce_sum(tf.multiply(adv_feature_bias, iv), 1, keep_dims=True) 
    xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_j), 1, keep_dims=True) \
                                                                        + tf.reduce_sum(tf.multiply(adv_feature_bias, jv), 1, keep_dims=True) 
    l2_norm = tf.add_n([
        l2_regulization * tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        adv_feature_regulization * tf.reduce_sum(tf.multiply(u_img, u_img)),
        l2_regulization * tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        l2_regulization * tf.reduce_sum(tf.multiply(j_emb, j_emb)),
        embed_regulization * tf.reduce_sum(tf.multiply(item_adv_w, item_adv_w)),
        bias_regulization * tf.reduce_sum(tf.multiply(i_b, i_b)),
        bias_regulization * tf.reduce_sum(tf.multiply(j_b, j_b)),
        adv_feature_bias_regulization * tf.reduce_sum(tf.multiply(adv_feature_bias, adv_feature_bias))
    ])
        
    xuij = xui - xuj

    auc = tf.reduce_mean(tf.to_float(xuij > 0))
    
    loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
    
    train_op = tf.train.AdamOptimizer().minimize(loss)
    
    return xuij,u, i, j, iv, jv, loss, auc, train_op

def session_run(num_iter, user_count, item_count, users, items, 
                user_ratings, item_ratings, advanced_features,hidden_dim=10,hidden_adv_dim=10,cold_start_thresh=10):
    ### Loading and parsing the review matrix for Women 5-core dataset
    auc_train = []
    auc_test = []
    auc_test_cs = []
    #data_path = os.path.join('/Users/nolanthomas/Public/amazon', 'out_topcategories_pricepercentile_seasonmeteorological.csv')
    #user_count, item_count, users, items, user_ratings, item_ratings, brands, features = load_data_hybrid(data_path, min_items=4, min_users=0, sampling= True, sample_size = 0.8)
    user_ratings_test = generate_test(user_ratings)
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope('abpr'):
            xuij,u, i, j, iv, jv, loss, auc, train_op = abpr(user_count, item_count, advanced_features,hidden_dim,hidden_adv_dim)

        session.run(tf.global_variables_initializer())
        

        for epoch in range(1, num_iter+1):
            print ("epoch ", epoch)
            _loss_train = 0.0
            user_count = 0
            auc_train_values = []
            for d, _iv, _jv in uniform_sample_batch(user_ratings, user_ratings_test, item_count, advanced_features):
                user_count += 1
                _loss, _auc, _ = session.run([loss, auc, train_op], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
                _loss_train += _loss
                auc_train_values.append(_auc)
            print ("train_loss:", _loss_train/user_count, "train auc: ", numpy.mean(auc_train_values))
            auc_train.append(numpy.mean(auc_train_values))

            user_items_test=[]
            auc_values = []
            _loss_test = 0.0
            user_count = 0
            for d, _iv, _jv in test_batch_generator_by_user(user_ratings, user_ratings_test, item_ratings, item_count, advanced_features, cold_start = False):
                user_count += 1
                _loss, _auc = session.run([loss, auc], feed_dict={u: d[:, 0], i: d[:, 1], j: d[:, 2], iv: _iv, jv: _jv})
                _loss_test += _loss
                auc_values.append(_auc)
            print ("test_loss: ", _loss_test / user_count, "test auc: ", numpy.mean(auc_values))
            auc_test.append(numpy.mean(auc_values))

            auc_values_cs = []
            _loss_test_cs = 0.0
            user_count = 0
            for d, _iv, _jv in test_batch_generator_by_user(user_ratings, user_ratings_test, item_ratings, item_count, advanced_features, cold_start=True,cold_start_thresh=cold_start_thresh):
                user_count += 1
                _xuij,_loss, _auc = session.run([xuij,loss, auc], feed_dict={u: d[:, 0], i: d[:, 1], j: d[:, 2], iv: _iv, jv: _jv})
                _loss_test_cs += _loss
                auc_values_cs.append(_auc)
                if epoch==num_iter:
                    user_items_test.append((d,_xuij))
            print ("cold start test_loss: ", _loss_test_cs / user_count, "cold start auc: ", numpy.mean(auc_values_cs))
            auc_test_cs.append(numpy.mean(auc_values_cs))
        return user_items_test,auc_train, auc_test, auc_test_cs

def run(num_sessions, user_count, item_count, users, items, 
            user_ratings, item_ratings, advanced_features,hidden_dim=10,hidden_adv_dim=10,cold_start_thresh=10):
    t1 = datetime.now()
    user_items_test, auc_train, auc_test, auc_test_cold = session_run(num_sessions, user_count, item_count, 
                                                     users, items, user_ratings, item_ratings, 
                                                     advanced_features, hidden_dim=10, hidden_adv_dim=10,cold_start_thresh=10)
    t2 = datetime.now()
    return {'num_sessions':num_sessions, 
            'num_features':len(advanced_features),
            'hidden_dim':hidden_dim,
            'hidden_adv_dim':hidden_adv_dim,
            'cold_start_thresh':cold_start_thresh,
            'user_count':user_count,
            'item_count':item_count,
            'sys.platform':str(sys.platform), 
            'platform.processor':str(platform.processor()), 
            'sys.version':str(sys.version), 
            'user_items_test': user_items_test,
            'auc_train': auc_train, 'auc_test': auc_test, 
            'auc_cold_test': auc_test_cold,
            'start':format_time(t1),'end':format_time(t1),
            'delta_sec':(t2-t1).total_seconds()}

import matplotlib as mpl
import seaborn as sns
mpl.style.use('seaborn')


def plot_auc_curve(results_to_graph, title, highlight):
    dt_str = datetime.now().strftime("%Y%m%d.%H%M")
    sns.set_context("poster")
    sns.set_palette("cubehelix",8)
    plt.figure(figsize=(20,10))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(title,fontsize=30)
    plt.xticks(range(0,20))
    for calc_desc, calc_results in results_to_graph.items():
        ls='solid'
        if calc_desc == 'BPR':
            ls='dashed'
        if calc_desc == highlight:
            ls='-.'
        plt.plot(calc_results['auc_test'],
                 label=calc_desc,
                 linestyle=ls,
                 marker='o')
        plt.annotate(xy=[calc_results['num_sessions']-1,calc_results['auc_test'][-1]],
                     s=str(round(calc_results['auc_test'][-1],4)), #+ ' ' + calc_desc, 
                     fontsize=15,
                     textcoords='offset points')
    plt.legend()
    plt.ylabel("Test AUC",fontsize=20)
    plt.xlabel("Number of Iterations",fontsize=20)
    #savefig('auc_curve.' + dt_str + '.png')
    #show()

def plot_auc_cold_start_curve(results_to_graph, title,highlight): 
    dt_str = datetime.now().strftime("%Y%m%d.%H%M")
    sns.set_context("poster")
    sns.set_palette("cubehelix",8)
    plt.figure(figsize=(20,10))
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.title(title,fontsize=30)
    plt.xticks(range(0,20))
    for calc_desc, calc_results in results_to_graph.items():
        ls='solid'
        if calc_desc == 'BPR':
            ls='dashed'
        if calc_desc == highlight:
            ls='-.'
        plt.plot(calc_results['auc_cold_test'],
            label=calc_desc,
            linestyle=ls,
            marker='o')
        plt.annotate(xy=[calc_results['num_sessions']-1,calc_results['auc_cold_test'][-1]],
             s=str(round(calc_results['auc_cold_test'][-1],4)), #+ ' ' + calc_desc, 
             fontsize=15,
             textcoords='offset points')
    plt.legend()
    plt.ylabel("Test AUC",fontsize=20)
    plt.xlabel("Number of Iterations",fontsize=20)
    #plt.savefig('auc_cold_start_curve.' + dt_str + '.png')
    #plt.show()
