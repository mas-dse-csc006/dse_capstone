from bpr_model import *
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int,
                    help="k is the num epoch")
parser.add_argument("--file",
                    help="data file; must be in the projects home data/processed directory")
parser.add_argument("--home",
                    help="projects home directory")
parser.add_argument("--min_users", type=int,
                    help="min user relationship")
parser.add_argument("--min_items", type=int,
                    help="min item relationship")
parser.add_argument("--desc", 
                    help="description about run")
args = parser.parse_args()
num_sessions = args.k
file_name = args.file
home = args.home
min_users = args.min_users
min_items = args.min_items
run_description = args.desc
print("home directory:", home)
print("k:", num_sessions)
print("file:", file_name)
print("desc:", run_description)
print("min_users:", min_users)
print("min_items:", min_items)

#home = str(Path.home())
bpr_data_file = os.path.join(home, 'data', 'processed', file_name)
print (bpr_data_file)
user_count, item_count, users, items, user_ratings, item_ratings, loaded_features   = load_data_hybrid(bpr_data_file, 
                                                                                                       min_items=min_items, 
                                                                                                       min_users=min_users)
print("Loaded")
loaded_features = transform_features(loaded_features)
print("Features have been transformed")
results = {}
single_features  = {
     'BPR':None,
     'Category-Tags':['top_categories'],
     'Subcat-L4':['level4'],
     'Brand':['brand_1hotencoded'],
     'Price-Ratio-26bin':['level4_ratio_price_1hotencoded'],
     'Price-Ratio-Log-12bin':['level4_ratio_price_log_1hotencoded'],
     'Standard-Price-20bin':['price_1hotencoded'],
     'Standard-Price-Log-10bin':['price_log_1hotencoded'],
     'L4-Avg-20bin':['level4_average_1hotencoded'],
     'L4-Avg-Log-10bin':['level4_average_log_1hotencoded']
}

print('running models')

print('Single features',single_features)

variants = single_features
dt = datetime.now().strftime("%Y%m%d.%H%M")
fname = run_description + '_k' + str(num_sessions) + '_minitems' + str(min_items) + '_minusers' + str(min_users) + "_results.singlefeatures." + dt + ".pickle"   
for desc, features_to_use in variants.items():
    print (desc + str(features_to_use))
    print (datetime.now().strftime("%Y%m%d.%H%M"))
    if features_to_use != None:
        features_list = feature_set([loaded_features[c] for c in features_to_use])
    else:
        features_list = feature_set(None,item_count)
    results[desc] = run(num_sessions, user_count, item_count, 
                         users, items, user_ratings, 
                         item_ratings,features_list,hidden_dim=10,hidden_adv_dim=10,cold_start_thresh=10)
    results[desc]['file_name'] = file_name
    results[desc]['desc'] = run_description
    pickle.dump( results, open( fname, "wb" ) ) 

result_combos = {}
combos = {}
"""
combos  = {
    'Category-Tags, Subcat-L4':['top_categories','level4','price_log_1hotencoded'],
    'Category-Tags, Subcat-L4, Brand':['top_categories', 'level4', 'brand_1hotencoded']
    #'Price-Log':['price_log_1hotencoded'],
    #'Subcat-L4, Price-Log':['level4','price_log_1hotencoded'],
    #'Category-Tags, Price-Log':['top_categories','price_log_1hotencoded'],
}
"""

print ('Combinations', combos)
variants = combos
dt = datetime.now().strftime("%Y%m%d.%H%M")
fname = desc + '_k' + str(num_sessions) + '_minitems' + str(min_items) + '_minusers' + str(min_users) + "_results.combinations." + dt + ".pickle"   
for desc, features_to_use in variants.items():
    print (desc + str(features_to_use))
    print (datetime.now().strftime("%Y%m%d.%H%M"))
    if features_to_use != None:
        features_list = feature_set([loaded_features[c] for c in features_to_use])
    else:
        features_list = feature_set(None,item_count)
    result_combos[desc] = run(num_sessions, user_count, item_count, 
                         users, items, user_ratings, 
                         item_ratings,features_list,hidden_dim=10,hidden_adv_dim=10,cold_start_thresh=10)
    results_combos[desc]['file_name'] = file_name
    results_combos[desc]['desc'] = run_description
    pickle.dump( result_combos, open( fname, "wb" ) ) 
