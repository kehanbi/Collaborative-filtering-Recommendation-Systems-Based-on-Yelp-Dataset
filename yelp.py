'''
Method Description:
I create Model-based recommendation system in this project. I use XGBregressor to train the model. 
1. The way I improved the accuracy is considering more valuable features in business.json and user.json.
    The features I use in this project are:
        features I used in HW3 task 2.3: 'business_stars', 'review_count_business', 'state', 'city'
        I add more features in this project: 
        business.json: 'BusinessAcceptsCreditCards', 'GoodForKids', 'RestaurantsDelivery', 
            'RestaurantsGoodForGroups', 'RestaurantsReservations', 'RestaurantsTakeOut', 'WiFi', 'NoiseLevel',
            'star_diff': this is "business_stars" - the average stars of business,
        user.json: 'review_count_user', 'average_stars', 'yelping_since','useful', 'funny', 'cool', 'fans', 
            'user_star_diff': this is 'average_stars' - the average value of 'average_stars'
    After adding these features in my model, the rmse decreasing by 0.984-0.9802 = 0.004.
2. I adjust the parameters in xgboost model:
    I adding these parameters on my model:
    min_child_weight: change from 1 to 1.2, the rmse decrease about 0.0002-0.0003
    n_estimators: change from 300 to 800, the rmse obvious decrease
    and gamma = 0.1, subsample = 0.8, colsample_bytree = 0.8, booster = 'gbtree', objective='reg:linear'

Error Distribution:
>=0 and <1: 27
>=1 and <2: 6777
>=2 and <3: 481
>=3 and <4: 50897
>=4: 43666

RMSE:
0.9784688137839643

Execution Time:
172.49129271507263 s
'''

import xgboost as xgb
import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")
from pyspark import SparkContext, SparkConf
from operator import add
import sys
from time import *
import math
from itertools import combinations
from collections import defaultdict
import random
import csv
from sklearn.metrics import mean_squared_error

conf = SparkConf().setAppName("DSCI_553_HW3").setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

start_time = time()
#folder_path = sys.argv[1]
#test_file_name = sys.argv[2]
#output_file_name = sys.argv[3]

folder_path = "./HW3StudentData"
test_file_name = "./HW3StudentData/yelp_val.csv"
output_file_name = "competition_output.csv"

train_file_name = folder_path + "/yelp_train.csv"
business_feature_path = folder_path + "/business.json"
user_feature_path = folder_path + "/user.json"

#--------------------------- select features from business.json----------------------------------
train_df = pd.read_csv(train_file_name, sep = ",")
#print(train_df)
business_feature = sc.textFile(business_feature_path).map(json.loads).map(lambda x: (x['business_id'], x['stars'], x['review_count'], x['attributes'], x['state'], x['city'])).collect()
bf_list = []
for i in business_feature:
    bf_list.append(list(i))
business_feature_df = pd.DataFrame(bf_list, columns = ['business_id', 'business_stars', 'review_count_business', 'attributes', 'state', 'city'])
business_feature_df['attributes'] = business_feature_df['attributes'].fillna(0)
#business_feature_df['state'] = business_feature_df['state'].astype(int)
#business_feature_df['city'] = business_feature_df['city'].astype(int)
#business_feature_df['state']= pd.to_numeric(business_feature_df['state'], errors='coerce')
#business_feature_df = business_feature_df.apply(pd.to_numeric)
#print(business_feature_df)
#print (business_feature_df.isnull().sum())


state_cate_list = list(set(business_feature_df['state']))
new_state = []
for i in range(len(business_feature_df['state'])):
    new_state.append(state_cate_list.index(business_feature_df['state'][i]))
business_feature_df['state']  = pd.DataFrame(new_state) 
#cate_create = pd.Categorial(cate_list)
#business_feature_df['state'] = business_feature_df['state'].astype('category')
#business_feature_df['city'] = business_feature_df['city'].astype('category')
#business_feature_df['city']
city_cate_list = list(set(business_feature_df['city']))
new_city = []
for i in range(len(business_feature_df['city'])):
    new_city.append(city_cate_list.index(business_feature_df['city'][i]))
business_feature_df['city']  = pd.DataFrame(new_city) 


Bus_acc_credit = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        Bus_acc_credit.append(-1)
    elif "BusinessAcceptsCreditCards" in i.keys():
        if i["BusinessAcceptsCreditCards"] == 'True':
            Bus_acc_credit.append(1)
        else:
            Bus_acc_credit.append(0)
    else:
        Bus_acc_credit.append(-1)
business_feature_df['BusinessAcceptsCreditCards'] = pd.DataFrame(Bus_acc_credit)


GoodForKids = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        GoodForKids.append(-1)
    elif "GoodForKids" in i.keys():
        if i["GoodForKids"] == 'True':
            GoodForKids.append(1)
        else:
            GoodForKids.append(0)
    else:
        GoodForKids.append(-1)
business_feature_df['GoodForKids'] = pd.DataFrame(GoodForKids)


RestaurantsDelivery = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        RestaurantsDelivery.append(-1)
    elif "RestaurantsDelivery" in i.keys():
        if i["RestaurantsDelivery"] == 'True':
            RestaurantsDelivery.append(1)
        else:
            RestaurantsDelivery.append(0)
    else:
        RestaurantsDelivery.append(-1)
business_feature_df['RestaurantsDelivery'] = pd.DataFrame(RestaurantsDelivery)



RestaurantsGoodForGroups = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        RestaurantsGoodForGroups.append(-1)
    elif "RestaurantsGoodForGroups" in i.keys():
        if i["RestaurantsGoodForGroups"] == 'True':
            RestaurantsGoodForGroups.append(1)
        else:
            RestaurantsGoodForGroups.append(0)
    else:
        RestaurantsGoodForGroups.append(-1)
business_feature_df['RestaurantsGoodForGroups'] = pd.DataFrame(RestaurantsGoodForGroups)



RestaurantsReservations = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        RestaurantsReservations.append(-1)
    elif "RestaurantsReservations" in i.keys():
        if i["RestaurantsReservations"] == 'True':
            RestaurantsReservations.append(1)
        else:
            RestaurantsReservations.append(0)
    else:
        RestaurantsReservations.append(-1)
business_feature_df['RestaurantsReservations'] = pd.DataFrame(RestaurantsReservations)


RestaurantsTakeOut = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        RestaurantsTakeOut.append(-1)
    elif "RestaurantsTakeOut" in i.keys():
        if i["RestaurantsTakeOut"] == 'True':
            RestaurantsTakeOut.append(1)
        else:
            RestaurantsTakeOut.append(0)
    else:
        RestaurantsTakeOut.append(-1)
business_feature_df['RestaurantsTakeOut'] = pd.DataFrame(RestaurantsTakeOut)


WiFi = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        WiFi.append(0)
    elif "WiFi" in i.keys():
        WiFi.append(i["WiFi"])
    else:
        WiFi.append(0)
cate_wifi = list(set(WiFi))
new_wifi = []
for i in range(len(WiFi)):
    new_wifi.append(cate_wifi.index(WiFi[i]))
business_feature_df['WiFi']  = pd.DataFrame(new_wifi) 


NoiseLevel = []
for i in list(business_feature_df['attributes']):
    if i ==0 :
        NoiseLevel.append(0)
    elif "NoiseLevel" in i.keys():
        NoiseLevel.append(i["NoiseLevel"])
    else:
        NoiseLevel.append(0)
cate_nl = list(set(NoiseLevel))
new_nl = []
for i in range(len(NoiseLevel)):
    new_nl.append(cate_nl.index(NoiseLevel[i]))
business_feature_df['NoiseLevel']  = pd.DataFrame(new_nl) 


star_mean = np.mean(list(business_feature_df['business_stars']))
business_feature_df['star_diff'] = business_feature_df['business_stars'] - star_mean


'''
features:
BusinessAcceptsCreditCards
GoodForKids
RestaurantsDelivery
RestaurantsGoodForGroups
RestaurantsReservations
RestaurantsTakeOut
WiFi
NoiseLevel'''


#-------------------- select features from user.json ---------------------------
user_feature = sc.textFile(user_feature_path).map(json.loads).map(lambda x: (x['user_id'], x['review_count'], x['average_stars'], x['yelping_since'], x['useful'], x['funny'], x['cool'], x['fans'])).collect()
uf_list = []
for i in user_feature:
    uf_list.append(list(i))
user_feature_df = pd.DataFrame(uf_list, columns = ['user_id', 'review_count_user', 'average_stars','yelping_since','useful', 'funny', 'cool', 'fans'])
user_feature_df = user_feature_df.fillna(0)
#user_feature_df['yelping_since'] = user_feature_df['yelping_since'].astype('category')
user_feature_df['yelping_since'] = pd.to_datetime(user_feature_df['yelping_since'])
latest_time = max(user_feature_df['yelping_since'])
user_feature_df['yelping_since'] = latest_time-pd.to_datetime(user_feature_df['yelping_since'])
#user_feature_df['yelping_since'] = user_feature_df['yelping_since'].days
#user_feature_df['yelping_since']= pd.to_numeric(user_feature_df['yelping_since'])/ 10**9
#print(latest_time)
#print(user_feature_df['yelping_since'])
#print(type(latest_time))
#user_feature_df['yelping_since'] = pd.to_numeric(user_feature_df['yelping_since'], errors='coerce')
#print(type(user_feature_df['yelping_since'][1]))
#print (user_feature_df.isnull().sum())

len(user_feature_df['yelping_since'])
new_yelping_since = []
for i in range(len(user_feature_df['yelping_since'])):
    new_yelping_since.append(user_feature_df['yelping_since'][i].days)
user_feature_df['yelping_since'] = pd.DataFrame(new_yelping_since)

user_star_mean = np.mean(list(user_feature_df['average_stars']))
user_feature_df['user_star_diff'] = user_feature_df['average_stars'] - user_star_mean


#---------------------prepare train and test data----------------------
train_df_b_feature = pd.merge(train_df,business_feature_df, on='business_id', how='left')
train_df_bu_feature = pd.merge(train_df_b_feature,user_feature_df, on='user_id', how='left')
#print(train_df_bu_feature)

use_columns = ['business_stars', 'review_count_business', 'state', 'city','BusinessAcceptsCreditCards', 
               'GoodForKids', 'RestaurantsDelivery', 'RestaurantsGoodForGroups', 'RestaurantsReservations',
               'RestaurantsTakeOut', 'WiFi', 'NoiseLevel','star_diff','review_count_user', 'average_stars', 
               'yelping_since','useful', 'funny', 'cool', 'fans', 'user_star_diff']
train_x = train_df_bu_feature[use_columns]
train_y = train_df_bu_feature['stars']
#print(train_x)
#print(train_y)

test_df = pd.read_csv(test_file_name, sep = ",")
test_df_b_feature = pd.merge(test_df,business_feature_df, on='business_id', how='left')
test_df_bu_feature = pd.merge(test_df_b_feature,user_feature_df, on='user_id', how='left')
test_x = test_df_bu_feature[use_columns]
test_y = test_df_bu_feature['stars']

#-------------------------setup model-----------------------------
model = xgb.XGBRegressor(max_depth = 6,
                         min_child_weight = 1.2, 
                         gamma = 0.1,
                         subsample = 0.8, colsample_bytree = 0.8,
                         booster = 'gbtree',  
                         objective='reg:linear', learning_rate=0.05, n_estimators= 800).fit(train_x, train_y)
#model = xgb.XGBRegressor(objective='reg:gamma').fit(train_x, train_y)
train_pred = model.predict(train_x)
train_mse = mean_squared_error(train_y, train_pred)

test_pred = model.predict(test_x)
test_mse = mean_squared_error(test_y, test_pred)
#print(train_mse, test_mse)

model_result = model.predict(test_x)
#print(type(model_result))
#print(len(model_result))
rmse = mean_squared_error(test_pred, test_y)**0.5
print("RMSE:", rmse)

#-------------------write output --------------------
with open(output_file_name, 'w') as f:
    f.write('user_id,business_id,prediction\n')
    for i in range(len(model_result)):
        f.write(test_df.iloc[i][0]+','+ test_df.iloc[i][1]+','+str(model_result[i])+'\n')

end_time = time()
seconds = end_time - start_time
print('Duration: {}'.format(seconds))



'''#--------------calculate the error distribution----------------------

out = pd.read_csv(output_file_name, sep = ",")
out_list = list(out['prediction'])
cnt_01 = 0
cnt_12 = 0
cnt_23 = 0
cnt_34 = 0
cnt_4p = 0

for i in out_list:
    if 0 <= i <1:
        cnt_01 += 1
    elif 1 <= i <2:
        cnt_12 += 1
    elif 2 <= i <3:
        cnt_23 += 1
    elif 3 <= i <4:
        cnt_34 += 1
    elif 4 <= i:
        cnt_4p += 1

true_ans = pd.read_csv(test_file_name, sep = ",")
true_list = list(true_ans['stars'])
cnt_01_true = 0
cnt_12_true = 0
cnt_23_true = 0
cnt_34_true = 0
cnt_4p_true = 0

for i in true_list:
    if 0 <= i <1:
        cnt_01_true += 1
    elif 1 <= i <2:
        cnt_12_true += 1
    elif 2 <= i <3:
        cnt_23_true += 1
    elif 3 <= i <4:
        cnt_34_true += 1
    elif 4 <= i:
        cnt_4p_true += 1

dif_01 = abs(cnt_01_true - cnt_01)
dif_12 = abs(cnt_12_true - cnt_12)
dif_23 = abs(cnt_23_true - cnt_23)
dif_34 = abs(cnt_34_true - cnt_34)
dif_4p = abs(cnt_4p_true - cnt_4p)
#print(dif_01)
#print(dif_12)
#print(dif_23)
#print(dif_34)
#print(dif_4p)'''