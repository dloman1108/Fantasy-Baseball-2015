
# coding: utf-8

# In[868]:

import numpy as np
import pandas as pd
import json


# In[869]:

filepath='/Users/DanLo1108/Documents/Projects/Fantasy Baseball Project/data/'


# In[870]:

N=18269
buckets=list(np.arange(0,N,800))
buckets.append(N)


# In[872]:

#Import json files from data acquisition and converts them to
#pandas dataframes
batting=pd.DataFrame()
pitching=pd.DataFrame()
for b in range(1,len(buckets)):
    start=buckets[b-1]
    end=buckets[b]
    
    with open(filepath+'batting_stats_'+str(start)+'-'+str(end)+'.json') as json_data:
        bat_dict = json.load(json_data)
        json_data.close()
    
    bat_dict['names']=bat_dict['names'][0:len(bat_dict['games'])]
        
    with open(filepath+'pitching_stats_'+str(start)+'-'+str(end)+'.json') as json_data:
        pitch_dict = json.load(json_data)
        json_data.close()
        
    pitch_dict['names']=pitch_dict['names'][0:len(pitch_dict['games'])]
    
    temp_bat=pd.DataFrame()
    for bat_key in bat_dict:
        temp_bat[bat_key]=bat_dict[bat_key]
        
    batting=batting.append(temp_bat)
        
    temp_pitch=pd.DataFrame()
    for pitch_key in pitch_dict:
        temp_pitch[pitch_key]=pitch_dict[pitch_key]
        
    pitching=pitching.append(temp_pitch)



# In[873]:

#Rearrange columns in batting dataframe
batting=batting.rename(columns={'leagues':'league','teams':'team','ages':'age','names':'name','years':'year',
                'batting_averages':'batting_average'})

col_order=['name','year','age','team','league','games','at_bats','batting_average','hits','doubles',
           'home_runs','rbis','runs','stolen_bases','strikeouts','walks']
batting=batting[col_order]

batting.to_csv(filepath+'batting.csv')


# In[874]:

#Rearrange columns in pitching dataframe
pitching=pitching.rename(columns={'leagues':'league','teams':'team','ages':'age','names':'name','years':'year',
                'eras':'era','whips':'whip'})

col_order=['name','year','age','team','league','games','innings','wins','era','strikeouts','whip',
           'saves','hits_9','hr_9','bb_9','so_9']
pitching=pitching[col_order]

pitching.to_csv(filepath+'pitching.csv')


# In[1082]:

#Gets batting stats from player, including lagged data

#Specify cutoff criteria
start_year=1960
min_ab=100
def get_player_batting_stats(group, stats, features,year=start_year,min_ab=min_ab):    
    inds=group.index.values
    for i in range(len(inds)):
        ind=inds[i]
        g=group.ix[ind]
        if g.year > 1990 and g.at_bats >= min_ab:
            stats['name'].append(g['name'])
            stats['age'].append(g['age'])
            stats['year'].append(g['year'])
            for j in range(0,4):
                if i > j:
                    for feat in features:
                        if j > 0:
                            strr=feat+'_lag_'+str(j)
                        else:
                            strr=feat
                        ind_lag=inds[i-j]
                        g_lag=group.ix[ind_lag]
                        if feat not in ['games','batting_average']:
                            stats[strr].append(g_lag[feat]/g_lag['games'])
                        else:
                            stats[strr].append(g_lag[feat])
                else:
                    for feat in features:
                        if j==0:
                            strr=feat
                        else:
                            strr=feat+'_lag_'+str(j)
                        stats[strr].append(np.nan)
            
        else:
            continue


# In[1083]:

#Gets pitching stats from player, including lagged data

#specify cutoff criteria
start_year=1960
min_innings=20
def get_player_pitching_stats(group, stats, features,year=start_year,min_innings=min_innings):    
    inds=group.index.values
    for i in range(len(inds)):
        ind=inds[i]
        g=group.ix[ind]
        if g.year >= year and g.innings >= min_innings:
            stats['name'].append(g['name'])
            stats['age'].append(g['age'])
            stats['year'].append(g['year'])
            for j in range(0,4):
                if i > j:
                    for feat in features:
                        if j > 0:
                            strr=feat+'_lag_'+str(j)
                        else:
                            strr=feat
                        ind_lag=inds[i-j]
                        g_lag=group.ix[ind_lag]
                        if feat not in ['games','era','whip','hits_9','bb_9','so_9','hr_9']:
                            stats[strr].append(g_lag[feat]/g_lag['games'])
                        else:
                            stats[strr].append(g_lag[feat])
                else:
                    for feat in features:
                        if j==0:
                            strr=feat
                        else:
                            strr=feat+'_lag_'+str(j)
                        stats[strr].append(np.nan)
            
        else:
            continue


# In[1084]:

#Get features for batting training data
bat_features=batting.columns.tolist()
total_batting_stats={}
bat_df_cols=[]
non_lag_features=['name','year','age','team','league']
for f in non_lag_features:
    bat_features.remove(f)
    if f in ['name','age','year']:
        bat_df_cols.append(f)
        total_batting_stats[f]=[]
    

for feat in bat_features:
    for j in range(0,4):
        if j > 0:
            strr=feat+'_lag_'+str(j)
        else:
            strr=feat
        bat_df_cols.append(strr)
        total_batting_stats[strr]=[]
        


# In[1085]:

#Get features for pitch training data
pit_features=pitching.columns.tolist()
total_pitching_stats={}
pit_df_cols=[]
non_lag_features=['name','year','age','team','league']
targets=['games','wins','era','strikeouts','saves','whip']
for f in non_lag_features:
    pit_features.remove(f)
    if f in ['name','age','year']:
        pit_df_cols.append(f)
        total_pitching_stats[f]=[]
    

for feat in pit_features:
    for j in range(0,4):
        if j > 0:
            strr=feat+'_lag_'+str(j)
        else:
            strr=feat
        pit_df_cols.append(strr)
        total_pitching_stats[strr]=[]


# In[1086]:

#Loop through each batter, extracting their stats
batting_groups=batting.groupby('name')
for name,group in batting_groups:
    group=group.sort('year')
    get_player_batting_stats(group,total_batting_stats,bat_features)


# In[1087]:

#Loop through each pitcher, extracting their stats
pitching_groups=pitching.groupby('name')
for name,group in pitching_groups:
    group=group.sort('year')
    get_player_pitching_stats(group,total_pitching_stats,pit_features)


# In[1088]:

#Rearrange batting stats dataframe - create training data
batting_stats=pd.DataFrame({'year':np.array(total_batting_stats['year'])})
for key in total_batting_stats:
    batting_stats[key]=np.array(total_batting_stats[key])

bat_train=batting_stats[bat_df_cols]


# In[1089]:

#Rearrange pitching stats dataframe - create training data
pitching_stats=pd.DataFrame({'year':np.array(total_pitching_stats['year'])})
for key in total_pitching_stats:
    pitching_stats[key]=np.array(total_pitching_stats[key])

pitch_train=pitching_stats[pit_df_cols]


# In[1090]:

#get last years stats
last_year_bat=bat_train[bat_train.year==2014]
last_year_pitch=pitch_train[pitch_train.year==2014]


# In[1091]:

#Creates testing data by using 2014 batters for 2015
bat_test_dict={}
bat_feats=last_year_bat.columns.tolist()
for ind in last_year_bat.index.values:
    player=last_year_bat.ix[ind]
    if 'name' not in bat_test_dict:
        bat_test_dict['name']=[player['name']]
        bat_test_dict['age']=[int(player['age'])+1]
        bat_test_dict['year']=[int(player['year'])+1]
    else:
        bat_test_dict['name'].append(player['name'])
        bat_test_dict['age'].append(int(player['age'])+1)
        bat_test_dict['year'].append(int(player['year'])+1)
    for i in range(3,len(bat_feats),4):
        feat=bat_feats[i]
        for j in range(0,3):
            strr1=feat+'_lag_'+str(j+1)
            if j==0:
                strr2=feat
            else:
                strr2=feat+'_lag_'+str(j)
                
            if strr1 not in bat_test_dict:
                bat_test_dict[strr1]=[player[strr2]]
            else:
                bat_test_dict[strr1].append(player[strr2])
            
bat_test=pd.DataFrame()
for key in bat_test_dict:
    bat_test[key]=bat_test_dict[key]
            


# In[885]:

#Creates testing data by using 2014 pitchers for 2015
pitch_test_dict={}
pitch_feats=last_year_pitch.columns.tolist()
for ind in last_year_pitch.index.values:
    player=last_year_pitch.ix[ind]
    if 'name' not in pitch_test_dict:
        pitch_test_dict['name']=[player['name']]
        pitch_test_dict['age']=[int(player['age'])+1]
        pitch_test_dict['year']=[int(player['year'])+1]
    else:
        pitch_test_dict['name'].append(player['name'])
        pitch_test_dict['age'].append(int(player['age'])+1)
        pitch_test_dict['year'].append(int(player['year'])+1)
    for i in range(3,len(pitch_feats),4):
        feat=pitch_feats[i]
        for j in range(0,3):
            strr1=feat+'_lag_'+str(j+1)
            if j==0:
                strr2=feat
            else:
                strr2=feat+'_lag_'+str(j)
                
            if strr1 not in pitch_test_dict:
                pitch_test_dict[strr1]=[player[strr2]]
            else:
                pitch_test_dict[strr1].append(player[strr2])
            
pitch_test=pd.DataFrame()
for key in pitch_test_dict:
    pitch_test[key]=pitch_test_dict[key]
            


# In[917]:

#Gets batting target variables and column order
bat_test_cols=bat_df_cols
for col in bat_train.columns.tolist():
    if col not in bat_test.columns.tolist():
        try:
            bat_test_cols.remove(col)
        except:
            continue

bat_targets=['games','at_bats','batting_average','runs','rbis','home_runs','stolen_bases']
        
bat_test=bat_test[bat_test_cols]


# In[918]:

#Gets pitching target variables
pitch_test_cols=pit_df_cols
for col in pitch_train.columns.tolist():
    if col not in pitch_test.columns.tolist():
        try:
            pitch_test_cols.remove(col)
        except:
            continue
        
pitch_targets=['games','innings','era','wins','whip','saves','strikeouts']
        
pitch_test=pitch_test[pitch_test_cols]


### Batting - Machine Learning

# In[964]:

#Runs regression for veteran players (all lagged data)

bat_train3=bat_train.dropna(subset=['games_lag_3'])
bat_train3=bat_train3.dropna()
bat_y_train3=bat_train3[bat_targets]
bat_X_train3=bat_train3.drop(bat_targets,axis=1)

bat_test3=bat_test.dropna(subset=['games_lag_3'])
bat_test3=bat_test3.dropna()

games_features=['age','games_lag_1','games_lag_2','games_lag_3']
ba_features=['age','batting_average_lag_1','batting_average_lag_2','batting_average_lag_3']
runs_features=['age','runs_lag_1','runs_lag_2','runs_lag_3']
hr_features=['age','home_runs_lag_1','home_runs_lag_2','home_runs_lag_3']
rbi_features=['age','rbis_lag_1','rbis_lag_2','rbis_lag_3']
sb_features=['age','stolen_bases_lag_1','stolen_bases_lag_2','stolen_bases_lag_3']

from sklearn.linear_model import LinearRegression

for i in range(1):
    
    X_train=bat_X_train3
    y_train=bat_y_train3
    X_test=bat_test3

    names=np.array(X_test['name'])
    results1=pd.DataFrame({'name':names})

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    r2s['games'].append(games_predict.score(X_train[games_features],y_train['games']))
    games_preds=games_predict.predict(X_test[games_features])
    results1['predicted_games']=games_preds
  

    #batting_average
    ba_predict=LinearRegression()
    ba_predict.fit(X_train[ba_features],y_train['batting_average'])
    r2s['ba'].append(ba_predict.score(X_train[ba_features],y_train['batting_average']))
    ba_preds=ba_predict.predict(X_test[ba_features])
    results1['predicted_ba']=ba_preds
    results1['predicted_ba_score']=ba_preds*games_preds
    
    #home runs
    hr_predict=LinearRegression()
    hr_predict.fit(X_train[hr_features],y_train['home_runs'])
    r2s['hr'].append(hr_predict.score(X_train[hr_features],y_train['home_runs']))
    hr_preds=hr_predict.predict(X_test[hr_features])
    results1['predicted_hr']=hr_preds*games_preds

    #rbis
    rbi_predict=LinearRegression()
    rbi_predict.fit(X_train[rbi_features],y_train['rbis'])
    r2s['rbis'].append(rbi_predict.score(X_train[rbi_features],y_train['rbis']))
    rbi_preds=rbi_predict.predict(X_test[rbi_features])
    results1['predicted_rbis']=rbi_preds*games_preds

    #runs
    runs_predict=LinearRegression()
    runs_predict.fit(X_train[runs_features],y_train['runs'])
    r2s['runs'].append(runs_predict.score(X_train[runs_features],y_train['runs']))
    runs_preds=runs_predict.predict(X_test[runs_features])
    results1['predicted_runs']=runs_preds*games_preds

    #stolen bases
    sb_predict=LinearRegression()
    sb_predict.fit(X_train[sb_features],y_train['stolen_bases'])
    sb_preds=sb_predict.predict(X_test[sb_features])
    r2s['sb'].append(sb_predict.score(X_train[sb_features],y_train['stolen_bases']))
    results1['predicted_sb']=sb_preds*games_preds


            


# In[965]:

#Third year (two lagged seasons)
bat_train2=bat_train[np.isnan(bat_train.games_lag_3)][bat_train.games_lag_2<1000]
bat_train2=bat_train2.dropna(subset=['games','batting_average_lag_1','batting_average_lag_2'])
bat_y_train2=bat_train2[bat_targets]
bat_X_train2=bat_train2.drop(bat_targets,axis=1)

bat_test2=bat_test[np.isnan(bat_test.games_lag_3)][bat_test.games_lag_2<1000]
#bat_test2=bat_test2.dropna(subset=['games','batting_average_lag_1','batting_average_lag_2'])

games_features=['age','games_lag_1','games_lag_2']
ab_features=['age','at_bats_lag_1','at_bats_lag_2']
ba_features=['age','batting_average_lag_1','batting_average_lag_2']
runs_features=['age','runs_lag_1','runs_lag_2']
hr_features=['age','home_runs_lag_1','home_runs_lag_2']
rbi_features=['age','rbis_lag_1','rbis_lag_2']
sb_features=['age','stolen_bases_lag_1','stolen_bases_lag_2']

from sklearn.linear_model import LinearRegression
for i in range(1):
    
    X_train=bat_X_train2
    y_train=bat_y_train2
    X_test=bat_test2

    names=np.array(X_test['name'])
    results2=pd.DataFrame({'name':names})

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    games_preds=games_predict.predict(X_test[games_features])
    results2['predicted_games']=games_preds
    

    #batting_average
    ba_predict=LinearRegression()
    ba_predict.fit(X_train[ba_features],y_train['batting_average'])
    ba_preds=ba_predict.predict(X_test[ba_features])
    results2['predicted_ba']=ba_preds
    results2['predicted_ba_score']=ba_preds*games_preds

    #home runs
    hr_predict=LinearRegression()
    hr_predict.fit(X_train[hr_features],y_train['home_runs'])
    hr_preds=hr_predict.predict(X_test[hr_features])
    results2['predicted_hr']=hr_preds*games_preds

    #rbis
    rbi_predict=LinearRegression()
    rbi_predict.fit(X_train[rbi_features],y_train['rbis'])
    rbi_preds=rbi_predict.predict(X_test[rbi_features])
    results2['predicted_rbis']=rbi_preds*games_preds

    #runs
    runs_predict=LinearRegression()
    runs_predict.fit(X_train[runs_features],y_train['runs'])
    runs_preds=runs_predict.predict(X_test[runs_features])
    results2['predicted_runs']=runs_preds*games_preds

    #stolen bases
    sb_predict=LinearRegression()
    sb_predict.fit(X_train[sb_features],y_train['stolen_bases'])
    sb_preds=sb_predict.predict(X_test[sb_features])
    results2['predicted_sb']=sb_preds*games_preds

            


# In[966]:

results=results1.append(results2)


# In[967]:

#Second year (one lagged season)
cols=['name','year','age','games','games_lag_1','batting_average','batting_average_lag_1',
      'home_runs','home_runs_lag_1','rbis','rbis_lag_1','runs','runs_lag_1','stolen_bases',
      'stolen_bases_lag_1','at_bats','at_bats_lag_1']
bat_train1=bat_train[np.isnan(bat_train.games_lag_3)][np.isnan(bat_train.games_lag_2)][bat_train.games_lag_1<1000]
bat_train1=bat_train1[cols].dropna()
bat_y_train1=bat_train1[bat_targets]
bat_X_train1=bat_train1.drop(bat_targets,axis=1)

bat_test1=bat_test[np.isnan(bat_test.games_lag_3)][np.isnan(bat_test.games_lag_2)][bat_test.games_lag_1<1000]

games_features=['age','games_lag_1']
ba_features=['age','batting_average_lag_1']
runs_features=['age','runs_lag_1']
hr_features=['age','home_runs_lag_1']
rbi_features=['age','rbis_lag_1']
sb_features=['age','stolen_bases_lag_1']

from sklearn.linear_model import LinearRegression
for i in range(1):
    
    X_train=bat_X_train1
    y_train=bat_y_train1
    X_test=bat_test1

    names=X_test['name']
    results3=pd.DataFrame({'name':names})

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    r2s['games'].append(games_predict.score(X_train[games_features],y_train['games']))
    games_preds=games_predict.predict(X_test[games_features])
    results3['predicted_games']=games_preds
    

    #batting_average
    ba_predict=LinearRegression()
    ba_predict.fit(X_train[ba_features],y_train['batting_average'])
    r2s['ba'].append(ba_predict.score(X_train[ba_features],y_train['batting_average']))
    ba_preds=ba_predict.predict(X_test[ba_features])
    results3['predicted_ba']=ba_preds
    results3['predicted_ba_score']=ba_preds*games_preds

    #home runs
    hr_predict=LinearRegression()
    hr_predict.fit(X_train[hr_features],y_train['home_runs'])
    r2s['hr'].append(hr_predict.score(X_train[hr_features],y_train['home_runs']))
    hr_preds=hr_predict.predict(X_test[hr_features])
    results3['predicted_hr']=hr_preds*games_preds

    #rbis
    rbi_predict=LinearRegression()
    rbi_predict.fit(X_train[rbi_features],y_train['rbis'])
    r2s['rbis'].append(rbi_predict.score(X_train[rbi_features],y_train['rbis']))
    rbi_preds=rbi_predict.predict(X_test[rbi_features])
    results3['predicted_rbis']=rbi_preds*games_preds

    #runs
    runs_predict=LinearRegression()
    runs_predict.fit(X_train[runs_features],y_train['runs'])
    r2s['runs'].append(runs_predict.score(X_train[runs_features],y_train['runs']))
    runs_preds=runs_predict.predict(X_test[runs_features])
    results3['predicted_runs']=runs_preds*games_preds

    #stolen bases
    sb_predict=LinearRegression()
    sb_predict.fit(X_train[sb_features],y_train['stolen_bases'])
    sb_preds=sb_predict.predict(X_test[sb_features])
    r2s['sb'].append(sb_predict.score(X_train[sb_features],y_train['stolen_bases']))
    results3['predicted_sb']=sb_preds*games_preds


# In[968]:

batter_results=results.append(results3)


# In[969]:

#Basic statistics of results (assume normality)
ave_ba=np.mean(batter_results.predicted_ba_score)
ave_hr=np.mean(batter_results.predicted_hr)
ave_rbis=np.mean(batter_results.predicted_rbis)
ave_runs=np.mean(batter_results.predicted_runs)
ave_sb=np.mean(batter_results.predicted_sb)

std_ba=np.std(batter_results.predicted_ba_score)
std_hr=np.std(batter_results.predicted_hr)
std_rbis=np.std(batter_results.predicted_rbis)
std_runs=np.std(batter_results.predicted_runs)
std_sb=np.std(batter_results.predicted_sb)


# In[970]:

def get_z_score(x,stat,mean,sd):
    return (x[stat]-mean)/sd

batter_results['ba_score']=batter_results.apply(lambda x: get_z_score(x,'predicted_ba_score',ave_ba,std_ba),axis=1)
batter_results['hr_score']=batter_results.apply(lambda x: get_z_score(x,'predicted_hr',ave_hr,std_hr),axis=1)
batter_results['rbi_score']=batter_results.apply(lambda x: get_z_score(x,'predicted_rbis',ave_rbis,std_rbis),axis=1)
batter_results['runs_score']=batter_results.apply(lambda x: get_z_score(x,'predicted_runs',ave_runs,std_runs),axis=1)
batter_results['sb_score']=batter_results.apply(lambda x: get_z_score(x,'predicted_sb',ave_sb,std_sb),axis=1)


# In[1026]:

def get_total_bat_score(x):
    return x.ba_score+x.hr_score+x.rbi_score+x.runs_score+x.sb_score

batter_results['total_score']=batter_results.apply(lambda x: get_total_bat_score(x), axis=1)


# In[1075]:

final_batting_results=batter_results.sort('total_score',ascending=False)[['name','total_score','ba_score','hr_score','rbi_score','runs_score','sb_score']]
final_batting_results=final_batting_results.drop_duplicates('name').reset_index().drop('index',axis=1)
final_batting_results[final_batting_results.name=='Adrian (Perez) Beltre ']


### Pitching Machine Learning

# In[982]:

#Runs regression for veteran players (all lagged data)

pitch_train3=pitch_train.dropna(subset=['games_lag_3'])
pitch_train3=pitch_train3.dropna()
pitch_y_train3=pitch_train3[pitch_targets]
pitch_X_train3=pitch_train3.drop(pitch_targets,axis=1)

pitch_test3=pitch_test.dropna(subset=['games_lag_3'])
pitch_test3=pitch_test3.dropna()

games_features=['age','games_lag_1','games_lag_2','games_lag_3']
innings_features=['age','innings_lag_1','innings_lag_2','innings_lag_3']
era_features=['age','era_lag_1','era_lag_2','era_lag_3']
whip_features=['age','whip_lag_1','whip_lag_2','whip_lag_3']
so_features=['age','strikeouts_lag_1','strikeouts_lag_2','strikeouts_lag_3']
wins_features=['age','wins_lag_1','wins_lag_2','wins_lag_3']
saves_features=['age','saves_lag_1','saves_lag_2','saves_lag_3']

from sklearn.linear_model import LinearRegression

for i in range(1):
    
    X_train=pitch_X_train3
    y_train=pitch_y_train3
    X_test=pitch_test3
    

    names=np.array(X_test['name'])
    results1=pd.DataFrame({'name':names})

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    games_preds=games_predict.predict(X_test[games_features])
    results1['predicted_games']=games_preds
    
    #innings
    innings_predict=LinearRegression()
    innings_predict.fit(X_train[innings_features],y_train['innings'])
    innings_preds=innings_predict.predict(X_test[innings_features])*games_preds
    results1['predicted_innings']=innings_preds

    #era
    era_predict=LinearRegression()
    era_predict.fit(X_train[era_features],y_train['era'])
    era_preds=era_predict.predict(X_test[era_features])
    results1['predicted_era']=era_preds
    results1['predicted_era_score']=(era_preds.max()-era_preds)*innings_preds

    #home runs
    whip_predict=LinearRegression()
    whip_predict.fit(X_train[whip_features],y_train['whip'])
    whip_preds=whip_predict.predict(X_test[whip_features])
    results1['predicted_whip']=whip_preds
    results1['predicted_whip_score']=(whip_preds.max()-whip_preds)*innings_preds

    #rbis
    so_predict=LinearRegression()
    so_predict.fit(X_train[so_features],y_train['strikeouts'])
    so_preds=so_predict.predict(X_test[so_features])
    results1['predicted_so']=so_preds*games_preds

    #runs
    wins_predict=LinearRegression()
    wins_predict.fit(X_train[wins_features],y_train['wins'])
    wins_preds=wins_predict.predict(X_test[wins_features])
    results1['predicted_wins']=wins_preds*games_preds

    #stolen bases
    saves_predict=LinearRegression()
    saves_predict.fit(X_train[saves_features],y_train['saves'])
    saves_preds=saves_predict.predict(X_test[saves_features])
    results1['predicted_saves']=saves_preds*games_preds


            


# In[1004]:

#Third season (2 lagged years)
cols=['name','year','age','games','games_lag_1','games_lag_2','innings','innings_lag_1',
      'innings_lag_2','era','era_lag_1','era_lag_2','whip','whip_lag_1','whip_lag_2',
      'strikeouts','strikeouts_lag_1','strikeouts_lag_2', 'wins','wins_lag_1','wins_lag_2',
      'saves','saves_lag_1','saves_lag_2']

cols1=['name','year','age','games_lag_1','games_lag_2','innings_lag_1',
      'innings_lag_2','era_lag_1','era_lag_2','whip_lag_1','whip_lag_2',
      'strikeouts_lag_1','strikeouts_lag_2', 'wins_lag_1','wins_lag_2',
      'saves_lag_1','saves_lag_2']

pitch_train2=pitch_train[np.isnan(pitch_train.games_lag_3)][pitch_train.games_lag_2<1000]
pitch_train2=pitch_train2[cols].dropna()
pitch_y_train2=pitch_train2[pitch_targets]
pitch_X_train2=pitch_train2.drop(pitch_targets,axis=1)

pitch_test2=pitch_test[np.isnan(pitch_test.games_lag_3)][pitch_test.games_lag_2<1000]
pitch_test2=pitch_test2[cols1].dropna()

games_features=['age','games_lag_1','games_lag_2']
innings_features=['age','innings_lag_1','innings_lag_2']
era_features=['age','era_lag_1','era_lag_2']
whip_features=['age','whip_lag_1','whip_lag_2']
so_features=['age','strikeouts_lag_1','strikeouts_lag_2']
wins_features=['age','wins_lag_1','wins_lag_2']
saves_features=['age','saves_lag_1','saves_lag_2']

from sklearn.linear_model import LinearRegression

for i in range(1):
    
    X_train=pitch_X_train2
    y_train=pitch_y_train2
    X_test=pitch_test2
    

    names=np.array(X_test['name'])
    results2=pd.DataFrame({'name':names})

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    games_preds=games_predict.predict(X_test[games_features])
    results2['predicted_games']=games_preds
    
    #innings
    innings_predict=LinearRegression()
    innings_predict.fit(X_train[innings_features],y_train['innings'])
    innings_preds=innings_predict.predict(X_test[innings_features])*games_preds
    results2['predicted_innings']=innings_preds

    #era
    era_predict=LinearRegression()
    era_predict.fit(X_train[era_features],y_train['era'])
    era_preds=era_predict.predict(X_test[era_features])
    results2['predicted_era']=era_preds
    results2['predicted_era_score']=(era_preds.max()-era_preds)*innings_preds

    #home runs
    whip_predict=LinearRegression()
    whip_predict.fit(X_train[whip_features],y_train['whip'])
    whip_preds=whip_predict.predict(X_test[whip_features])
    results2['predicted_whip']=whip_preds
    results2['predicted_whip_score']=(whip_preds.max()-whip_preds)*innings_preds

    #rbis
    so_predict=LinearRegression()
    so_predict.fit(X_train[so_features],y_train['strikeouts'])
    so_preds=so_predict.predict(X_test[so_features])
    results2['predicted_so']=so_preds*games_preds

    #runs
    wins_predict=LinearRegression()
    wins_predict.fit(X_train[wins_features],y_train['wins'])
    wins_preds=wins_predict.predict(X_test[wins_features])
    results2['predicted_wins']=wins_preds*games_preds

    #stolen bases
    saves_predict=LinearRegression()
    saves_predict.fit(X_train[saves_features],y_train['saves'])
    saves_preds=saves_predict.predict(X_test[saves_features])
    results2['predicted_saves']=saves_preds*games_preds



# In[1007]:

results=results1.append(results2)


# In[1008]:

#second season (1 lagged year)
cols=['name','year','age','games','games_lag_1','innings','innings_lag_1',
      'era','era_lag_1','whip','whip_lag_1',
      'strikeouts','strikeouts_lag_1', 'wins','wins_lag_1','saves','saves_lag_1']

cols1=['name','year','age','games_lag_1','innings_lag_1','era_lag_1','whip_lag_1',
      'strikeouts_lag_1', 'wins_lag_1','saves_lag_1']

pitch_train1=pitch_train[np.isnan(pitch_train.games_lag_3)][np.isnan(pitch_train.games_lag_2)][pitch_train.games_lag_1<1000]
pitch_train1=pitch_train1[cols].dropna()
pitch_y_train1=pitch_train1[pitch_targets]
pitch_X_train1=pitch_train1.drop(pitch_targets,axis=1)

pitch_test1=pitch_test[np.isnan(pitch_test.games_lag_3)][np.isnan(pitch_train.games_lag_2)][pitch_test.games_lag_1<1000]
pitch_test1=pitch_test1[cols1].dropna()

games_features=['age','games_lag_1']
innings_features=['age','innings_lag_1']
era_features=['age','era_lag_1']
whip_features=['age','whip_lag_1']
so_features=['age','strikeouts_lag_1']
wins_features=['age','wins_lag_1']
saves_features=['age','saves_lag_1']

from sklearn.linear_model import LinearRegression

for i in range(1):
    
    X_train=pitch_X_train1
    y_train=pitch_y_train1
    X_test=pitch_test1
    

    names=np.array(X_test['name'])
    results3=pd.DataFrame({'name':names})

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    games_preds=games_predict.predict(X_test[games_features])
    results3['predicted_games']=games_preds
    
    #innings
    innings_predict=LinearRegression()
    innings_predict.fit(X_train[innings_features],y_train['innings'])
    innings_preds=innings_predict.predict(X_test[innings_features])*games_preds
    results3['predicted_innings']=innings_preds

    #era
    era_predict=LinearRegression()
    era_predict.fit(X_train[era_features],y_train['era'])
    era_preds=era_predict.predict(X_test[era_features])
    results3['predicted_era']=era_preds
    results3['predicted_era_score']=(era_preds.max()-era_preds)*innings_preds

    #home runs
    whip_predict=LinearRegression()
    whip_predict.fit(X_train[whip_features],y_train['whip'])
    whip_preds=whip_predict.predict(X_test[whip_features])
    results3['predicted_whip']=whip_preds
    results3['predicted_whip_score']=(whip_preds.max()-whip_preds)*innings_preds

    #rbis
    so_predict=LinearRegression()
    so_predict.fit(X_train[so_features],y_train['strikeouts'])
    so_preds=so_predict.predict(X_test[so_features])
    results3['predicted_so']=so_preds*games_preds

    #runs
    wins_predict=LinearRegression()
    wins_predict.fit(X_train[wins_features],y_train['wins'])
    wins_preds=wins_predict.predict(X_test[wins_features])
    results3['predicted_wins']=wins_preds*games_preds

    #stolen bases
    saves_predict=LinearRegression()
    saves_predict.fit(X_train[saves_features],y_train['saves'])
    saves_preds=saves_predict.predict(X_test[saves_features])
    results3['predicted_saves']=saves_preds*games_preds


# In[1009]:

pitching_results=results.append(results3)


# In[1011]:

#Basic statistics of results (assume normality)
ave_era=np.mean(pitching_results.predicted_era_score)
ave_whip=np.mean(pitching_results.predicted_whip_score)
ave_so=np.mean(pitching_results.predicted_so)
ave_wins=np.mean(pitching_results.predicted_wins)
ave_saves=np.mean(pitching_results.predicted_saves)

std_era=np.std(pitching_results.predicted_era_score)
std_whip=np.std(pitching_results.predicted_whip_score)
std_so=np.std(pitching_results.predicted_so)
std_wins=np.std(pitching_results.predicted_wins)
std_saves=np.std(pitching_results.predicted_saves)


# In[1012]:

pitching_results['era_score']=pitching_results.apply(lambda x: get_z_score(x,'predicted_era_score',ave_era,std_era),axis=1)
pitching_results['whip_score']=pitching_results.apply(lambda x: get_z_score(x,'predicted_whip_score',ave_whip,std_whip),axis=1)
pitching_results['so_score']=pitching_results.apply(lambda x: get_z_score(x,'predicted_so',ave_so,std_so),axis=1)
pitching_results['wins_score']=pitching_results.apply(lambda x: get_z_score(x,'predicted_wins',ave_wins,std_wins),axis=1)
pitching_results['saves_score']=pitching_results.apply(lambda x: get_z_score(x,'predicted_saves',ave_saves,std_saves),axis=1)


# In[1119]:

def get_total_pitch_score(x):
    return x.era_score+x.whip_score+x.so_score+x.wins_score+x.saves_score

pitching_results['total_score']=pitching_results.apply(lambda x: get_total_pitch_score(x), axis=1)


# In[1054]:

final_pitching_results=pitching_results.sort('total_score',ascending=False)[['name','total_score','era_score','whip_score','so_score','wins_score','saves_score']]
final_pitching_results=final_pitching_results.drop_duplicates('name').reset_index().drop('index',axis=1)
final_pitching_results


# In[ ]:




# In[ ]:




#### Example of checking MAE

# In[594]:

#Second year (one lagged season)
cols=['name','year','age','games','games_lag_1','batting_average','batting_average_lag_1',
      'home_runs','home_runs_lag_1','rbis','rbis_lag_1','runs','runs_lag_1','stolen_bases',
      'stolen_bases_lag_1']
bat_train1=bat_train[np.isnan(bat_train.games_lag_3)][np.isnan(bat_train.games_lag_2)][bat_train.games_lag_1<1000]
bat_train1=bat_train1[cols].dropna()
#bat_train1=bat_train1.dropna()
bat_X_test1=bat_train1[bat_targets]
bat_X_train1=bat_train1.drop(bat_targets,axis=1)

games_features=['age','games_lag_1']
ba_features=['age','batting_average_lag_1']
runs_features=['age','runs_lag_1']
hr_features=['age','home_runs_lag_1']
rbi_features=['age','rbis_lag_1']
sb_features=['age','stolen_bases_lag_1']

from sklearn.linear_model import LinearRegression
maes={}
r2s={}
targets=['games','ba','hr','rbis','runs','sb']
for t in targets:
    r2s[t]=[]
for i in range(10):
    
    X_inds=bat_X_train1.index.values
    X_train=bat_X_train1.drop(['name','year'],axis=1).ix[X_inds[0]:X_inds[int(len(X_inds)*.9)]]
    y_train=bat_X_test1.ix[X_inds[0]:X_inds[int(len(X_inds)*.9)]]
    X_test=bat_X_train1.drop(['name','year'],axis=1).ix[X_inds[int(len(X_inds)*.9)+1]:X_inds[len(X_inds)-1]]
    y_test=bat_X_test1.ix[X_inds[int(len(X_inds)*.9)+1]:X_inds[len(X_inds)-1]]

    names=bat_X_train1.ix[X_inds[int(len(X_inds)*.9)+1]:X_inds[len(X_inds)-1]]['name']
    results=pd.DataFrame({'name':names})
    #from sklearn.cross_validation import train_test_split
    #X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)

    #games
    games_predict=LinearRegression()
    games_predict.fit(X_train[games_features],y_train['games'])
    r2s['games'].append(games_predict.score(X_train[games_features],y_train['games']))
    games_preds=games_predict.predict(X_test[games_features])
    results['predicted_games']=games_preds
    results['actual_games']=y_test['games']

    #batting_average
    ba_predict=LinearRegression()
    ba_predict.fit(X_train[ba_features],y_train['batting_average'])
    r2s['ba'].append(ba_predict.score(X_train[ba_features],y_train['batting_average']))
    ba_preds=ba_predict.predict(X_test[ba_features])
    results['predicted_ba']=ba_preds
    results['actual_ba']=y_test['batting_average']

    #home runs
    hr_predict=LinearRegression()
    hr_predict.fit(X_train[hr_features],y_train['home_runs'])
    r2s['hr'].append(hr_predict.score(X_train[hr_features],y_train['home_runs']))
    hr_preds=hr_predict.predict(X_test[hr_features])
    results['predicted_hr']=hr_preds*games_preds
    results['actual_hr']=y_test['home_runs']*y_test['games']

    #rbis
    rbi_predict=LinearRegression()
    rbi_predict.fit(X_train[rbi_features],y_train['rbis'])
    r2s['rbis'].append(rbi_predict.score(X_train[rbi_features],y_train['rbis']))
    rbi_preds=rbi_predict.predict(X_test[rbi_features])
    results['predicted_rbis']=rbi_preds*games_preds
    results['actual_rbis']=y_test['rbis']*y_test['games']

    #runs
    runs_predict=LinearRegression()
    runs_predict.fit(X_train[runs_features],y_train['runs'])
    r2s['runs'].append(runs_predict.score(X_train[runs_features],y_train['runs']))
    runs_preds=runs_predict.predict(X_test[runs_features])
    results['predicted_runs']=runs_preds*games_preds
    results['actual_runs']=y_test['runs']*y_test['games']

    #stolen bases
    sb_predict=LinearRegression()
    sb_predict.fit(X_train[sb_features],y_train['stolen_bases'])
    sb_preds=sb_predict.predict(X_test[sb_features])
    r2s['sb'].append(sb_predict.score(X_train[sb_features],y_train['stolen_bases']))
    results['predicted_sb']=sb_preds*games_preds
    results['actual_sb']=y_test['stolen_bases']*y_test['games']


    for t in targets:
        pred='predicted_'+t
        act='actual_'+t
        mae=np.mean(abs(results[pred]-results[act]))
        if t not in maes:
            maes[t]=[mae]
        else:
            maes[t].append(mae)
            


# In[690]:

for mae in maes:
    print mae, np.mean(maes[mae])


# In[ ]:



