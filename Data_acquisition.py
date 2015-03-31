
# coding: utf-8

# In[1]:

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import urllib2
import re
import json


# In[2]:

filepath='/Users/DanLo1108/Documents/Projects/Fantasy Baseball Project/data/'


# In[3]:

#Get url of every player letter
url_base='http://www.baseball-reference.com/players/'
letters=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
letter_urls=[]
for l in letters:
    url=url_base+l+'/'
    letter_urls.append(url)


# In[4]:

#Extract player urls from each letter url
player_urls=[]
for url in letter_urls:
    #Gets contents of url web page
    request=urllib2.Request(url)
    page = urllib2.urlopen(request)
    #Reads contents of page
    content=page.read()
    soup=BeautifulSoup(content,'lxml')
    
    results=soup.find_all('a')
    
    #Extracts all player urls
    for r in results:
        search=re.search(r'(/players/)(\w)(/)(\w+)(\d+)(.shtml)',str(r))
        if search is not None:
            player_url='http://www.baseball-reference.com'+search.group()
            player_urls.append(player_url)
        
player_urls=pd.unique(player_urls)


# In[5]:

#Gets batting stats from a player's batting stats table
def get_batting_stats(name,stats,bat_dict):
    
    for result in stats:
        i=0
        bat_dict['names'].append(name)
        for r in result:
            if i==1:
                if r.string == None:
                    if len(bat_dict['years'])>0:
                        bat_dict['years'].append(bat_dict['years'][-1]+1)
                    else:
                        bat_dict['years'].append(0.0)
                else:
                    bat_dict['years'].append(int(r.string))
            if i==3:
                try:
                    bat_dict['ages'].append(float(r.string))
                except:
                    bat_dict['ages'].append(np.nan)
            if i==5:
                if r.string==None:
                    bat_dict['teams'].append(np.nan)
                else:
                    bat_dict['teams'].append(r.string)
            if i==7:
                if r.string==None:
                    bat_dict['leagues'].append(np.nan)
                else:
                    bat_dict['leagues'].append(r.string)
            if i==9:
                try:
                    bat_dict['games'].append(float(r.string))
                except:
                    bat_dict['games'].append(np.nan)
            if i==13:
                try:
                    bat_dict['at_bats'].append(float(r.string))
                except:
                    bat_dict['at_bats'].append(np.nan)
            if i==15:
                try:
                    bat_dict['runs'].append(float(r.string))
                except:
                    bat_dict['runs'].append(np.nan)
            if i==17:
                try:
                    bat_dict['hits'].append(float(r.string))
                except:
                    bat_dict['hits'].append(np.nan)
            if i==19:
                try:
                    bat_dict['doubles'].append(float(r.string))
                except:
                    bat_dict['doubles'].append(np.nan)
            if i==23:
                try:
                    bat_dict['home_runs'].append(float(r.string))
                except:
                    bat_dict['home_runs'].append(np.nan)
            if i==25:
                try:
                    bat_dict['rbis'].append(float(r.string))
                except:
                    bat_dict['rbis'].append(np.nan)
            if i==27: 
                try:
                    bat_dict['stolen_bases'].append(float(r.string))
                except:
                    bat_dict['stolen_bases'].append(np.nan)
            if i==31: 
                try:
                    bat_dict['walks'].append(float(r.string))
                except:
                    bat_dict['walks'].append(np.nan)
            if i==33: 
                try:
                    bat_dict['strikeouts'].append(float(r.string))
                except:
                    bat_dict['strikeouts'].append(np.nan)
            if i==35:
                try:
                    bat_dict['batting_averages'].append(float(r.string))
                except:
                    bat_dict['batting_averages'].append(np.nan)
            i+=1



# In[6]:

#Gets pitching stats from a player's pitching stats table
def get_pitching_stats(name,stats,pitch_dict):

    for result in stats:
        i=0
        pitch_dict['names'].append(name)
        for r in result:
            if i==1:
                if r.string == None:
                    if len(pitch_dict['years'])>0:
                        pitch_dict['years'].append(pitch_dict['years'][-1]+1)
                    else:
                        pitch_dict['years'].append(0.0)
                else:
                    pitch_dict['years'].append(int(r.string))
            if i==3:
                try:
                    pitch_dict['ages'].append(float(r.string))
                except:
                    pitch_dict['ages'].append(np.nan)
            if i==5:
                if r.string==None:
                    pitch_dict['teams'].append(np.nan)
                else:
                    pitch_dict['teams'].append(r.string)
            if i==7:
                if r.string==None:
                    pitch_dict['leagues'].append(np.nan)
                else:
                    pitch_dict['leagues'].append(r.string)
            if i==9:
                try:
                    pitch_dict['wins'].append(float(r.string))
                except:
                    pitch_dict['wins'].append(np.nan)
            if i==15:
                try:
                    pitch_dict['eras'].append(float(r.string))
                except:
                    pitch_dict['eras'].append(np.nan)
            if i==17:
                try:
                    pitch_dict['games'].append(float(r.string))
                except:
                    pitch_dict['games'].append(np.nan)
            if i==27:
                try:
                    pitch_dict['saves'].append(float(r.string))
                except:
                    pitch_dict['saves'].append(np.nan)
            if i==29:
                try:
                    pitch_dict['innings'].append(float(r.string))
                except:
                    pitch_dict['innings'].append(np.nan)
            if i==43:
                try:
                    pitch_dict['strikeouts'].append(float(r.string))
                except:
                    pitch_dict['strikeouts'].append(np.nan)
            if i==57:
                try:
                    pitch_dict['whips'].append(float(r.string))
                except:
                    pitch_dict['whips'].append(np.nan)
            if i==59:
                try:
                    pitch_dict['hits_9'].append(float(r.string))
                except:
                    pitch_dict['hits_9'].append(np.nan)
            if i==63:
                try:
                    pitch_dict['bb_9'].append(float(r.string))
                except:
                    pitch_dict['bb_9'].append(np.nan)
            if i==65:
                try:
                    pitch_dict['so_9'].append(float(r.string))
                except:
                    pitch_dict['so_9'].append(np.nan)
            if i==61:
                try:
                    pitch_dict['hr_9'].append(float(r.string))
                except:
                    pitch_dict['hr_9'].append(np.nan)

            i+=1


# In[7]:

#Initializes batting and pitching dictionaries
def initialize_batting(bat_dict):
    bat_dict['names']=[]
    bat_dict['years']=[]
    bat_dict['ages']=[]
    bat_dict['teams']=[]
    bat_dict['leagues']=[]
    bat_dict['games']=[]
    bat_dict['at_bats']=[]
    bat_dict['runs']=[]
    bat_dict['hits']=[]
    bat_dict['home_runs']=[]
    bat_dict['rbis']=[]
    bat_dict['stolen_bases']=[]
    bat_dict['batting_averages']=[]
    bat_dict['doubles']=[]
    bat_dict['walks']=[]
    bat_dict['strikeouts']=[]
    
def initialize_pitching(pitch_dict):
    pitch_dict['names']=[]
    pitch_dict['years']=[]
    pitch_dict['ages']=[]
    pitch_dict['teams']=[]
    pitch_dict['leagues']=[]
    pitch_dict['wins']=[]
    pitch_dict['eras']=[]
    pitch_dict['games']=[]
    pitch_dict['saves']=[]
    pitch_dict['strikeouts']=[]
    pitch_dict['whips']=[]
    pitch_dict['innings']=[]
    pitch_dict['hits_9']=[]
    pitch_dict['bb_9']=[]
    pitch_dict['so_9']=[]
    pitch_dict['hr_9']=[]


# In[8]:

#Initialize stats:

#Batting
batting_dict={}
initialize_batting(batting_dict)

#Pitching:
pitching_dict={}
initialize_pitching(pitching_dict)


# In[10]:

#Loops through every player url to extract player stats
buckets=list(np.arange(0,len(player_urls),800))
buckets.append(len(player_urls))
counter=0

#Loops through 800 urls at a time
for b in range(20,len(buckets)):
    start=buckets[b-1]
    end=buckets[b]
    print start,end
    
    #Initialize Batting
    #batting_dict={}
    #initialize_batting(batting_dict)

    #Initialize Pitching:
    pitching_dict={}
    initialize_pitching(pitching_dict)
    for url in player_urls[start:end]:

        try:
            request=urllib2.Request(url)
            page = urllib2.urlopen(request)
        except:
            continue

        #Reads contents of page
        try:
            content=page.read()
            soup=BeautifulSoup(content,'lxml')
        except:
            continue

        #get player name
        try:
            name=soup.find_all('strong')[0].string
        except:
            continue

        #get player position
        try:
            pos=soup.find_all('span',{'itemprop':'role'})[0].string
        except:
            continue

        #Pitchers
        if pos == 'Pitcher':
            try:
                stats=soup.find_all('tr', attrs={'id': re.compile(r'(pitching_standard.)(\d+)')})
                get_pitching_stats(name,stats,pitching_dict)
            except:
                continue

        #Batters
        #if pos != 'Pitcher':
        #    try:
        #        stats=soup.find_all('tr', attrs={'id': re.compile(r'(batting_standard.)(\d+)')})
        #        get_batting_stats(name,stats,batting_dict)
        #    except:
        #        continue
                
        counter+=1
        if np.mod(counter,100)==0:
            print "player #: " + str(counter)

    #Save to json file
    #with open(filepath+'batting_stats_'+str(start)+'-'+str(end)+'.json', 'w') as outfile:
    #    json.dump(batting_dict, outfile)
            
    with open(filepath+'pitching_stats_'+str(start)+'-'+str(end)+'.json', 'w') as outfile:
        json.dump(pitching_dict, outfile)


# In[ ]:



