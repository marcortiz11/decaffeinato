import csv
import h5py
import numpy as np


equip = {}
with open('APRIL_offense_team_stats.csv') as file:
	reader = csv.reader(file,delimiter=',')
	for row in reader:
		equip[row[0]] = [float(i) * 0.001 for i in row[1:]]
    
equip_defense = {}


with open('APRIL_defense_team_stats.csv') as file:
	reader = csv.reader(file,delimiter=',')
	for row in reader:
		equip_defense[row[0]] = [float(i) * 0.001 for i in row[1:]]

team_scores = {}
with open('team_stats.csv') as file:
    reader = csv.reader(file,delimiter=',')
    for row in reader:
        team_scores[row[1]] = [float(i) for i in row[4:]]
  

results = []
stats = []

with open('players_stats.csv') as file:
    reader = csv.reader(file,delimiter=',')
    for row in reader:
        equip[row[1]] = equip[row[1]] + [float(i) * 0.001 for i in row[2:]]


with open('APRIL_scores.csv') as file: 
    reader = csv.reader(file,delimiter=',')
    for row in reader:
        stats = stats + [equip[row[2]] + team_scores[row[2]] + equip[row[4]] +  team_scores[row[4]]]
        if float(row[3]) > float(row[5]):
            results = results + [1]
        else: 
            results = results + [0]
       
print(len(stats[0]))

f = h5py.File('train.h5', 'w') 
# 1200 data, each is a 128-dim vector
f.create_dataset('data', (len(stats),len(stats[0])), dtype='f8')
# Data's labels, each is a 4-dim vector
f.create_dataset('label', (len(stats),1), dtype='f4')

# Fill in something with fixed pattern
# Regularize values to between 0 and 1, or SigmoidCrossEntropyLoss will not work
for i in range(len(stats)):
    f['data'][i] = stats[i]
    f['label'][i] = results[i]
  
f.close()

stats = []
results = []

with open('APRIL_scores_test.csv') as file: 
    reader = csv.reader(file,delimiter=',')
    for row in reader:
        stats = stats + [equip[row[2]] + team_scores[row[2]] + equip[row[4]] +  team_scores[row[4]]]
        if float(row[3]) > float(row[5]):
            results = results + [1]
        else: 
            results = results + [0]


f = h5py.File('test.h5', 'w') 
# 1200 data, each is a 128-dim vector
f.create_dataset('data', (len(stats),len(stats[0])), dtype='f8')
# Data's labels, each is a 4-dim vector
f.create_dataset('label', (len(stats),1), dtype='f4')

# Fill in something with fixed pattern
# Regularize values to between 0 and 1, or SigmoidCrossEntropyLoss will not work
for i in range(len(stats)):
    f['data'][i] = stats[i]
    f['label'][i] = results[i]
    

f.close()
