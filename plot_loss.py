# This script is used to plot the loss during training with darknet


import matplotlib.pyplot as plt
import numpy as np

logfile_path = "weights/05_02_synthetic_LeagueAI/train.log"

with open(logfile_path, 'r') as f:
    data=[]
    for l in f:
        if "avg" in l:
            data.append(l)
avg1 = []
avg2 = []
for i in data:
    i = i.replace(",", "")
    s = i.split(" ")
    avg1.append(float(s[1]))
    avg2.append(float(s[2]))

print(avg1)

x = np.arange(len(avg1))

fig = plt.figure()
a = plt.plot(x, avg1, color='r')
plt.title('Average Loss')
plt.xlabel('Episodes (batch size = 64)')
plt.ylabel('Loss')
plt.xlim([0,40000])
plt.ylim([0, 300])
plt.grid()
plt.show()
