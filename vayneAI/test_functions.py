"""
from matplotlib import pyplot as plt
import numpy as np
import math

x = np.arange(0, 3440)
y = []
for i in x:
    #i2 = i * 0.0075
    #y.append((100 * (1 / (1 + math.exp(-2 * (i2 - 3))) - 1 / (1 + math.exp(-3 * (i2 - 6)))))/94.78)
    i2 = i
    y.append(1-(0.5 * math.log((i2+1), 5000)) - 0.5*(80/100))

print(max(y))

plt.plot(x,y)
plt.show()
"""
import matplotlib.pyplot as plt
import copy
import numpy as np
import math


pc_e_dim = 3
pc_e_var = 1
pc_i_dim = 3
pc_i_var = 1

inhib_factor = 0.002

pc_e = []
pc_i = []

pc_dim_xy = 10

def norm2d(var, x, center):
    return 1.0 / (var * math.sqrt(2 * np.pi)) * math.exp((-math.pow(x - center, 2))/ (2.0 * var * var))

# Build the excite pdf
total = 0
center = pc_e_dim / 2
for i in range(0, pc_e_dim):
    temp = norm2d(pc_e_var, i, center)
    pc_e.append(temp)
total = sum(pc_e)
for i in range(0, pc_e_dim):
        pc_e[i] = pc_e[i] / total

# Build inhibit pdf
total = 0
center = pc_i_dim / 2
for i in range(0, pc_i_dim):
    temp = norm2d(pc_i_var, i, center)
    pc_i.append(temp)
total = np.sum(pc_i)
for i in range(0, pc_i_dim):
        pc_i[i] = pc_i[i] / total

x1 = np.arange(0, pc_e_dim, 1)
x2 = np.arange(0, pc_i_dim, 1)
fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)
ax1.plot(x1, pc_e)
ax2.plot(x2, pc_i)
ax1.set_title('Excitation')
ax2.set_title('Inhibition')

plt.show()

def excite(posecells):
    posecells_new = [0] * pc_dim_xy
    for i in range(0, pc_dim_xy):
        cur_pc_value = posecells[i]
        if cur_pc_value > 0:
            excite_index = 0
            for k in range(i, i+pc_e_dim):
                if k >= 0 and k < pc_dim_xy:
                    xw = k-(pc_e_dim/2)
                    posecells_new[xw] = posecells_new[xw] + cur_pc_value * pc_e[excite_index]
                    excite_index += 1
    posecells = posecells_new
    return posecells
def local_inhibit(posecells):
    posecells_new = [0] * pc_dim_xy
    for i in range(0, pc_dim_xy):
        cur_pc_value = posecells[i]
        if cur_pc_value > 0:
            inhibit_index = 0
            for k in range(i, i+pc_i_dim):
                if k >= 0 and k < pc_dim_xy:
                    xw = k-(pc_i_dim/2)
                    posecells_new[xw] = posecells_new[xw] + cur_pc_value * pc_i[inhibit_index]
                    inhibit_index += 1
    posecells = posecells_new
    return posecells
def global_inhibit(posecells_i):
    inp = copy.copy(posecells_i)
    posecells_new = [0] * pc_dim_xy
    for i in range(0, pc_dim_xy):
        posecells_new[i] = inp[i] - inhib_factor
        if posecells_new[i] < inhib_factor:
            posecells_new[i] = 0
    return posecells_new

def normalize(posecells_b):
    inp = copy.copy(posecells_b)
    posecells_new = [0] * pc_dim_xy
    s = sum(inp)
    for i in range(0, pc_dim_xy):
        posecells_new[i] = inp[i] / s
    return posecells_new

def draw(inp, excited, local_inhibit,subtracted, inhibit, normalized):
    x = np.arange(0, pc_dim_xy, 1)
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax1.plot(x, inp)
    ax1.set_ylim([-0.5, 1.2])
    ax1.set_title("input")
    ax2.plot(x, excited)
    ax2.set_title("excited")
    ax3.plot(x, local_inhibit)
    ax3.set_title("local_inhibit")
    ax4.plot(x, subtracted)
    ax4.set_title("subtracted")
    ax5.plot(x, inhibit)
    ax5.set_title("inhibit")
    ax6.plot(x, normalized)
    ax6.set_title("normalized")
    plt.pause(0.01)

posecells = [0] * pc_dim_xy
posecells[pc_dim_xy/2-1] = 1

exc = True
loc_inh = True
glo_inh = True
norm = True
inject = False

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6,sharey=True)
injection_rate = 10
for i in range(0, 1000):

    if inject:
        if i%injection_rate == 0:
            posecells[10] += 0.3
            if injection_rate > 1:
                injection_rate -= 1
    print("iteration ", i)
    if exc:
        posecells_excited = copy.copy(excite(posecells))
    else:
        posecells_excited = copy.copy(posecells)
    if loc_inh:
        posecells_local_inhib = copy.copy(local_inhibit(copy.copy(posecells_excited)))
    else:
        posecells_local_inhib = [0] * pc_dim_xy
    posecells_temp = copy.copy([k-j for k,j in zip(posecells_excited, posecells_local_inhib)])


    if glo_inh:
        posecells_inhib = copy.copy(global_inhibit(posecells_temp))
    else:
        posecells_inhib = copy.copy(posecells_temp)
    if norm:
        posecells_norm = copy.copy(normalize(posecells_inhib))
    else:
        posecells_norm = copy.copy(posecells_inhib)
    print(posecells_temp)
    print(posecells_inhib)
    print(posecells_norm)
    draw(posecells, posecells_excited, posecells_local_inhib, posecells_temp, posecells_inhib, posecells_norm)
    posecells = posecells_norm
    print(sum(posecells))
