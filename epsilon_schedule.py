EXPLORATION_FRAMES = 1000000
EPSILON_START = 1
EPSILON_END = 0.1
glob_frame = 0

def get_epsilon():
    # Linear Annealing
    if glob_frame < EXPLORATION_FRAMES:
        return EPSILON_END + (EXPLORATION_FRAMES-glob_frame)*(EPSILON_START-EPSILON_END)/EXPLORATION_FRAMES
    elif glob_frame < 2*EXPLORATION_FRAMES:
        return EPSILON_END/10 + (2*EXPLORATION_FRAMES-glob_frame)*(EPSILON_END-EPSILON_END/10)/EXPLORATION_FRAMES
    else:
        return EPSILON_END/10

eps = []
for i in range(5*EXPLORATION_FRAMES):
    e = get_epsilon()
    # print(e)
    eps.append(e)
    glob_frame+=1

import matplotlib.pyplot as plt
plt.plot(list(range(5*EXPLORATION_FRAMES)), eps)
plt.show()