import csv
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from matplotlib.lines import Line2D
import scipy.stats as stats

BASE_PATH = "adgrl3-sprints/"
COLORS = ["#0000FF", "#FF0000"]
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

DATES = ["2024-04-21", "2024-04-22"]
EXPERIMENTS = [[2,4],[2,4,6,8]]
MUTANTS_ON_LEFT = [[1,0],[0,1,0,1]]

WHOLE_TIME = 357
SPIKE_TIME = 178
BIN_COUNT = 12

ACC_TIME_MS = 1000*WHOLE_TIME/BIN_COUNT

bin_stop_time = np.arange(ACC_TIME_MS, 1000*WHOLE_TIME+ACC_TIME_MS, ACC_TIME_MS)

mutant_sprint_count = np.array([0 for _ in range(BIN_COUNT)])
mutant_sprint_dur = [[] for _ in range(BIN_COUNT)]
wildtype_sprint_count = np.array([0 for _ in range(BIN_COUNT)])
wildtype_sprint_dur = [[] for _ in range(BIN_COUNT)]

mutant_all_sprints = [[] for _ in range(BIN_COUNT)]
wildtype_all_sprints = [[] for _ in range(BIN_COUNT)]

ymax = 0
divider = 0

fig = plt.figure(figsize=(7, 4))

for date_i, date in enumerate(DATES):
    for exp_i, exp in enumerate(EXPERIMENTS[date_i]):

        mutants_on_left = bool(MUTANTS_ON_LEFT[date_i][exp_i])

        for alph in ALPH_MAP:
            for num in [0,13]:
                COMP_ID = f"{alph}{num}"

                with open(BASE_PATH + date + f"/ex{exp}/compartments/{COMP_ID}.csv", mode='r') as file:
                    csvFile = csv.reader(file)
                    header = True
                    sprint_count = [0 for _ in range(BIN_COUNT)]

                    for line in csvFile:
                        if not header:
                            start_time_us = int(line[1])
                            duration_us = int(line[0])

                            bin_index = int(start_time_us // (ACC_TIME_MS * 1000))
                            sprint_count[bin_index] += 1

                        else:
                            header = False

                    temp_time_axis = [t/1000 - ACC_TIME_MS/2000 + 0.6*(ACC_TIME_MS/2000 * uniform(-1.0,1.0)) for t in bin_stop_time]
                    plt.scatter(temp_time_axis, sprint_count, color='k', marker=".", alpha=0.1)


                    divider += 1
                    mutant_sprint_count += sprint_count
                    #mutant_sprint_dur[bin_index].append(duration_us)
                    for bini in range(BIN_COUNT):
                        mutant_all_sprints[bini].append(sprint_count[bini])


                highest = max(sprint_count)
                if highest > ymax:
                    ymax = highest
  

for bins in mutant_sprint_count:
    print(bins)

plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,mutant_sprint_count/divider, color='gray')

mutant1 = Line2D([0], [0], color='gray')
mutant2 = Line2D([0], [0], marker=".", linestyle="none", color='black', alpha=0.2)
black = Line2D([0], [0], color='k')
plt.title("Sprint count for all fish with accumulation time ~30s")
plt.legend([(mutant1,mutant2), black] ,["Noise","Light off"],loc='upper right')
plt.xlim([0,357])
plt.ylabel("Sprint count [#]")
plt.xlabel("Time [s]")
plt.ylim(bottom=0)
plt.grid(axis="y")
plt.vlines(SPIKE_TIME, 0, ymax*1.1, "k")
time_step_axis = np.linspace(0,357,100000)
plt.fill_between(time_step_axis, 0, ymax*1.1, where=(np.array(time_step_axis) >= SPIKE_TIME) & (np.array(time_step_axis) <= 357), color='gray', alpha=0.2)
plt.ylim([0, ymax*0.6])
fig.savefig(f"graphics/graphs/plot-sprint-count-all-noise.png")
plt.show()


# mutant_on  = sum(mutant_all_sprints[:6], [])
# mutant_off = sum(mutant_all_sprints[6:], [])

# wildtype_on  = sum(wildtype_all_sprints[:6], [])
# wildtype_off = sum(wildtype_all_sprints[6:], [])


# fig1 = plt.figure(figsize=(5, 4))
# plt.rcParams['font.size'] = 16
# plt.grid(axis="y")
# stats.probplot(mutant_on, dist="norm", plot=plt)
# plt.gca().get_lines()[0].set_color('blue')
# plt.gca().get_lines()[1].set_color('k')
# plt.title("SC Q-Q $\it{adgrl3.1}$ on")
# plt.legend(["$\it{adgrl3.1}$ mutants"],loc='upper left')
# plt.tight_layout()
# fig1.savefig(f"graphics/graphs/qqplots/sc_qq_on_mt.png")
# plt.show()

# fig2 = plt.figure(figsize=(5, 4))
# plt.rcParams['font.size'] = 16
# plt.grid(axis="y")
# stats.probplot(wildtype_on, dist="norm", plot=plt)
# plt.gca().get_lines()[0].set_color('red')
# plt.gca().get_lines()[1].set_color('k')
# plt.title(f"SC Q-Q wildtype on")
# plt.legend(["wildtype"],loc='upper left')
# plt.tight_layout()
# fig2.savefig(f"graphics/graphs/qqplots/sc_qq_on_wt.png")
# plt.show()

# fig3 = plt.figure(figsize=(5, 4))
# ax = plt.axes()
# ax.set_facecolor('#e6e6e6')
# plt.rcParams['font.size'] = 16
# plt.grid(axis="y")
# stats.probplot(mutant_off, dist="norm", plot=plt)
# plt.gca().get_lines()[0].set_color('blue')
# plt.gca().get_lines()[1].set_color('k')
# plt.title("SC Q-Q $\it{adgrl3.1}$ off")
# plt.legend(["$\it{adgrl3.1}$ mutants"],loc='upper left')
# plt.tight_layout()
# fig3.savefig(f"graphics/graphs/qqplots/sc_qq_off_mt.png")
# plt.show()


# fig4 = plt.figure(figsize=(5, 4))
# ax = plt.axes()
# ax.set_facecolor('#e6e6e6')
# plt.rcParams['font.size'] = 16
# plt.grid(axis="y")
# stats.probplot(wildtype_off, dist="norm", plot=plt)
# plt.gca().get_lines()[0].set_color('red')
# plt.gca().get_lines()[1].set_color('k')
# plt.title(f"SC Q-Q wildtype off")
# plt.legend(["wildtype"],loc='upper left')
# plt.tight_layout()
# fig4.savefig(f"graphics/graphs/qqplots/sc_qq_off_wt.png")
# plt.show()