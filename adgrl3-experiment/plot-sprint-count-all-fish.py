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

fig = plt.figure(figsize=(8, 4))

for date_i, date in enumerate(DATES):
    for exp_i, exp in enumerate(EXPERIMENTS[date_i]):

        mutants_on_left = bool(MUTANTS_ON_LEFT[date_i][exp_i])

        for alph in ALPH_MAP:
            for num in range(1,13):
                COMP_ID = f"{alph}{num}"

                if (mutants_on_left == True and num < 7) or (mutants_on_left == False and num >= 7):
                    is_mutant = True
                    c = COLORS[0]
                else:
                    is_mutant = False
                    c = COLORS[1]

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
                    plt.scatter(temp_time_axis, sprint_count, color=c, marker=".", alpha=0.1)

                    if is_mutant:
                        divider += 1
                        mutant_sprint_count += sprint_count
                        #mutant_sprint_dur[bin_index].append(duration_us)
                        for bini in range(BIN_COUNT):
                            mutant_all_sprints[bini].append(sprint_count[bini])
                    else:
                        wildtype_sprint_count += sprint_count
                        #wildtype_sprint_dur[bin_index].append(duration_us)
                        for bini in range(BIN_COUNT):
                            wildtype_all_sprints[bini].append(sprint_count[bini])


                highest = max(sprint_count)
                if highest > ymax:
                    ymax = highest

print(f"p-value\tSignificant")
for bin_i in range(BIN_COUNT):
    # _, p_value = stats.ttest_ind(mutant_all_sprints[bin_i], wildtype_all_sprints[bin_i])
    # p_value_one_tailed = p_value / 2
    # print(f"{p_value_one_tailed:.5f}\t{bool(p_value_one_tailed < 0.05)}")
    _, p_value = stats.mannwhitneyu(mutant_all_sprints[bin_i], wildtype_all_sprints[bin_i])
    print(f"{p_value:.5f}\t{bool(p_value < 0.05)}")
    

plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,mutant_sprint_count/divider, color=COLORS[0])
plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,wildtype_sprint_count/divider, color=COLORS[1])

mutant1 = Line2D([0], [0], color='b')
mutant2 = Line2D([0], [0], marker=".", linestyle="none", color='b', alpha=0.2)
wildtype1 = Line2D([0], [0], color='r')
wildtype2 = Line2D([0], [0], marker=".", linestyle="none", color='r', alpha=0.2)
black = Line2D([0], [0], color='k')
plt.title("Sprint count for all fish with accumulation time ~30s")
plt.legend([(mutant1,mutant2), (wildtype1,wildtype2), black] ,["adgrl3.1","WT","Light off"])
plt.xlim([0,357])
plt.ylabel("Sprint count [#]")
plt.xlabel("Time [s]")
plt.ylim(bottom=0)
plt.grid(axis="y")
plt.vlines(SPIKE_TIME, 0, ymax*1.1, "k")
time_step_axis = np.linspace(0,357,100000)
plt.fill_between(time_step_axis, 0, ymax*1.1, where=(np.array(time_step_axis) >= SPIKE_TIME) & (np.array(time_step_axis) <= 357), color='gray', alpha=0.2)
plt.ylim([0, ymax*1.1])
plt.show()

# fig.savefig(f"graphics/graphs/plot-sprint-count-all-fish.png")

mutant_on  = sum(mutant_all_sprints[:6], [])
mutant_off = sum(mutant_all_sprints[6:], [])

wildtype_on  = sum(wildtype_all_sprints[:6], [])
wildtype_off = sum(wildtype_all_sprints[6:], [])


plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
stats.probplot(mutant_on, dist="norm", plot=plt)
plt.title(f"Q-Q Plot for Mutant Data (Lights on)")

plt.subplot(1, 2, 2)
stats.probplot(wildtype_on, dist="norm", plot=plt)
plt.title(f"Q-Q Plot for Wildtype Data (Lights on)")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
stats.probplot(mutant_off, dist="norm", plot=plt)
plt.title(f"Q-Q Plot for Mutant Data  (Lights off)")

plt.subplot(1, 2, 2)
stats.probplot(wildtype_off, dist="norm", plot=plt)
plt.title(f"Q-Q Plot for Wildtype Data (Lights off)")

plt.tight_layout()
plt.show()