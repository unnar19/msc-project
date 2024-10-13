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

def split_event_list(all_events_list):
    result_on = [a+b+c+d+e+f for a,b,c,d,e,f in zip(all_events_list[0],
                                                all_events_list[1],
                                                all_events_list[2],
                                                all_events_list[3],
                                                all_events_list[4],
                                                all_events_list[5])]

    result_off = [a+b+c+d+e+f for a,b,c,d,e,f in zip(all_events_list[6],
                                                    all_events_list[7],
                                                    all_events_list[8],
                                                    all_events_list[9],
                                                    all_events_list[10],
                                                    all_events_list[11])]
    return result_on, result_off


ACC_TIME_MS = 1000*WHOLE_TIME/BIN_COUNT

bin_stop_time = np.arange(ACC_TIME_MS, 1000*WHOLE_TIME+ACC_TIME_MS, ACC_TIME_MS)

mutant_time_moving = [[] for _ in range(BIN_COUNT)]
wildtype_time_moving = [[] for _ in range(BIN_COUNT)]

mutant_all_sprints = [[] for _ in range(BIN_COUNT)]
wildtype_all_sprints = [[] for _ in range(BIN_COUNT)]

ymax = 0
divider = 0

fig = plt.figure(figsize=(7, 4))

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
                    sprint_durations = [[] for _ in range(BIN_COUNT)]
                    temp_time_axis = [[] for _ in range(BIN_COUNT)]

                    for line in csvFile:
                        if not header:
                            start_time_us = int(line[1])
                            duration_us = int(line[0])

                            bin_index = int(start_time_us // (ACC_TIME_MS * 1000))
                            sprint_durations[bin_index].append(duration_us)

                        else:
                            header = False

                    temp_time_axis = [t/1000 - ACC_TIME_MS/2000 + 0.6*(ACC_TIME_MS/2000 * uniform(-1.0,1.0)) for t in bin_stop_time]
                    
                    time_moving = []
                    for i in range(BIN_COUNT):
                        time_moving.append(np.sum(sprint_durations[i])/1000000)
                        if time_moving[i] > ymax:
                            ymax = time_moving[i]

                    time_moving = np.nan_to_num(time_moving)
                    plt.scatter(temp_time_axis, time_moving, color=c, marker=".", alpha=0.1)

                    if is_mutant:
                        for bini in range(BIN_COUNT):
                            mutant_time_moving[bini].append(time_moving[bini])
                    else:
                        for bini in range(BIN_COUNT):
                            wildtype_time_moving[bini].append(time_moving[bini])



mt_on, mt_off = split_event_list(mutant_time_moving)
wt_on, wt_off = split_event_list(wildtype_time_moving)

print(f"Non parametric (Not normal distribution)")
print(f"p-value\tSignificant")

_, p_value = stats.mannwhitneyu(mt_on, wt_on)
print(f"{p_value:.6f}\t{bool(p_value < 0.05)}")
_, p_value = stats.mannwhitneyu(mt_off, wt_off)
print(f"{p_value:.6f}\t{bool(p_value < 0.05)}")

quit()


mt_avg = [0 for _ in range(BIN_COUNT)]
wt_avg = [0 for _ in range(BIN_COUNT)]
print(f"p-value\tSignificant")
for bin_i in range(BIN_COUNT):
    # _, p_value = stats.ttest_ind(np.nan_to_num(mutant_time_moving[bin_i]), np.nan_to_num(wildtype_time_moving[bin_i]))
    # p_value_one_tailed = p_value / 2
    # print(f"{p_value_one_tailed:.5f}\t{bool(p_value_one_tailed < 0.05)}")
    _, p_value = stats.mannwhitneyu(np.nan_to_num(mutant_time_moving[bin_i]), np.nan_to_num(wildtype_time_moving[bin_i]))
    print(f"{p_value:.5f}\t{bool(p_value < 0.05)}")
    mt_avg[bin_i] = np.mean(np.nan_to_num(mutant_time_moving[bin_i]))
    wt_avg[bin_i] = np.mean(np.nan_to_num(wildtype_time_moving[bin_i]))

plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,mt_avg, color=COLORS[0])
plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,wt_avg, color=COLORS[1])

mutant1 = Line2D([0], [0], color='b')
mutant2 = Line2D([0], [0], marker=".", linestyle="none", color='b', alpha=0.2)
wildtype1 = Line2D([0], [0], color='r')
wildtype2 = Line2D([0], [0], marker=".", linestyle="none", color='r', alpha=0.2)
black = Line2D([0], [0], color='k')
plt.title("Time moving for each fish with accumulation time ~30s")
plt.legend([(mutant1,mutant2), (wildtype1,wildtype2), black] ,["$\it{adgrl3.1}$ mutants","wildtype","Light off"],loc="upper right")
plt.xlim([0,357])
plt.ylabel("Time moving [s]")
plt.xlabel("Time [s]")
plt.ylim(bottom=0)
plt.grid(axis="y")
plt.vlines(SPIKE_TIME, 0, ymax*1.1, "k")
time_step_axis = np.linspace(0,357,100000)
plt.fill_between(time_step_axis, 0, ymax*1.1, where=(np.array(time_step_axis) >= SPIKE_TIME) & (np.array(time_step_axis) <= 357), color='gray', alpha=0.2)
plt.ylim([0, ymax*0.5])
plt.show()
fig.savefig(f"graphics/graphs/plot-time-moving-all-fish.png")

mutant_on  = sum(mutant_time_moving[:6], [])
mutant_off = sum(mutant_time_moving[6:], [])

wildtype_on  = sum(wildtype_time_moving[:6], [])
wildtype_off = sum(wildtype_time_moving[6:], [])


fig1 = plt.figure(figsize=(5, 4))
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(mutant_on, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title("TM Q-Q $\it{adgrl3.1}$ on")
plt.legend(["$\it{adgrl3.1}$ mutants"])
plt.tight_layout()
fig1.savefig(f"graphics/graphs/qqplots/tm_qq_on.png_mt.png")
plt.show()

fig2 = plt.figure(figsize=(5, 4))
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(wildtype_on, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title("TM Q-Q wildtype on")
plt.legend(["wildtype"])
plt.tight_layout()
fig2.savefig(f"graphics/graphs/qqplots/tm_qq_on_wt.png")
plt.show()

fig3 = plt.figure(figsize=(5, 4))
ax = plt.axes()
ax.set_facecolor('#e6e6e6')
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(mutant_off, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title("TM Q-Q $\it{adgrl3.1}$ off")
plt.legend(["$\it{adgrl3.1}$ mutants"])
plt.tight_layout()
fig3.savefig(f"graphics/graphs/qqplots/tm_qq_off_mt.png")
plt.show()

fig4 = plt.figure(figsize=(5, 4))
ax = plt.axes()
ax.set_facecolor('#e6e6e6')
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(wildtype_off, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title("TM Q-Q wildtype off")
plt.legend(["wildtype"])
plt.tight_layout()
fig4.savefig(f"graphics/graphs/qqplots/tm_qq_off_wt.png")
plt.show()