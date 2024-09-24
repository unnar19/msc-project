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

mutant_sprint_dur = [[] for _ in range(BIN_COUNT)]
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
                    sprint_durations = [[] for _ in range(BIN_COUNT)]
                    temp_time_axis = [[] for _ in range(BIN_COUNT)]

                    for line in csvFile:
                        if not header:
                            start_time_us = int(line[1])
                            duration_us = int(line[0])

                            if duration_us > ymax:
                                ymax = duration_us

                            bin_index = int(start_time_us // (ACC_TIME_MS * 1000))
                            sprint_durations[bin_index].append(duration_us)

                            #temp_time_axis[bin_index].append((bin_stop_time[bin_index])/1000 - ACC_TIME_MS/2000 + 0.6*(ACC_TIME_MS/2000 * uniform(-1.0,1.0)))

                        else:
                            header = False

                    temp_time_axis = [t/1000 - ACC_TIME_MS/2000 + 0.6*(ACC_TIME_MS/2000 * uniform(-1.0,1.0)) for t in bin_stop_time]

                    
                    mean_sprint_dur = []
                    for i in range(BIN_COUNT):
                        mean_sprint_dur.append(np.mean(sprint_durations[i]))

                    mean_sprint_dur = np.nan_to_num(mean_sprint_dur)

                    temp_time_axis = np.array(temp_time_axis)
                    mean_sprint_dur = np.array(mean_sprint_dur)

                    temp1 = temp_time_axis[np.nonzero(mean_sprint_dur)]
                    temp2 = mean_sprint_dur[np.nonzero(mean_sprint_dur)]

                    plt.scatter(temp1, temp2, color=c, marker=".", alpha=0.1)

                    if is_mutant:
                        #mutant_sprint_count += sprint_durations
                        #mutant_sprint_dur[bin_index].append(duration_us)
                        for bini in range(BIN_COUNT):
                            mutant_sprint_dur[bini].append(mean_sprint_dur[bini])
                    else:
                        #wildtype_sprint_count += sprint_count
                        #wildtype_sprint_dur[bin_index].append(duration_us)
                        for bini in range(BIN_COUNT):
                            wildtype_sprint_dur[bini].append(mean_sprint_dur[bini])

mt_avg = [0 for _ in range(BIN_COUNT)]
wt_avg = [0 for _ in range(BIN_COUNT)]
print(f"p-value\tSignificant")
for bin_i in range(BIN_COUNT):
    # _, p_value = stats.ttest_ind(np.nan_to_num(mutant_sprint_dur[bin_i]), np.nan_to_num(wildtype_sprint_dur[bin_i]))
    # p_value_one_tailed = p_value / 2
    # print(f"{p_value_one_tailed:.5f}\t{bool(p_value_one_tailed < 0.05)}")
    mutant_sprint_dur[bin_i] = np.array(mutant_sprint_dur[bin_i])
    mutant_sprint_dur[bin_i] = mutant_sprint_dur[bin_i][~np.isnan(mutant_sprint_dur[bin_i])]
    mutant_sprint_dur[bin_i] = mutant_sprint_dur[bin_i][np.nonzero(mutant_sprint_dur[bin_i])]
    mutant_sprint_dur[bin_i] = list(mutant_sprint_dur[bin_i])

    wildtype_sprint_dur[bin_i] = np.array(wildtype_sprint_dur[bin_i])
    wildtype_sprint_dur[bin_i] = wildtype_sprint_dur[bin_i][~np.isnan(wildtype_sprint_dur[bin_i])]
    wildtype_sprint_dur[bin_i] = wildtype_sprint_dur[bin_i][np.nonzero(wildtype_sprint_dur[bin_i])]
    wildtype_sprint_dur[bin_i] = list(wildtype_sprint_dur[bin_i])

    _, p_value = stats.mannwhitneyu(np.nan_to_num(mutant_sprint_dur[bin_i]), np.nan_to_num(wildtype_sprint_dur[bin_i]))
    print(f"{p_value:.5f}\t{bool(p_value < 0.05)}")
    mt_avg[bin_i] = np.mean(np.nan_to_num(mutant_sprint_dur[bin_i]))
    wt_avg[bin_i] = np.mean(np.nan_to_num(wildtype_sprint_dur[bin_i]))

plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,mt_avg, color=COLORS[0])
plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,wt_avg, color=COLORS[1])

mutant1 = Line2D([0], [0], color='b')
mutant2 = Line2D([0], [0], marker=".", linestyle="none", color='b', alpha=0.2)
wildtype1 = Line2D([0], [0], color='r')
wildtype2 = Line2D([0], [0], marker=".", linestyle="none", color='r', alpha=0.2)
black = Line2D([0], [0], color='k')
plt.title("Mean sprint duration for each fish with accumulation time ~30s")
plt.legend([(mutant1,mutant2), (wildtype1,wildtype2), black] ,["adgrl3.1","WT","Light off"])
plt.xlim([0,357])
plt.ylabel("Mean sprint duration [µs]")
plt.xlabel("Time [s]")
plt.ylim(bottom=0)
plt.grid(axis="y")
plt.vlines(SPIKE_TIME, 0, ymax*1.1, "k")
time_step_axis = np.linspace(0,357,100000)
plt.fill_between(time_step_axis, 0, ymax*1.1, where=(np.array(time_step_axis) >= SPIKE_TIME) & (np.array(time_step_axis) <= 357), color='gray', alpha=0.2)
plt.ylim([0, ymax*0.15])
plt.show()

# fig.savefig(f"graphics/graphs/plot-sprint-duration-all-fish.png")



mutant_on  = sum(mutant_sprint_dur[:6], [])
mutant_off = sum(mutant_sprint_dur[6:], [])

wildtype_on  = sum(wildtype_sprint_dur[:6], [])
wildtype_off = sum(wildtype_sprint_dur[6:], [])


plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
stats.probplot(mutant_on, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title(f"Mean sprint duration Q-Q Plot for ADGRL3.1 (Lights on)")
fig.savefig(f"graphics/graphs/qqplots/sd_qq_adgrl_on.png")

plt.subplot(1, 2, 2)
stats.probplot(wildtype_on, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title(f"Mean sprint duration Q-Q Plot for WT (Lights on)")
fig.savefig(f"graphics/graphs/qqplots/sd_qq_wt_on.png")

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 8))

plt.subplot(1, 2, 1)
stats.probplot(mutant_off, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title(f"Mean sprint duration Q-Q Plot for ADGRL3.1 (Lights off)")
fig.savefig(f"graphics/graphs/qqplots/sd_qq_adgrl_off.png")

plt.subplot(1, 2, 2)
stats.probplot(wildtype_off, dist="norm", plot=plt)
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title(f"Mean sprint duration Q-Q Plot for WT (Lights off)")
fig.savefig(f"graphics/graphs/qqplots/sd_qq_wt_off.png")

plt.tight_layout()
plt.show()