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

mutant_sprint_dur_on = [[] for _ in range(BIN_COUNT)]
mutant_sprint_dur_off = [[] for _ in range(BIN_COUNT)]
wildtype_sprint_dur_on = [[] for _ in range(BIN_COUNT)]
wildtype_sprint_dur_off = [[] for _ in range(BIN_COUNT)]

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
                            if bini < 6:
                                mutant_sprint_dur_on[bini].append(mean_sprint_dur[bini])
                            else:
                                mutant_sprint_dur_off[bini].append(mean_sprint_dur[bini])
                    else:
                        #wildtype_sprint_count += sprint_count
                        #wildtype_sprint_dur[bin_index].append(duration_us)
                        for bini in range(BIN_COUNT):
                            if bini < 6:
                                wildtype_sprint_dur_on[bini].append(mean_sprint_dur[bini])
                            else:
                                wildtype_sprint_dur_off[bini].append(mean_sprint_dur[bini])

mt_avg = [0 for _ in range(BIN_COUNT)]
wt_avg = [0 for _ in range(BIN_COUNT)]
print(f"p-value\tSignificant")
for bin_i in range(BIN_COUNT):
    
    if bini < 6:
        mutant_sprint_dur_on[bin_i] = np.array(mutant_sprint_dur_on[bin_i])
        mutant_sprint_dur_on[bin_i] = mutant_sprint_dur_on[bin_i][~np.isnan(mutant_sprint_dur_on[bin_i])]
        mutant_sprint_dur_on[bin_i] = mutant_sprint_dur_on[bin_i][np.nonzero(mutant_sprint_dur_on[bin_i])]
        mutant_sprint_dur_on[bin_i] = list(mutant_sprint_dur_on[bin_i])
    else:
        mutant_sprint_dur_off[bin_i] = np.array(mutant_sprint_dur_off[bin_i])
        mutant_sprint_dur_off[bin_i] = mutant_sprint_dur_off[bin_i][~np.isnan(mutant_sprint_dur_off[bin_i])]
        mutant_sprint_dur_off[bin_i] = mutant_sprint_dur_off[bin_i][np.nonzero(mutant_sprint_dur_off[bin_i])]
        mutant_sprint_dur_off[bin_i] = list(mutant_sprint_dur_off[bin_i])

    if bini < 6:
        wildtype_sprint_dur_on[bin_i] = np.array(wildtype_sprint_dur_on[bin_i])
        wildtype_sprint_dur_on[bin_i] = wildtype_sprint_dur_on[bin_i][~np.isnan(wildtype_sprint_dur_on[bin_i])]
        wildtype_sprint_dur_on[bin_i] = wildtype_sprint_dur_on[bin_i][np.nonzero(wildtype_sprint_dur_on[bin_i])]
        wildtype_sprint_dur_on[bin_i] = list(wildtype_sprint_dur_on[bin_i])
    else:
        wildtype_sprint_dur_off[bin_i] = np.array(wildtype_sprint_dur_off[bin_i])
        wildtype_sprint_dur_off[bin_i] = wildtype_sprint_dur_off[bin_i][~np.isnan(wildtype_sprint_dur_off[bin_i])]
        wildtype_sprint_dur_off[bin_i] = wildtype_sprint_dur_off[bin_i][np.nonzero(wildtype_sprint_dur_off[bin_i])]
        wildtype_sprint_dur_off[bin_i] = list(wildtype_sprint_dur_off[bin_i])

    # _, p_value = stats.ttest_ind(np.nan_to_num(mutant_sprint_dur[bin_i]), np.nan_to_num(wildtype_sprint_dur[bin_i]))
    # p_value_one_tailed = p_value / 2
    # print(f"{p_value_one_tailed:.5f}\t{bool(p_value_one_tailed < 0.05)}")
    # # _, p_value = stats.mannwhitneyu(np.nan_to_num(mutant_sprint_dur[bin_i]), np.nan_to_num(wildtype_sprint_dur[bin_i]))
    # # print(f"{p_value:.5f}\t{bool(p_value < 0.05)}")
    if bini < 6:
        mt_avg[bin_i] = np.mean(np.nan_to_num(mutant_sprint_dur_on[bin_i]))
        wt_avg[bin_i] = np.mean(np.nan_to_num(wildtype_sprint_dur_on[bin_i]))
    else:
        mt_avg[bin_i] = np.mean(np.nan_to_num(mutant_sprint_dur_off[bin_i]))
        wt_avg[bin_i] = np.mean(np.nan_to_num(wildtype_sprint_dur_off[bin_i]))

mt_on, mt_off = split_event_list(sum([mutant_sprint_dur_on[:6],mutant_sprint_dur_off[6:]],[]))
wt_on, wt_off = split_event_list(sum([wildtype_sprint_dur_on[:6],wildtype_sprint_dur_off[6:]],[]))


print(len(mt_on), len(wt_on))
print(len(mt_off), len(wt_off))
# print(mutant_sprint_dur_off)
# print(wildtype_sprint_dur_off)

print(np.var(mt_on, ddof=1)/np.var(wt_on, ddof=1))
print(np.var(wt_off, ddof=1)/np.var(mt_off, ddof=1))

print(f"Parametric (Normal distribution)")
print(f"p-value\tSignificant")

_, p_value = stats.ttest_ind(mt_on, wt_on,equal_var=False)
# _, p_value = stats.mannwhitneyu(mt_on, wt_on)
print(f"{p_value:.6f}\t{bool(p_value < 0.05)}")
_, p_value = stats.ttest_ind(mt_off, wt_off,equal_var=False)
# _, p_value = stats.mannwhitneyu(mt_off, wt_off)
print(f"{p_value:.6f}\t{bool(p_value < 0.05)}")

stat, p_value = stats.bartlett(mt_off, wt_off)
print(p_value)

quit()

plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,mt_avg, color=COLORS[0])
plt.plot((bin_stop_time - ACC_TIME_MS/2)/1000,wt_avg, color=COLORS[1])

mutant1 = Line2D([0], [0], color='b')
mutant2 = Line2D([0], [0], marker=".", linestyle="none", color='b', alpha=0.2)
wildtype1 = Line2D([0], [0], color='r')
wildtype2 = Line2D([0], [0], marker=".", linestyle="none", color='r', alpha=0.2)
black = Line2D([0], [0], color='k')
plt.title("Mean sprint duration for each fish with accumulation time ~30s")
plt.legend([(mutant1,mutant2), (wildtype1,wildtype2), black] ,["$\it{adgrl3.1}$ mutants","WT","Light off"],loc='upper right')
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

fig.savefig(f"graphics/graphs/plot-sprint-duration-all-fish.png")

mutant_on  = sum(mutant_sprint_dur[:6], [])
mutant_off = sum(mutant_sprint_dur[6:], [])

wildtype_on  = sum(wildtype_sprint_dur[:6], [])
wildtype_off = sum(wildtype_sprint_dur[6:], [])


fig1 = plt.figure(figsize=(5, 4))
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(np.array(mutant_on)/1000, dist="norm", plot=plt)
plt.ylabel("Ordered Values [thousands]      ")
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title("SD Q-Q $\it{adgrl3.1}$ on")
plt.legend(["$\it{adgrl3.1}$ mutants"],loc='upper left')
plt.tight_layout()
fig1.savefig(f"graphics/graphs/qqplots/sd_qq_on_mt.png")
plt.show()

fig2 = plt.figure(figsize=(5, 4))
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(np.array(wildtype_on)/1000, dist="norm", plot=plt)
plt.ylabel("Ordered Values [thousands]      ")
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title("SD Q-Q wildtype on")
plt.legend(["wildtype"],loc='upper left')
plt.tight_layout()
fig2.savefig(f"graphics/graphs/qqplots/sd_qq_on_wt.png")
plt.show()

fig3 = plt.figure(figsize=(5, 4))
ax = plt.axes()
ax.set_facecolor('#e6e6e6')
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(np.array(mutant_off)/1000, dist="norm", plot=plt)
plt.ylabel("Ordered Values [thousands]      ")
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title("SD Q-Q $\it{adgrl3.1}$ off")
plt.legend(["$\it{adgrl3.1}$ mutants"],loc='upper left')
plt.tight_layout()
fig3.savefig(f"graphics/graphs/qqplots/sd_qq_off_mt.png")
plt.show()

fig4 = plt.figure(figsize=(5, 4))
ax = plt.axes()
ax.set_facecolor('#e6e6e6')
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(np.array(wildtype_off)/1000, dist="norm", plot=plt)
plt.ylabel("Ordered Values [thousands]      ")
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title("SD Q-Q wildtype off")
plt.legend(["wildtype"],loc='upper left')
plt.tight_layout()
fig4.savefig(f"graphics/graphs/qqplots/sd_qq_off_wt.png")
plt.show()