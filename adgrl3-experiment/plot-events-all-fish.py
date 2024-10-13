import csv
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from matplotlib.lines import Line2D
import scipy.stats as stats

BASE_PATH = "adgrl3-clean-adjusted/"
COLORS = ["#0000FF", "#FF0000"]
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

DATES = ["2024-04-21", "2024-04-22"]
EXPERIMENTS = [[2,4],[2,4,6,8]]
MUTANTS_ON_LEFT = [[1,0],[0,1,0,1]]

WHOLE_TIME = 357
SPIKE_TIME = 178
BIN_COUNT = 12

def split_event_list(all_events_list):

    start = 0
    stop = 48

    result_on =[]
    result_off=[]

    for i in range(1,7):
        result_on.append([a+b+c+d+e+f for a,b,c,d,e,f in zip(all_events_list[0][start:stop],
                                                    all_events_list[1][start:stop],
                                                    all_events_list[2][start:stop],
                                                    all_events_list[3][start:stop],
                                                    all_events_list[4][start:stop],
                                                    all_events_list[5][start:stop])])

        result_off.append([a+b+c+d+e+f for a,b,c,d,e,f in zip(all_events_list[6][start:stop],
                                                        all_events_list[7][start:stop],
                                                        all_events_list[8][start:stop],
                                                        all_events_list[9][start:stop],
                                                        all_events_list[10][start:stop],
                                                        all_events_list[11][start:stop])])
        
        start += 48
        stop += 48
        
    return sum(result_on, []), sum(result_off, [])

ACC_TIME_MS = 1000*WHOLE_TIME/BIN_COUNT

bin_stop_time = np.arange(ACC_TIME_MS, 1000*WHOLE_TIME+ACC_TIME_MS, ACC_TIME_MS)
mutant_whole_average = np.zeros(BIN_COUNT)
wildtype_whole_average = np.zeros(BIN_COUNT)

mutant_all_events = [[] for _ in range(BIN_COUNT)]
wildtype_all_events = [[] for _ in range(BIN_COUNT)]

ymax = 0
fig = plt.figure(figsize=(7, 4))

for date_i, date in enumerate(DATES):
    for exp_i, exp in enumerate(EXPERIMENTS[date_i]):

        mutants_on_left = bool(MUTANTS_ON_LEFT[date_i][exp_i])
        mutant_file_average = [0] * BIN_COUNT
        wildtype_file_average = [0] * BIN_COUNT
        time_axis_file_average = [time for time in list(np.linspace(ACC_TIME_MS/2000,357-ACC_TIME_MS/2000,BIN_COUNT))]

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
                    first = True

                    time_axis = []
                    time_axis_plot = []
                    events = []
                    event_count = 0
                    last_sample_time = 0

                    bin_index = 0

                    for line in csvFile:
                        if not header:
                            if first:
                                last_sample_time = int(line[3])
                                first = False

                            if int(line[3]) <= 1000*int(bin_stop_time[bin_index]):
                                event_count += 1
                            else:
                                time_axis.append(last_sample_time/1000000)
                                time_axis_plot.append(last_sample_time/1000000 + ACC_TIME_MS/2000 + 0.6*(ACC_TIME_MS/2000 * uniform(-1.0,1.0)))
                                last_sample_time = int(line[3])

                                events.append(event_count)
                                if is_mutant:
                                    mutant_file_average[bin_index] += event_count
                                    mutant_all_events[bin_index].append(event_count)
                                else:
                                    wildtype_file_average[bin_index] += event_count
                                    wildtype_all_events[bin_index].append(event_count)


                                event_count = 1
                                bin_index += 1

                        else:
                            header = False

                    time_axis.append(last_sample_time/1000000)
                    time_axis_plot.append(last_sample_time/1000000 + ACC_TIME_MS/2000 + 0.6*(ACC_TIME_MS/2000 * uniform(-1.0,1.0)))
                    events.append(event_count)
                    if is_mutant:
                        mutant_file_average[bin_index] += event_count
                        mutant_all_events[bin_index].append(event_count)
                    else:
                        wildtype_file_average[bin_index] += event_count
                        wildtype_all_events[bin_index].append(event_count)

                highest = max(events)
                if highest > ymax:
                    ymax = highest

                plt.scatter(time_axis_plot, events, color=c, marker=".", alpha=0.1)

        mutant_whole_average += np.array(mutant_file_average)/48
        wildtype_whole_average += np.array(wildtype_file_average)/48

mutant_whole_average = mutant_whole_average/6
wildtype_whole_average = wildtype_whole_average/6


mt_on, mt_off = split_event_list(mutant_all_events)
wt_on, wt_off = split_event_list(wildtype_all_events)

print(f"Non parametric (Not normal distribution)")
print(f"p-value\tSignificant")

_, p_value = stats.mannwhitneyu(mt_on, wt_on)
print(f"{p_value:.6f}\t{bool(p_value < 0.05)}")
_, p_value = stats.mannwhitneyu(mt_off, wt_off)
print(f"{p_value:.6f}\t{bool(p_value < 0.05)}")

# print(f"Non parametric (Not normal distribution)")
# print(f"p-value\tSignificant")
# for bin_i in range(BIN_COUNT):
#     _, p_value = stats.mannwhitneyu(mutant_all_events[bin_i], wildtype_all_events[bin_i])
#     print(f"{p_value:.5f}\t{bool(p_value < 0.05)}")

# print(f"Parametric (Normal distribution)")
# print(f"p-value\tSignificant")
# for bin_i in range(BIN_COUNT):
#     _, p_value = stats.ttest_ind(mutant_all_events[bin_i], wildtype_all_events[bin_i])
#     p_value_one_tailed = p_value / 2
#     print(f"{p_value_one_tailed:.5f}\t{bool(p_value_one_tailed < 0.05)}")

quit()

plt.plot(time_axis_file_average,mutant_whole_average, color=COLORS[0])
plt.plot(time_axis_file_average,wildtype_whole_average, color=COLORS[1])

mutant1 = Line2D([0], [0], color='b')
mutant2 = Line2D([0], [0], marker=".", linestyle="none", color='b', alpha=0.2)
wildtype1 = Line2D([0], [0], color='r')
wildtype2 = Line2D([0], [0], marker=".", linestyle="none", color='r', alpha=0.2)
black = Line2D([0], [0], color='k')

plt.title("Event count over time for all fish with accumulation time ~30s")
plt.legend([(mutant1,mutant2), (wildtype1,wildtype2), black] ,["$\it{adgrl3.1}$ mutants","wildtype","Light off"],loc="upper right")
plt.xlim([0,357])
plt.ylabel("Event count [#]")
plt.xlabel("Time [s]")
plt.ylim(bottom=0)
plt.grid(axis="y")
plt.vlines(SPIKE_TIME, 0, ymax*1.1, "k")
time_step_axis = np.linspace(min(time_axis),357,100000)
plt.fill_between(time_step_axis, 0, ymax*1.1, where=(np.array(time_step_axis) >= SPIKE_TIME) & (np.array(time_step_axis) <= 357), color='gray', alpha=0.2)
plt.ylim([0, 8000])
plt.show()
fig.savefig(f"graphics/graphs/plot-all-fish.png")


mutant_on  = sum(mutant_all_events[:6], [])
mutant_off = sum(mutant_all_events[6:], [])

wildtype_on  = sum(wildtype_all_events[:6], [])
wildtype_off = sum(wildtype_all_events[6:], [])


fig1 = plt.figure(figsize=(5, 4))
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(np.array(mutant_on)/1000, dist="norm", plot=plt)
plt.ylabel("Ordered Values [thousands]      ")
plt.gca().get_lines()[0].set_color('blue')
plt.gca().get_lines()[1].set_color('k')
plt.title("EC Q-Q $\it{adgrl3.1}$ on")
plt.legend(["$\it{adgrl3.1}$ mutants"])
plt.tight_layout()
fig1.savefig(f"graphics/graphs/qqplots/ec_qq_on_mt.png")
plt.show()

fig2 = plt.figure(figsize=(5, 4))
plt.rcParams['font.size'] = 16
plt.grid(axis="y")
stats.probplot(np.array(wildtype_on)/1000, dist="norm", plot=plt)
plt.ylabel("Ordered Values [thousands]      ")
plt.gca().get_lines()[0].set_color('red')
plt.gca().get_lines()[1].set_color('k')
plt.title(f"EC Q-Q wildtype on")
plt.legend(["wildtype"])
plt.tight_layout()
fig2.savefig(f"graphics/graphs/qqplots/ec_qq_on_wt.png")
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
plt.title("EC Q-Q $\it{adgrl3.1}$ off")
plt.legend(["$\it{adgrl3.1}$ mutants"])
plt.tight_layout()
fig3.savefig(f"graphics/graphs/qqplots/ec_qq_off_mt.png")
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
plt.title(f"EC Q-Q wildtype off")
plt.legend(["wildtype"])
plt.tight_layout()
fig4.savefig(f"graphics/graphs/qqplots/ec_qq_off_wt.png")
plt.show()