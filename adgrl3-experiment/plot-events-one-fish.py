import csv
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = "adgrl3-clean-adjusted/"
DATE = "2024-04-22"
COLORS = ['#499BDA', '#E5705C']

EXPERIMENT = 4
COMP_ID = "B10"
ACC_TIME_MS = 1000
SPIKE_TIME = 178

with open(BASE_PATH + DATE + f"/ex{EXPERIMENT}/compartments/{COMP_ID}.csv", mode='r') as file:
    csvFile = csv.reader(file)
    header = True
    first = True

    event_on = []
    event_off = []

    time_axis = []

    event_count_on = 0
    event_count_off = 0

    last_sample_time = 0

    for line in csvFile:
        if not header:
            if first:
                last_sample_time = int(line[3])
                first = False
            
            if int(line[3]) < last_sample_time + ACC_TIME_MS*1000:
                if int(line[2]) == 1:
                    event_count_on += 1
                else:
                    event_count_off += 1
            else:
                time_axis.append(last_sample_time/1000000)
                last_sample_time = int(line[3])
                event_on.append(event_count_on)
                event_off.append(event_count_off)
                event_count_on = 0
                event_count_off = 0
                if int(line[2]) == 1:
                    event_count_on += 1
                else:
                    event_count_off += 1

        else:
            header = False

ymin = min([min(event_on),min(event_off)])
ymax = max([max(event_on),max(event_off)])

fig = plt.figure(figsize=(10, 6))
plt.plot(time_axis, event_on, color=COLORS[0], linewidth=2)
plt.plot(time_axis, event_off, color=COLORS[1])
plt.vlines(SPIKE_TIME, 0, ymax*1.1, "k")
time_step_axis = np.linspace(min(time_axis),max(time_axis),100000)
plt.fill_between(time_step_axis, 0, ymax*1.1, where=(np.array(time_step_axis) >= SPIKE_TIME) & (np.array(time_step_axis) <= max(time_step_axis)), color='gray', alpha=0.2)


#plt.yticks(plt.yticks()[0], [int(tick/1000) for tick in plt.yticks()[0]])
plt.ylim(bottom=0)
plt.title(f"Events in experiment {EXPERIMENT} on {DATE} over time for fish {COMP_ID} (Events sampled at {ACC_TIME_MS}ms)")
plt.ylabel("Events [Ev]")
plt.xlabel("Time [s]")
#plt.axis("tight")
plt.legend(["Positive","Negative"])
plt.grid(axis="y")
plt.xlim([0,356])
plt.ylim([0, ymax*1.1])
plt.show()
#fig.savefig(f"graphics/plot-one-fish-ex{EXPERIMENT}-{COMP_ID.lower()}.png")