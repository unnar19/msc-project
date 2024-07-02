import csv
import matplotlib.pyplot as plt

BASE_PATH = "clean-data-full-frame/"
BASE_PATH2 = "clean-data-no-spike/"

EXPERIMENT = 4
COMP_ID = "D4"
ACC_TIME_MS = 1000
COLORS = ['#499BDA', '#E5705C']

with open(BASE_PATH2 + f"ex{EXPERIMENT}/compartments/{COMP_ID}.csv", mode='r') as file:
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

fig = plt.figure(figsize=(8, 4))
plt.plot(time_axis, event_on, color=COLORS[0])
plt.plot(time_axis, event_off, color=COLORS[1])


#plt.yticks(plt.yticks()[0], [int(tick/1000) for tick in plt.yticks()[0]])
plt.ylim(bottom=0)
plt.title(f"Events in experiment {EXPERIMENT} over time for fish {COMP_ID} (Events sampled at {ACC_TIME_MS}ms)")
plt.ylabel("Events [Ev]")
plt.xlabel("Time [s]")
plt.axis("tight")
plt.legend(["Positive","Negative"])
plt.grid(axis="y")
plt.show()
#fig.savefig(f"graphics/plot-one-fish-ex{EXPERIMENT}-{COMP_ID.lower()}.png")