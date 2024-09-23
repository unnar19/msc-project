import csv
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = "adgrl3-clean-adjusted/"
DATES = ["2024-04-21","2024-04-22"]
EXPERIMENTS = [4,8]
EXPERIMENTS = [[2,4],[2,4,6,8]]
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

SPRINT_THRESHOLD_US = 15000
MIN_EVENTS_PER_SPRINT = 1

d = ","

COMPARTMENTS = [f"{alph}{num}" for alph in ALPH_MAP for num in range(1,13)]
NOISE = [f"{alph}{num}" for alph in ALPH_MAP for num in [0,13]]

Time_moving_array = []
Time_moving_array_on = []
Time_moving_array_off = []
Threshold_array = []

SPRINT_THRESHOLD_US = 100
max_threshold = 30001
factor = 1.1

while SPRINT_THRESHOLD_US <= max_threshold:

    time_moving = 0
    time_moving_on = 0
    time_moving_off = 0
    compartment_count = 0

    for date_i, date in enumerate(DATES):
        for exp_i in EXPERIMENTS[date_i]: 

            for comp_id in COMPARTMENTS:
                with open(f"{BASE_PATH}{date}/ex{exp_i}/compartments/{comp_id}.csv", mode='r') as file:
                    csvFile = csv.reader(file)
                    next(csvFile)
                    
                    event_count = 0
                    last_time = 0
                    sprint_start_time = 0
                    
                    for line in csvFile:
                        time = int(line[3])

                        if time < last_time + SPRINT_THRESHOLD_US:
                            event_count += 1
                            if event_count == 1:
                                sprint_start_time = last_time
                        else:
                            if event_count >= MIN_EVENTS_PER_SPRINT:
                                time_moving += (last_time - sprint_start_time)/1000000
                                if time <= 178000000:
                                    time_moving_on += (last_time - sprint_start_time)/1000000
                                else:
                                    time_moving_off += (last_time - sprint_start_time)/1000000

                            event_count = 0

                        last_time = time

                    compartment_count += 1

    Time_moving_array.append(time_moving/(2*compartment_count))
    Time_moving_array_on.append(time_moving_on/compartment_count)
    Time_moving_array_off.append(time_moving_off/compartment_count)
    Threshold_array.append(SPRINT_THRESHOLD_US)

    SPRINT_THRESHOLD_US *= factor   

plt.plot(Threshold_array, Time_moving_array)
plt.plot(Threshold_array, Time_moving_array_on)
plt.plot(Threshold_array, Time_moving_array_off)
plt.xlabel("Threshold [µs]")
plt.ylabel("Probability")
plt.legend(["(Full recording)/2","Light on","Light off"])
plt.title("ADGRL3.1 & WT - Sensitivity analysis of counting false positive sprints")
plt.savefig("graphics\graphs\sensitivity\sensitivity.png")
plt.show()


