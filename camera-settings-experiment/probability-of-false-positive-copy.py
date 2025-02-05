import csv
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = "clean-data-no-spike/"
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

SPRINT_THRESHOLD_US = 15000
MIN_EVENTS_PER_SPRINT = 20

d = ","

COMPARTMENTS = [f"{alph}{num}" for alph in ALPH_MAP for num in range(1,13)]
NOISE = [f"{alph}{num}" for alph in ALPH_MAP for num in [0,13]]

# This will help identify a good sprint threshold
print("\nMeasure the probability of false positive sprint")

for exp_i in range(1,6): 
    print(f"ex{exp_i} threshold{d}P(FP){d}P(FP)^{MIN_EVENTS_PER_SPRINT}")

    P_FP_array   = []
    P_FP20_array = []
    Threshold_array = []

    SPRINT_THRESHOLD_US = 100
    max_threshold = 30001
    factor = 1.1

    while SPRINT_THRESHOLD_US <= max_threshold:
        total_time_intervals = 0
        total_sprint_events = 0

        for comp_id in NOISE:
            with open(f"{BASE_PATH}ex{exp_i}/compartments/{comp_id}.csv", mode='r') as file:
                csvFile = csv.reader(file)
                next(csvFile)

                timestamps = [int(row[3]) for row in csvFile]
                time_diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
                time_diffs = np.array(time_diffs)

                # plt.plot(time_diffs)
                # plt.plot([0,len(time_diffs)],[SPRINT_THRESHOLD_US,SPRINT_THRESHOLD_US],"k-")
                # plt.legend(["Time between events","Threshold"])
                # plt.show()

                time_intervals = len(time_diffs)
                sprint_events = len(time_diffs[time_diffs <= SPRINT_THRESHOLD_US])
                # print(sprint_events/time_intervals)
                total_time_intervals += time_intervals
                total_sprint_events += sprint_events
            
        SPRINT_THRESHOLD_US *= factor

        P_FP = total_sprint_events/total_time_intervals
        P_FP20 = pow(P_FP, MIN_EVENTS_PER_SPRINT)

        P_FP_array.append(P_FP)
        P_FP20_array.append(P_FP20)
        Threshold_array.append(SPRINT_THRESHOLD_US)

        print(f"{SPRINT_THRESHOLD_US}{d}{P_FP:.6f}{d}{P_FP20:.12f}")

    plt.plot(Threshold_array, P_FP_array)
    plt.xlabel("Threshold [µs]")
    plt.ylabel("Probability")
    plt.legend(["P(FP)"])
    plt.title(f"Camera Ex{exp_i} - Sensitivity analysis of counting false positive sprints")
    plt.savefig(f"graphics\sensitivity\PFP-ex{exp_i}.png")
    plt.show()

    plt.plot(Threshold_array, P_FP20_array)
    plt.xlabel("Threshold [µs]")
    plt.ylabel("Probability")
    plt.legend(["P(FP)$^{"+ f"{MIN_EVENTS_PER_SPRINT}" +"}$"])
    plt.title(f"Camera Ex{exp_i} - Sensitivity analysis of counting false positive sprints")
    plt.savefig(f"graphics\sensitivity\PFP20-ex{exp_i}.png")
    plt.show()


