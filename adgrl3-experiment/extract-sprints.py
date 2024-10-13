import csv
import matplotlib.pyplot as plt

BASE_PATH = "adgrl3-clean-adjusted/"
OUT_PATH = "adgrl3-sprints/"
DATES = ["2024-04-21","2024-04-22"]
EXPERIMENTS = [4,8]
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

SPRINT_THRESHOLD_US = 15000
MIN_EVENTS_PER_SPRINT = 20

COMPARTMENTS = [f"{alph}{num}" for alph in ALPH_MAP for num in range(1,13)]
NOISE = [f"{alph}{num}" for alph in ALPH_MAP for num in [0,13]]

# This will help identify a good sprint threshold
print("\nCalculate average time between events where there is no fish")
print("ex [#]\tavg_time [µs]")

for date_i, date in enumerate(DATES):
    for exp_i in range(1,EXPERIMENTS[date_i]+1):
        average_time_diff = []

        for comp_id in NOISE:
            with open(f"{BASE_PATH}{date}/ex{exp_i}/compartments/{comp_id}.csv", mode='r') as file:
                csvFile = csv.reader(file)
                next(csvFile)

                timestamps = [int(row[3]) for row in csvFile]
                time_diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
                average_time_diff.append(sum(time_diffs) / len(time_diffs))

        new_average = int(sum(average_time_diff) / len(average_time_diff))
        print(f"{exp_i}  \t{new_average:.0f}")

count = 0
for date_i, date in enumerate(DATES):
    for exp_i in range(1,EXPERIMENTS[date_i]+1):
        for comp_id in COMPARTMENTS:
            sprint_count = 0
            #print(f"Compartment: {comp_id}")
            with open(f"{BASE_PATH}{date}/ex{exp_i}/compartments/{comp_id}.csv", mode='r') as file:
                with open(f"{OUT_PATH}{date}/ex{exp_i}/compartments/{comp_id}.csv", mode='w', newline='') as out_file:
                    csvFile = csv.reader(file)
                    next(csvFile)
                    csvWriter = csv.writer(out_file)
                    csvWriter.writerow(["duration","start","stop"])
                    
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
                                #a sprint was detected between sprint_start_time and last_time
                                duration = last_time - sprint_start_time
                                #sprint_count += 1
                                csvWriter.writerow([duration, sprint_start_time, last_time])
                                #print(f"{duration/1000:.0f}ms ({sprint_start_time/1000:.2f} to {last_time/1000:.2f})")

                            event_count = 0

                        last_time = time
        count += 1
        print(f"{count}/{sum(EXPERIMENTS)}")
                

# Output file will be: start_time, stop_time, duration


