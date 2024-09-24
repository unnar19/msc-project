import csv
import matplotlib.pyplot as plt
import numpy as np

BASE_PATH = "adgrl3-clean-adjusted/"
BASE_PATH2 = "adgrl3-sprints/"
COLORS = ["#0000FF", "#FF0000"]
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

DATES = ["2024-04-21", "2024-04-22"]
EXPERIMENTS = [[2,4],[2,4,6,8]]
MUTANTS_ON_LEFT = [[1,0],[0,1,0,1]]

WHOLE_TIME = 357
SPIKE_TIME = 178

TRAIN_RATIO = 0.85

ec_mutant_on = []
ec_mutant_off = []
ec_wildtype_on = []
ec_wildtype_off = []
sc_mutant_on = []
sc_mutant_off = []
sc_wildtype_on = []
sc_wildtype_off = []
sd_mutant_on = []
sd_mutant_off = []
sd_wildtype_on = []
sd_wildtype_off = []
tm_mutant_on = []
tm_mutant_off = []
tm_wildtype_on = []
tm_wildtype_off = []

comp_ids = []
for alph in ALPH_MAP:
    for num in range(1,13):
        comp_ids.append(f"{alph}{num}")

def plot_histogram_bins(mutant_on, mutant_off, wildtype_on, wildtype_off,metric,ymax,bins,unit):
    fig = plt.figure(figsize=(10, 6))
    bins = np.histogram_bin_edges(mutant_on + mutant_off + wildtype_on + wildtype_off, bins=bins)

    plt.subplot(1, 2, 1)
    plt.hist(mutant_on, bins=bins, color="blue", alpha=0.5, rwidth=0.85, label="ADGRL3.1")
    # plt.axvline(x=np.mean(mutant_on), color="blue", label="ADGRL3.1 Mean")
    plt.hist(wildtype_on, bins=bins, color="red", alpha=0.5, rwidth=0.85, label="WT")
    plt.axvline(x=np.mean(wildtype_on + mutant_on), color="k", label="Mean")
    # plt.axvline(x=np.mean(wildtype_on), color="red", label="WT Mean")
    plt.ylim((0,ymax))
    plt.xlabel(f"{metric} bins [{unit}]")
    plt.ylabel("Count of larvae [#]")
    plt.title(f"{metric} histogram for lights on")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(mutant_off, bins=bins, color="blue", alpha=0.5, rwidth=0.85, label="ADGRL3.1")
    # plt.axvline(x=np.mean(mutant_off), color="blue", label="ADGRL3.1 Mean")
    plt.hist(wildtype_off, bins=bins, color="red", alpha=0.5, rwidth=0.85, label="WT")
    plt.axvline(x=np.mean(wildtype_off + mutant_off), color="k", label="Mean")
    plt.ylim((0,ymax))
    plt.xlabel(f"{metric} bins [{unit}]")
    plt.ylabel("Count of larvae [#]")
    plt.title(f"{metric} histogram for lights off")
    plt.legend()

    plt.tight_layout()
    # fig.savefig(f"graphics\graphs\histograms\histogram-{'-'.join(metric.lower().split())}.png")
    plt.show()

N_train = int(TRAIN_RATIO * 96)
N_test = 96 - N_train
train_array = np.concatenate((np.ones((1,N_train)),np.zeros((1,N_test))),axis=1)[0]

train_store = []
seed = 2616854216

for date_i, date in enumerate(DATES):
    for exp_i, exp in enumerate(EXPERIMENTS[date_i]):

        mutants_on_left = bool(MUTANTS_ON_LEFT[date_i][exp_i])
        comp_count = 0

        np.random.seed(seed)

        seed += 1

        np.random.shuffle(train_array)
        train_store.append(train_array.copy())

        for a, alph in enumerate(ALPH_MAP):
            for num in range(1,13):
                # detect if this compartment is train
                if train_array[comp_count] == 1:
                    COMP_ID = f"{alph}{num}"

                    if (mutants_on_left == True and num < 7) or (mutants_on_left == False and num >= 7):
                        is_mutant = True
                    else:
                        is_mutant = False

                    with open(BASE_PATH + date + f"/ex{exp}/compartments/{COMP_ID}.csv", mode="r") as file:
                        csvFile = csv.reader(file)
                        next(csvFile)

                        timestamps = [int(row[3]) for row in csvFile]
                        event_count_on = len([time for time in timestamps if time < SPIKE_TIME*1000000])
                        event_count_off = len([time for time in timestamps if time >= (SPIKE_TIME+1)*1000000])

                        if is_mutant:
                            ec_mutant_on.append(event_count_on)
                            ec_mutant_off.append(event_count_off)
                        else:
                            ec_wildtype_on.append(event_count_on)
                            ec_wildtype_off.append(event_count_off)

                    with open(BASE_PATH2 + date + f"/ex{exp}/compartments/{COMP_ID}.csv", mode="r") as file:
                        csvFile = csv.reader(file)
                        next(csvFile)

                        durs  = []
                        starts = []
                        ends = []

                        for row in csvFile:
                            durs.append(int(row[0]))
                            starts.append(int(row[1]))
                            ends.append(int(row[2]))

                        durations_on =  [dur for start,dur in zip(starts,durs) if start < SPIKE_TIME*1000000]
                        durations_off = [dur for start,dur in zip(starts,durs) if start >= (SPIKE_TIME+1)*1000000]

                        sprint_count_on = len(durations_on)
                        sprint_count_off = len(durations_off)
                        sprint_dur_on = np.mean(durations_on)/1000000
                        sprint_dur_off = np.mean(durations_off)/1000000
                        time_moving_on = sum(durations_on)/1000000
                        time_moving_off = sum(durations_off)/1000000

                        if is_mutant:
                            sc_mutant_on.append(sprint_count_on)
                            sc_mutant_off.append(sprint_count_off)
                            sd_mutant_on.append(sprint_dur_on)
                            sd_mutant_off.append(sprint_dur_off)
                            tm_mutant_on.append(time_moving_on)
                            tm_mutant_off.append(time_moving_off)
                        else:
                            sc_wildtype_on.append(sprint_count_on)
                            sc_wildtype_off.append(sprint_count_off)
                            sd_wildtype_on.append(sprint_dur_on)
                            sd_wildtype_off.append(sprint_dur_off)
                            tm_wildtype_on.append(time_moving_on)
                            tm_wildtype_off.append(time_moving_off)
                
                comp_count += 1

sd_mutant_on = np.array(sd_mutant_on)
sd_mutant_off = np.array(sd_mutant_off)
sd_wildtype_on = np.array(sd_wildtype_on)
sd_wildtype_off = np.array(sd_wildtype_off)

sd_mutant_on = sd_mutant_on[~np.isnan(sd_mutant_on)]
sd_mutant_off = sd_mutant_off[~np.isnan(sd_mutant_off)]
sd_wildtype_on = sd_wildtype_on[~np.isnan(sd_wildtype_on)]
sd_wildtype_off = sd_wildtype_off[~np.isnan(sd_wildtype_off)]

sd_mutant_on    = list(sd_mutant_on)
sd_mutant_off   = list(sd_mutant_off)
sd_wildtype_on  = list(sd_wildtype_on)
sd_wildtype_off = list(sd_wildtype_off)

ec_threshold_on = np.mean([np.mean(ec_mutant_on), np.mean(ec_wildtype_on)])
ec_threshold_off = np.mean([np.mean(ec_mutant_off), np.mean(ec_wildtype_off)])
sc_threshold_on = np.mean([np.mean(sc_mutant_on), np.mean(sc_wildtype_on)])
sc_threshold_off = np.mean([np.mean(sc_mutant_off), np.mean(sc_wildtype_off)])
sd_threshold_on = np.mean([np.mean(sd_mutant_on), np.mean(sd_wildtype_on)])
sd_threshold_off = np.mean([np.mean(sd_mutant_off), np.mean(sd_wildtype_off)])
tm_threshold_on = np.mean([np.mean(tm_mutant_on), np.mean(tm_wildtype_on)])
tm_threshold_off = np.mean([np.mean(tm_mutant_off), np.mean(tm_wildtype_off)])

plot_histogram_bins(ec_mutant_on,ec_mutant_off,ec_wildtype_on,ec_wildtype_off,"Event count",90,50,"#")
plot_histogram_bins(sc_mutant_on,sc_mutant_off,sc_wildtype_on,sc_wildtype_off,"Sprint count",160,50,"#")
plot_histogram_bins(sd_mutant_on,sd_mutant_off,sd_wildtype_on,sd_wildtype_off,"Avg. sprint duration",45,50,"s")
plot_histogram_bins(tm_mutant_on,tm_mutant_off,tm_wildtype_on,tm_wildtype_off,"Time moving",250,50,"s")



recording_count = 0

for date_i, date in enumerate(DATES):
    for exp_i, exp in enumerate(EXPERIMENTS[date_i]):

        mutants_on_left = bool(MUTANTS_ON_LEFT[date_i][exp_i])
        comp_count = 0

        train_store

        for a, alph in enumerate(ALPH_MAP):
            for num in range(1,13):
                # detect if this compartment is test
                if train_store[recording_count][comp_count] == 0:
                    COMP_ID = f"{alph}{num}"

                    if (mutants_on_left == True and num < 7) or (mutants_on_left == False and num >= 7):
                        is_mutant = True
                    else:
                        is_mutant = False

                    with open(BASE_PATH + date + f"/ex{exp}/compartments/{COMP_ID}.csv", mode="r") as file:
                        csvFile = csv.reader(file)
                        next(csvFile)

                        timestamps = [int(row[3]) for row in csvFile]
                        event_count_on = len([time for time in timestamps if time < SPIKE_TIME*1000000])
                        event_count_off = len([time for time in timestamps if time >= (SPIKE_TIME+1)*1000000])

                        if is_mutant:
                            ec_mutant_on.append(event_count_on)
                            ec_mutant_off.append(event_count_off)
                        else:
                            ec_wildtype_on.append(event_count_on)
                            ec_wildtype_off.append(event_count_off)

                    with open(BASE_PATH2 + date + f"/ex{exp}/compartments/{COMP_ID}.csv", mode="r") as file:
                        csvFile = csv.reader(file)
                        next(csvFile)

                        durs  = []
                        starts = []
                        ends = []

                        for row in csvFile:
                            durs.append(int(row[0]))
                            starts.append(int(row[1]))
                            ends.append(int(row[2]))

                        durations_on =  [dur for start,dur in zip(starts,durs) if start < SPIKE_TIME*1000000]
                        durations_off = [dur for start,dur in zip(starts,durs) if start >= (SPIKE_TIME+1)*1000000]

                        sprint_count_on = len(durations_on)
                        sprint_count_off = len(durations_off)
                        sprint_dur_on = np.mean(durations_on)/1000000
                        sprint_dur_off = np.mean(durations_off)/1000000
                        time_moving_on = sum(durations_on)/1000000
                        time_moving_off = sum(durations_off)/1000000

                        if is_mutant:
                            sc_mutant_on.append(sprint_count_on)
                            sc_mutant_off.append(sprint_count_off)
                            sd_mutant_on.append(sprint_dur_on)
                            sd_mutant_off.append(sprint_dur_off)
                            tm_mutant_on.append(time_moving_on)
                            tm_mutant_off.append(time_moving_off)
                        else:
                            sc_wildtype_on.append(sprint_count_on)
                            sc_wildtype_off.append(sprint_count_off)
                            sd_wildtype_on.append(sprint_dur_on)
                            sd_wildtype_off.append(sprint_dur_off)
                            tm_wildtype_on.append(time_moving_on)
                            tm_wildtype_off.append(time_moving_off)
                
                comp_count += 1
        recording_count += 1