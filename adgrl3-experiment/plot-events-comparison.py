import csv
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

BASE_PATH = "adgrl3-clean-adjusted/"
FILE_NAME = "adjusted-statistics.csv"
DATES = ["2024-04-21", "2024-04-22"]
EXPERIMENTS = [[2,4],[2,4,6,7]]
MUTANTS_ON_LEFT = [[1,0],[0,1,0,1]]

mutant_events = []
wildtype_events = []
noise_events = []

# Plot total events for ADGRL and WT fish
for date_i, date in enumerate(DATES):
    for exp_i, exp in enumerate(EXPERIMENTS[date_i]):
        data = []
        with open(BASE_PATH + date + f"/ex{exp}/" + FILE_NAME, mode='r') as file:
            csvFile = csv.reader(file)
            header = True
            for line in csvFile:
                if not header:
                    index = int(line[0][1:])
                    mutants_on_left = bool(MUTANTS_ON_LEFT[date_i][exp_i])
                    events = int(line[1])+int(line[2])
                    if index not in [0,13]:
                        if (mutants_on_left == True and index < 7) or (mutants_on_left == False and index >= 7):
                            mutant_events.append(events)
                        else:
                            wildtype_events.append(events)
                    else:
                        noise_events.append(events)
                else:
                    header = False

# avg_noise = sum(noise_events)/len(noise_events)

for i in range(len(mutant_events)):
    mutant_events[i] = np.log10(mutant_events[i])
    wildtype_events[i] = np.log10(wildtype_events[i])

print(mutant_events)
print(len(wildtype_events))
   
mutant, bins, _ = plt.hist(mutant_events,bins=40,fc=(0,0,1,0.5))
wildtype,  _, _ = plt.hist(wildtype_events,bins=bins,fc=(1,0,0,0.5))

plt.legend(["adgrl3.1","WT"])
plt.title(f"Frequency of event count per fish")
plt.xlabel("log10(Events)")
plt.ylabel("Frequency [f]")
plt.show()
#fig.savefig(f"graphics/graphs/adjusted/box-plot-adjusted-{date}-ex{exp}.png")
#tikzplotlib.save(f"graphics/graphs/adjusted/tex/box-plot-adjusted-{date}-ex{exp}.tex")