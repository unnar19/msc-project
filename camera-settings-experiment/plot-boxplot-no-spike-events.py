import csv
import matplotlib.pyplot as plt

BASE_PATH = "clean-data-no-spike/"
FILE_NAME = "no-spike-statistics.csv"

colors = ['#499BDA', '#E5705C', '#5D7D96', '#9A574C']

for exp_i in range(1,6):
    data = []
    with open(BASE_PATH + f"ex{exp_i}/" + FILE_NAME, mode='r') as file:
        csvFile = csv.reader(file)
        header = True
        event_on = []
        event_off = []
        noise_on = []
        noise_off = []
        for line in csvFile:
            if not header:
                if int(line[0][1:]) in [13, 0]:
                    noise_on.append(int(line[1]))
                    noise_off.append(int(line[2]))
                else:
                    event_on.append(int(line[1]))
                    event_off.append(int(line[2]))
            else:
                header = False
    data.append(event_on)
    data.append(event_off)
    data.append(noise_on)
    data.append(noise_off)

    e_on = sum(event_on)/len(event_on)
    e_off= sum(event_off)/len(event_off)
    n_on = sum(noise_on)/len(noise_on)
    n_off= sum(noise_off)/len(noise_off)
    
    signal_and_noise = e_on + e_off
    noise = n_on + n_off
    signal = signal_and_noise - noise

    SNR = signal/noise
    PNR = (e_on + n_on)/(e_off + n_off)

    fig = plt.figure(figsize=(4, 4))
    events = plt.boxplot(data, positions=[1,2,4,5], widths=0.35, patch_artist=True)

    for box, color in zip(events['boxes'], colors):
        box.set(color=color, linewidth=2)
        box.set_facecolor(color)


    plt.yticks(plt.yticks()[0], [int(tick/1000) for tick in plt.yticks()[0]])
    plt.xticks(range(1,6), ["Event +","Event -","","Noise +","Noise -"])
    plt.ylim(bottom=0)
    plt.title(f"Box plot of no spike events in experiment {exp_i}")
    plt.ylabel("Kilo events [kE]")
    plt.axis("tight")
    plt.grid(axis="y")
    plt.show()
    fig.savefig(f"graphics/no-spike/box-plot-no-spike-ex{exp_i}.png")