import csv
import matplotlib.pyplot as plt
import tikzplotlib

BASE_PATH = "adgrl3-clean-adjusted/"
FILE_NAME = "adjusted-statistics.csv"
DATES = ["2024-04-21", "2024-04-22"]
EXPERIMENTS = [4,8]

colors = ['#499BDA', '#E5705C', '#5D7D96', '#9A574C']

for date_i, date in enumerate(DATES):
    for exp_i in range(1,EXPERIMENTS[date_i]+1):
        data = []
        with open(BASE_PATH + date + f"/ex{exp_i}/" + FILE_NAME, mode='r') as file:
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

        print(f"\n{e_on:.0f} & {e_off:.0f} & {n_on:.0f} & {n_off:.0f}")

        print(f"\ne on  {int(round(e_on,0))}")
        print(f"e off {int(round(e_off,0))}")
        print(f"n on  {int(round(n_on,0))}")
        print(f"n off {int(round(n_off,0))}\n")
        
        print(f"SNR = {SNR:.5f}")
        print(f"PNR = {PNR:.5f}")

        fig = plt.figure(figsize=(6, 7))
        events = plt.boxplot(data, positions=[1,2,4,5], widths=0.35, patch_artist=True)

        for box, color in zip(events['boxes'], colors):
            box.set(color=color, linewidth=2)
            box.set_facecolor(color)


        plt.axis([0.5, 5.5, 0, 65000])
        plt.yticks(range(0, 65001, 5000), [str(int(ytick / 1000)) for ytick in range(0, 65001, 5000)])

        plt.xticks(range(1,6), ["Event +","Event -","","Noise +","Noise -"])
        plt.title(f"Box plot of adjusted events in experiment {exp_i}")
        plt.ylabel("Kilo events [kE]")
        plt.grid(axis="y")
        plt.show()
        #fig.savefig(f"graphics/graphs/adjusted/box-plot-adjusted-{date}-ex{exp_i}.png")
        #tikzplotlib.save(f"graphics/graphs/adjusted/tex/box-plot-adjusted-{date}-ex{exp_i}.tex")
