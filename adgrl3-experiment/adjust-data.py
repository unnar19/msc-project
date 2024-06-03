import csv

IN_PATH = 'adgrl3-clean-frame/'
OUT_PATH = 'adgrl3-clean-adjusted/'

DATES = ["2024-04-21", "2024-04-22"]
EXPERIMENTS = [4,8]

ALPH_MAP = ["A","B","C","D","E","F","G","H"]

S_55 = (2*60+55)*1000000
S_65 = (2*60+65)*1000000
SPIKE_DURATION = 1 * 1000000

END_GOAL = (357)*1000000
SPIKE_GOAL = (END_GOAL - 1000000)/2

# Find start and stop time of spike using some fish
# Check 58s to 65s
for date_i, date in enumerate(DATES):
    for exp_i in range(1,EXPERIMENTS[date_i]+1):
        start_list = []
        stop_list = []
        
        for alph in ALPH_MAP:
            for num in [0,13]:
                spike_start = S_55
                spike_stop = S_65

                comp_id = f"{alph}{num}"
                with open(IN_PATH + date + f"/ex{exp_i}/compartments/{comp_id}.csv", mode='r') as in_file:
                    reader = csv.reader(in_file)
                    first = True
                    
                    zero_counter = 0
                    zero_start_time = 0

                    best_zero_count = 0

                    for line in reader:
                        if not first:
                            t = int(line[3])
                            if t > S_55 and t < S_65:
                                p = int(line[2])

                                if p == 0:
                                    zero_counter += 1
                                    if zero_start_time == 0:
                                        zero_start_time = t

                                    if zero_counter > best_zero_count:
                                        best_zero_count = zero_counter
                                        spike_start = zero_start_time
                                        spike_stop = t
                                else:
                                    zero_counter = 0
                                    zero_start_time = 0

                        else:
                            first = False

                start_list.append(spike_start)
                stop_list.append(spike_stop)


        avg_spike_start = min(start_list)#int(sum(start_list)/len(start_list))
        
        difference = SPIKE_GOAL - avg_spike_start
        # shift all so spike start is at 150s


        print(f"AVG. SPIKE START:     {avg_spike_start}")
        print(f"AVG. SPIKE STOP:      {int(sum(stop_list)/len(stop_list))}") 
        print(f"AVG. SPIKE DURATION:  {(int(sum(stop_list)/len(stop_list)) - avg_spike_start)/1000} ms") 


        print(difference)
        print(avg_spike_start + difference)

        with open(OUT_PATH + date + f"/ex{exp_i}/adjusted-statistics.csv", mode='w', newline='') as stat_file:
            stat_writer = csv.writer(stat_file)
            stat_writer.writerow(["id","positive","negative","positive_light_on","positive_light_off","negative_light_on","negative_light_off"])

            for alph in ALPH_MAP:
                for num in range(0,14):

                    pre_positive = 0
                    pre_total = 0

                    post_positive = 0
                    post_total = 0

                    comp_id = f"{alph}{num}"
                    with open(OUT_PATH + date + f"/ex{exp_i}/compartments/{comp_id}.csv", mode='w', newline='') as out_file:
                        with open(IN_PATH + date + f"/ex{exp_i}/compartments/{comp_id}.csv", mode='r') as in_file:
                            reader = csv.reader(in_file)
                            writer = csv.writer(out_file)
                            stat_writer = csv.writer(stat_file)
                            
                            writer.writerow(["x","y","p","t"])
                            
                            first = True
                            for line in reader:
                                if not first:
                                    t = int(line[3])
                                    if t < spike_start:
                                        if t + difference >= 0:
                                            writer.writerow([line[0],line[1],line[2],int(t+difference)])
                                            pre_positive += int(line[2])
                                            pre_total += 1
                                    elif t > spike_start + SPIKE_DURATION: 
                                        if t + difference <= END_GOAL:
                                            writer.writerow([line[0],line[1],line[2],int(t+difference)])
                                            post_positive += int(line[2])
                                            post_total += 1
                                        else:
                                            a=1
                                else:
                                    first = False

                    stat_writer.writerow([comp_id,
                                        pre_positive+post_positive,
                                        pre_total + post_total - pre_positive -post_positive,
                                        pre_positive,
                                        post_positive,
                                        pre_total - pre_positive,
                                        post_total - post_positive])










