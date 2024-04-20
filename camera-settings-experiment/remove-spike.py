import csv

IN_PATH = 'clean-data-full-frame/'
OUT_PATH = 'clean-data-no-spike/'

EXPERIMENT = 1
ALPH_MAP = ["A","B","C","D","E","F","G","H"]

S_55 = 55*1000000
S_65 = 65*1000000

CHECK_FISH = 'E9'

start_list = []
stop_list = []

# Find start and stop time of spike using some fish
# Check 58s to 65s
for alph in ALPH_MAP:
    for num in range(0,14):
        spike_start = S_55
        spike_stop = S_65

        comp_id = f"{alph}{num}"
        with open(IN_PATH + f"ex{EXPERIMENT}/compartments/{comp_id}.csv", mode='r') as in_file:
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

print(f"AVG. SPIKE START: {int(sum(start_list)/len(start_list))}")
print(f"AVG. SPIKE STOP:  {int(sum(stop_list)/len(stop_list))}") 

with open(OUT_PATH + f"ex{EXPERIMENT}/no-spike-statistics.csv", mode='w', newline='') as stat_file:
    stat_writer = csv.writer(stat_file)
    stat_writer.writerow(["id","positive","negative","positive_light_on","positive_light_off","negative_light_on","negative_light_off"])

    for alph in ALPH_MAP:
        for num in range(0,14):

            pre_positive = 0
            pre_total = 0

            post_positive = 0
            post_total = 0

            comp_id = f"{alph}{num}"
            with open(OUT_PATH + f"ex{EXPERIMENT}/compartments/{comp_id}.csv", mode='w', newline='') as out_file:
                with open(IN_PATH + f"ex{EXPERIMENT}/compartments/{comp_id}.csv", mode='r') as in_file:
                    reader = csv.reader(in_file)
                    writer = csv.writer(out_file)
                    stat_writer = csv.writer(stat_file)
                    
                    writer.writerow(["x","y","p","t"])
                    
                    first = True
                    for line in reader:
                        
                        if not first:
                            t = int(line[3])
                            if t < spike_start:
                                writer.writerow(line)
                                pre_positive += int(line[2])
                                pre_total += 1
                            elif t > spike_stop:
                                writer.writerow(line)
                                post_positive += int(line[2])
                                post_total += 1
                        else:
                            first = False

            stat_writer.writerow([comp_id,
                                  pre_positive+post_positive,
                                  pre_total + post_total - pre_positive -post_positive,
                                  pre_positive,
                                  post_positive,
                                  pre_total - pre_positive,
                                  post_total - post_positive])










