import csv
from time import time

BASE_PATH = 'clean-data-full-frame/ex1/'
FILE_NAME = 'data.csv'

COMPARTMENTS_X = 12
COMPARTMENTS_Y = 8

OFFSET_X = 274
OFFSET_Y = 53

SIZE_X = 66
SIZE_Y = 66

ALPH_MAP = ["A","B","C","D","E","F","G","H"]

HEADER = "x,y,p,t"

def oob(x,y):
    """
    True if event is out of bounds
    """
    return (x <= OFFSET_X - SIZE_X or x > SIZE_X + OFFSET_X + width or y <= OFFSET_Y or y > OFFSET_Y + height)

def get_compartment(x,y):
    """
    Return id string of assigned compartment
    """
    comp_x = ((x - OFFSET_X - 1) // SIZE_X) + 1
    comp_y = ((y - OFFSET_Y - 1) // SIZE_Y) + 1
    return f"{ALPH_MAP[comp_y-1]}{comp_x}", comp_x-1, comp_y-1

start_time = time()

width = SIZE_X * COMPARTMENTS_X
height = SIZE_Y * COMPARTMENTS_Y

# Create and/or empty all files
fd_dict = {}
for alph in ALPH_MAP:
    for num in range(0,COMPARTMENTS_X+2):
        with open(BASE_PATH + f"compartments/{alph}{num}.csv", mode='w', newline='') as empty_file:
            pass
        fd = open(BASE_PATH + f"compartments/{alph}{num}.csv", mode='a', newline='')
        csvWriter = csv.writer(fd)
        csvWriter.writerow(HEADER.split(","))
        fd_dict[f"{alph}{num}"] = [fd, csvWriter, 0, 0]

# Read CSV and calculate compartment ids
with open(BASE_PATH + FILE_NAME, mode='r') as file:
    csvFile = csv.reader(file)
    header = True
    first_time = True
    for line in csvFile:
        if not header:
            line = [int(line[0]),int(line[1]),int(line[2]),int(line[3])]
            
            if first_time:
                first_t = line[3]
                first_time = False

            if not oob(line[0],line[1]):
                comp_id, comp_xi, comp_yi = get_compartment(line[0],line[1])
                line[0] -= (OFFSET_X + (SIZE_X * comp_xi))
                line[1] -= (OFFSET_Y + (SIZE_Y * comp_yi))
                line[3] -= first_t

                fd_dict[f"{comp_id}"][1].writerow(line)
                fd_dict[f"{comp_id}"][2] += line[2]
                fd_dict[f"{comp_id}"][3] += 1

        else:
            header = False

end_time = time()
print(f"Took {end_time - start_time} seconds to load and write")

print("Closing all file descriptors and")
print("writing statistics file")

with open(BASE_PATH + "statistics.csv", mode='w', newline='') as stat_file:
    csvWriter = csv.writer(stat_file)
    csvWriter.writerow(["id","events_on","events_off"])

    fd_list = list(fd_dict)
    for key in fd_list:
        fd_dict[key][0].close()
        csvWriter.writerow([key,fd_dict[key][2],fd_dict[key][3] - fd_dict[key][2]])


    



