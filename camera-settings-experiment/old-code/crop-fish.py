import csv
from time import time

BASE_PATH = 'clean-data-full-frame/ex2/'
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

# Generate a dictionary containing empty lists for events
comp_dict = {}
for alph in ALPH_MAP:
    for num in range(0,COMPARTMENTS_X+2):
        comp_dict[f"{alph}{num}"] = []

ids = list(comp_dict)

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
                comp_dict[comp_id].append(line)

        else:
            header = False
        
mid_time = time()
print(f"Took {mid_time - start_time} seconds to load")

# Write each compartment in a separate file
for comp_id in ids:
    with open(BASE_PATH + f"compartments/{comp_id}.csv", mode='w', newline='') as file:
        csvWriter = csv.writer(file)
        csvWriter.writerow(HEADER.split(","))
        csvWriter.writerows(comp_dict[comp_id])

end_time = time()
print(f"Took {end_time - mid_time} seconds to write")
print(f"Total time was {end_time - start_time} seconds")


print("Writing statistics")
# Write statistics for each fish in a file
with open(BASE_PATH + "event-count.csv", mode='w', newline='') as file:
    csvWriter = csv.writer(file)
    csvWriter.writerow(["id","events_on","events_off",f"{end_time - start_time:.3f}"])
    for comp_id in ids:
        events = comp_dict[comp_id]
        pos = 0
        for e in events:
            pos += e[2]
        csvWriter.writerow([comp_id,pos,abs(pos-len(comp_dict[comp_id]))])

print("Finished")
