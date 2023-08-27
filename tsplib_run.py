# -*- coding: utf-8 -*-
import tsplib95
import numpy as np
import torch
from utils import load_model
import copy
from tabulate import tabulate
import csv

# Path where the model to be tested is located
model_path = 'pretrained/cauchy_tsp_100/'

# This function takes in all the cities and normalizes their coordinates to the range [0,1].
# normalization used here is the standard  (x - min_x )/ max_x - min_x
def Normalize_Cities(cities):
    dimension = len(cities)
    X = [cities[i][0] for i in range(dimension)]                    # x-coordinates of all cities
    Y = [cities[i][1] for i in range(dimension)]                    # y-coordinates of all cities
    max_x, min_x, max_y, min_y = max(X), min(X),max(Y), min(Y)      # Find the max and min values of each coordinate
    for i in range(dimension):                                      # Normalize
        cities[i][0] = (cities[i][0] - min_x) / (max_x - min_x)
        cities[i][1] = (cities[i][1] - min_y) / (max_y - min_y)
    return cities


# Input: raw problem data and a tour (excluding the return to the starting point).
# Output: total distance
def calc_total_distance(problem,tour):
    total = 0                       # initial total distance =0
    dimension = problem.dimension   # number of cities
    for i in range(dimension-1):
        edge = tour[i],tour[i+1]
        total = total + problem.get_weight(*edge)
    # Remember to return to the starting point at the end
    edge = tour[dimension-1], tour[0]
    total = total + problem.get_weight(*edge)
    return total


# This function is directly copied from the original code
def make_oracle(model, xy, temperature=1.0):
    num_nodes = len(xy)
    xyt = torch.tensor(xy).float()[None]  # Add batch dimension
    with torch.no_grad():  # Inference only
        embeddings, _ = model.embedder(model._init_embed(xyt))
        fixed = model._precompute(embeddings)

    def oracle(tour):
        with torch.no_grad():  # Inference only
            tour = torch.tensor(tour).long()
            if len(tour) == 0:
                step_context = model.W_placeholder
            else:
                step_context = torch.cat((embeddings[0, tour[0]], embeddings[0, tour[-1]]), -1)

            query = fixed.context_node_projected + model.project_step_context(step_context[None, None, :])
            mask = torch.zeros(num_nodes, dtype=torch.uint8) > 0
            mask[tour] = 1
            mask = mask[None, None, :]  # Add batch and step dimension
            log_p, _ = model._one_to_many_logits(query, fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key, mask)
            p = torch.softmax(log_p / temperature, -1)[0, 0]
            assert (p[tour] == 0).all()
            if (p.sum() - 1).abs() > 1e-5:
                print("Please check this map.")
            #assert (p.sum() - 1).abs() < 1e-5
        return p.numpy()
    return oracle


# TSPLIB problem list
problems_2D = [
    'eil51',
    'berlin52',
    'st70',
    'eil76',
    'pr76',
    'rat99',
    'kroA100',
    'kroB100',
    'kroC100',
    'kroD100',
    'kroE100',
    'rd100',
    'eil101',
    'lin105',
    'pr107',
    'pr124',
    'bier127',
    'ch130',
    'pr136',
    'pr144',
    'ch150',
    'kroA150',
    'kroB150',
    'pr152',
    'u159',
    'rat195',
    'd198',
    'kroA200',
    'kroB200',
    'tsp225',
    'pr226',
    'gil262',
    'pr264',
    'a280',
    'pr299',
    'lin318',
    'rd400',
    'fl417',
    'pr439',
    'pcb442',
    'd493',
    'u574',
    'rat575',
    'p654',
    'd657',
    'u724',
    'rat783',
    'pr1002',
    'u1060',
    'vm1084',
    'pcb1173',
    'd1291',
    'rl1304',
    'rl1323',
    'nrw1379',
    'fl1400',
    'u1432',
    'fl1577',
    'd1655',
    'vm1748',
    'u1817',
    'rl1889',
    'd2103',
    'u2152',
    'u2319',
    'pr2392',
    'pcb3038',
    'fl3795',
    'fnl4461'
]

# Best know solution for TSPLIBs
bestknow = {
'a280' : 2579,
'ali535' : 202310,
'att48' : 10628,
'att532' : 27686,
'bayg29' : 1610,
'bays29' : 2020,
'berlin52' : 7542,
'bier127' : 118282,
'brazil58' : 25395,
'brd14051' : [468942,469935],
'brg180' : 1950,
'burma14' : 3323,
'ch130' : 6110,
'ch150' : 6528,
'd198' : 15780,
'd493' : 35002,
'd657' : 48912,
'd1291' : 50801,
'd1655' : 62128,
'd2103' : [79952,80529],
'd18512' : [644650,645923],
'dantzig42' : 699,
'dsj1000' : 18659688,
'eil101' : 629,
'eil51' : 426,
'eil76' : 538,
'fl417' : 11861,
'fl1400' : 20127,
'fl1577' : [22204,22249],
'fl3795' : [28723,28772],
'fnl4461' : 182566,
'fri26' : 937,
'gil262' : 2378,
'gr17' : 2085,
'gr21' : 2707,
'gr24' : 1272,
'gr48' : 5046,
'gr96' : 55209,
'gr120' : 6942,
'gr137' : 69853,
'gr202' : 40160,
'gr229' : 134602,
'gr431' : 171414,
'gr666' : 294358,
'hk48' : 11461,
'kroA100' : 21282,
'kroB100' : 22141,
'kroC100' : 20749,
'kroD100' : 21294,
'kroE100' : 22068,
'kroA150' : 26524,
'kroB150' : 26130,
'kroA200' : 29368,
'kroB200' : 29437,
'lin105' : 14379,
'lin318' : 42029,
'linhp318' : 41345,
'nrw1379' : 56638,
'p654' : 34643,
'pa561' : 2763,
'pcb442' : 50778,
'pcb1173' : 56892,
'pcb3038' : 137694,
'pla7397' : 23260728,
'pla33810' : [65913275,66138592],
'pla85900' : [141904862,142514146],
'pr76' : 108159,
'pr107' : 44303,
'pr124' : 59030,
'pr136' : 96772,
'pr144' : 58537,
'pr152' : 73682,
'pr226' : 80369,
'pr264' : 49135,
'pr299' : 48191,
'pr439' : 107217,
'pr1002' : 259045,
'pr2392' : 378032,
'rat99' : 1211,
'rat195' : 2323,
'rat575' : 6773,
'rat783' : 8806,
'rd100' : 7910,
'rd400' : 15281,
'rl1304' : 252948,
'rl1323' : 270199,
'rl1889' : 316536,
'rl5915' : [565040,565544],
'rl5934' : [554070,556050],
'rl11849' : [920847,923473],
'si175' : 21407,
'si535' : 48450,
'si1032' : 92650,
'st70' : 675,
'swiss42' : 1273,
'ts225' : 126643,
'tsp225' : 3919,
'u159' : 42080,
'u574' : 36905,
'u724' : 41910,
'u1060' : 224094,
'u1432' : 152970,
'u1817' : 57201,
'u2152' : 64253,
'u2319' : 234256,
'ulysses16' : 6859,
'ulysses22' : 7013,
'usa13509' : [19947008,20167722],
'vm1084' : 239297,
'vm1748' : 336556
}

# create csv file
file = open('tsp_output.csv','w', newline='')
csv_writer = csv.writer(file)

# Default csv format with header and data
headers=["TSPLIB", "Best known solution", "tsp_100"]
data = []
csv_writer.writerow(headers)

# Start evaluation
model, _ = load_model(model_path)
model.eval()

for name in problems_2D:
    print(name)
    problem = tsplib95.load(f"TSPLIB/{name}.tsp")
    opt = bestknow[name]
    dimension = problem.dimension
    cities = [problem.node_coords[i] for i in range(1, dimension + 1)]
    cities_normalized = Normalize_Cities(copy.deepcopy(cities))

    xy = np.array(cities_normalized)
    oracle = make_oracle(model, xy)
    sample = False
    tour = []
    tour_p = []
    while (len(tour) < len(xy)):
        p = oracle(tour)
        if sample:
            g = -np.log(-np.log(np.random.rand(*p.shape)))
            i = np.argmax(np.log(p) + g)
        else:
            i = np.argmax(p)
        tour.append(i)
        tour_p.append(p)

    tour = [tour[i] + 1 for i in range(len(tour))]
    Kool_distance = calc_total_distance(problem, tour)
    data.append([name, opt, Kool_distance])
    
	# Writing data into csv
    csv_writer.writerow([name, opt, Kool_distance])

# Print our the final result
print(tabulate(data, headers=headers))




