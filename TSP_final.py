# -*- coding: utf-8 -*-
"""
Created on Thu May 13 00:29:23 2021

@author: vicky
"""

import numpy as np
import pandas as pd
import random
from itertools import combinations
import copy
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']


##Dynamic Programming
class Dynamic_Programming:
    def __init__(self, vertex_num):
        self.vertex_num = vertex_num
        self.vertex_list_without_v0  = {l for l in range(1, self.vertex_num)}
    
    def notInSubset(self, subset):
        candidate = self.vertex_list_without_v0
        s = set(subset)   #set(('1',)) = {'1'}、 set(('1','2')) = {'1', '2'}
        return candidate-s
    
    def remaining_tuple(self, tuple1, int2):
        remaining_tuple = set(tuple1)- {int2}
        remaining_tuple = tuple(sorted(remaining_tuple))
        return remaining_tuple
    
    def analyze_path(self, P):
        subset_last = tuple(self.vertex_list_without_v0)
        vertices = [0]  #起點
        to_there = 0
        remaining = subset_last
        
        for i in range(self.vertex_num-1):
            to_there = P[(to_there, remaining)]
            vertices.append(to_there)
            remaining = self.remaining_tuple(remaining, to_there)
        vertices.append(0) #走回起點
        return vertices
    
    
    def TravelingSalesmanProblem(self):
        D = {}   #D[vi][[A]代表length of shortest path from vi to v0 passing through A
        P = {}   
        
        #空集合
        for i in range(1, self.vertex_num):
            D[(i, ())] = W[i][0]
        #非空集合
        for k in range(1, self.vertex_num-1):  #k from 1 to n-2
            for subset in set(combinations(self.vertex_list_without_v0, k)):
                for i in self.notInSubset(subset):
#                    print('subset:{}, i={}'.format(subset, i))
                    remaining_sets = sorted(set(combinations(subset, k-1)), reverse=True)
#                    print("remaining_sets:", remaining_sets)
                    minimum = float('inf')
                    for j, remaining in zip(subset, remaining_sets):              
                        temp = W[i][j] + D[(j, remaining)]
#                        print('j={}, remaining_set={}'.format(j, remaining))
#                        print("value:", temp)
                        if temp < minimum:
                            minimum = temp
                            D[(i, subset)] = temp
                            P[(i, subset)] = j
#                        print("minimum:", minimum)

        #i從0出發走回0
        subset_last = tuple(self.vertex_list_without_v0)  #(1, 2, 3)
#        print('subset:{}, i={}'.format(subset_last, 0))
        remaining_sets = sorted(set(combinations(subset_last, self.vertex_num-2)), reverse=True)
#        print("remaining_sets:", remaining_sets)
        minimum = float('inf')
        for j, remaining in zip(subset_last, remaining_sets):
            temp = W[0][j] + D[(j, remaining)]
#            print('j={}, remaining_set={}'.format(j, remaining))
#            print("value:", temp)
            if temp < minimum:
                minimum = temp
                D[(0, subset_last)] = temp
                P[(0, subset_last)] = j
#            print("minimum:", minimum)
        
#        print('D:\n{}'.format(D))
#        print('P:\n{}'.format(P))
        
        vertices = self.analyze_path(P)
        return minimum, vertices
        

##Genetic Algotithm
class Location:
    def __init__(self, name, loc_index):
        self.name = name
        self.loc_index = loc_index

    def distance_between(self, location2):
        assert isinstance(location2, Location)
        return W[self.loc_index][location2.loc_index]


class Route:
    def __init__(self, path):
        # path is a list of Location obj
        self.path = path
        self.length = self._set_length()

    def _set_length(self):
        total_length = 0
        path_copy = self.path[:]
        from_here = path_copy.pop(0)
        init_node = copy.deepcopy(from_here)
        while path_copy:
            to_there = path_copy.pop(0)
            total_length += to_there.distance_between(from_here)
            from_here = copy.deepcopy(to_there)
        total_length += from_here.distance_between(init_node)
        return total_length


class GeneticAlgo:
    def __init__(self, locs, level=10, populations=100, variant=3, mutate_percent=0.01, elite_save_percent=0.1):
        self.locs = locs
        self.level = level
        self.variant = variant
        self.populations = populations
        self.mutates = int(populations * mutate_percent)
        self.elite = int(populations * elite_save_percent)

    def _find_path(self):
            # locs is a list containing all the Location obj
            locs_copy = self.locs[:]
            path = []
            while locs_copy:
                location = locs_copy.pop(locs_copy.index(random.choice(locs_copy)))
                path.append(location)
            return path

    def _init_routes(self):
        routes = []
        for _ in range(self.populations):
            path = self._find_path()
            routes.append(Route(path))
        return routes
    
    def _get_next_route(self, routes):
        routes.sort(key=lambda x: x.length, reverse=False)  #依照Route物件的length排序  #reverse=False代表小到大
        elites = routes[:self.elite][:]
        crossovers = self._crossover(elites)
        return crossovers[:] + elites

    def _crossover(self, elites):
        # Route is a class type
        normal_breeds = []
        mutate_ones = []
        for _ in range(self.populations - self.mutates):
            father, mother = random.choices(elites[:4], k=2)
            index_start = random.randrange(0, len(father.path)- self.variant- 1)
            # list of Location obj
            father_gene = father.path[index_start: index_start+self.variant]
            father_gene_names = [loc.name for loc in father_gene]
            mother_gene = [gene for gene in mother.path if gene.name not in father_gene_names]
            mother_gene_cut = random.randrange(1, len(mother_gene))
            # create new route path
            next_route_path = mother_gene[:mother_gene_cut] + father_gene + mother_gene[mother_gene_cut:]
            next_route = Route(next_route_path)
            # add Route obj to normal_breeds
            normal_breeds.append(next_route)

            # for mutate purpose
            copy_father = copy.deepcopy(father)
            idx = range(len(copy_father.path))
            gene1, gene2 = random.sample(idx, 2)
            copy_father.path[gene1], copy_father.path[gene2] = copy_father.path[gene2], copy_father.path[gene1]
            mutate_ones.append(copy_father)
        mutate_breeds = random.choices(mutate_ones, k=self.mutates)
        return normal_breeds + mutate_breeds
    
        
    def evolution(self):
        routes = self._init_routes()  #創建出第一代可行路徑們
        for _ in range(self.level):
            routes = self._get_next_route(routes)  #根據上一代創造出下一代可行路徑們
        routes.sort(key=lambda x: x.length)
        return routes[0].path, routes[0].length


def create_locations(vertex_num):
    cities = [str(i) for i in range(vertex_num)]
    indexes = [i for i in range(vertex_num)]

    locations = []
    for name, index in zip(cities, indexes):
        locations.append(Location(name, index))
    return locations, cities, indexes


##共用函數
def make_graph(vertex_num):
    global W
    W = [[0 for i in range(vertex_num)] for j in range(vertex_num)]
#    W = np.zeros([vertex_num, vertex_num])
    for i in range(vertex_num):
        for j in range(i, vertex_num):
            if i == j:
                W[i][j] = 0
            else:
                weight = random.randint(1,30)
                W[i][j] = weight
                W[j][i] = weight
#    print('W:\n{}'.format(W))
    return W

    
def draw_time(DP_avg_time, Genetic_avg_time):
    vertex = np.arange(min_num, max_num+1)
    plt.plot(vertex, DP_avg_time, label="Dynamic Programming",marker = "o")
    plt.plot(vertex, Genetic_avg_time, label="Genetic Algorithm", marker = "o")
    plt.xticks(vertex)
    plt.title("演算法平均時間") # title
    plt.ylabel("平均時間(秒)") # y label
    plt.xlabel("頂點數") # x label
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.savefig("./img/time.png", dpi=300)
    plt.show()


def draw_error(error):
    vertex = np.arange(min_num, max_num+1)
    plt.xticks(vertex)
    plt.plot(vertex, error, marker = "o")
    plt.title("基因演算法各點平均誤差") # title
    plt.ylabel("平均相對誤差") # y label
    plt.xlabel("頂點數") # x label
    plt.grid(True)
    plt.savefig("./img/error.png", dpi=300)
    plt.show()
    

if __name__ == '__main__':
    DP_avg_time = []
    Genetic_avg_time = []
    error = []
    
    min_num = 4
    max_num = 20
    
    for n in range(min_num, max_num+1):  #點數
        col = ["DP_Time", "Genetic_Time", "DP_length", "Genetic_length"]
        record = pd.DataFrame(np.zeros([5, 4]), columns=col)
        
        for j in range(5):  #各點跑5次
            #Dynamic Programming
            start1 = time.time()
            W = make_graph(n)
            my_DP = Dynamic_Programming(n)
            DP_minlength, DP_path = my_DP.TravelingSalesmanProblem()
            print('DP最短路徑長:{}'.format(DP_minlength))
            print('DP路線:{}'.format(DP_path))
            end1 = time.time()
            DP_total_time = end1 - start1
            print("DP執行時間:{}秒".format(DP_total_time))
            
            record["DP_Time"][j] = DP_total_time
            record["DP_length"][j] = DP_minlength
            
            
            #Genetic Algorithm
            start2 = time.time()
            locations, cities, indexes = create_locations(n)
            my_GeneticAlgo = GeneticAlgo(locations, level=200, populations=150, variant=2, mutate_percent=0.05, elite_save_percent=0.15)
            best_route, best_route_length = my_GeneticAlgo.evolution()
            best_route.append(best_route[0])
            print('Genetic最短路徑長:{}'.format(best_route_length))
            print('Genetic路線:{}'.format([loc.name for loc in best_route]))
            end2 = time.time()
            Genetic_total_time = end2 - start2
            print("Genetic執行時間:{}秒".format(Genetic_total_time))
            
            record["Genetic_Time"][j] = Genetic_total_time
            record["Genetic_length"][j] = best_route_length


        print('點數:{}'.format(n))
        print(record)
        
        DP_avg_time.append(record['DP_Time'].mean())
        Genetic_avg_time.append(record['Genetic_Time'].mean())
        error.append(((record['Genetic_length']-record['DP_length'])/abs(record['DP_length'])).mean())
    
    draw_time(DP_avg_time, Genetic_avg_time)
    draw_error(error)
    

