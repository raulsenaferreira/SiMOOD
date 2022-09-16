import numpy as np
import calculate_fitness_score
import data_converter
from numpy.linalg import norm 
import random
import collections
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')



class Experiment:
  def __init__(self, population_size, num_generations, omega):
    self.population_size = population_size
    self.num_generations = num_generations
    self.omega = omega
    self.individuals = None
    self.hazards = 0

  def count_unique_individuals(self):
    size_individual = len(self.individuals[0])
    unique_individuals = set()

    for ind in self.individuals:
      individual_name = ''
      
      for gene in ind:
        name=gene[0]
        intensity=gene[1]
        individual_name = individual_name+'{}{}'.format(name, intensity)
      
      unique_individuals.add(individual_name)

    return len(unique_individuals)


  def count_unique_genes(self):
    size_individual = len(self.individuals[0])
    unique_genes = set()

    for ind in self.individuals:
      for gene in ind:
        unique_genes.add('{}{}'.format(gene[0], gene[1]))

    return len(unique_genes)



# How many unique individuals are generated when we change the smoothness term across all amounts of generation and population size?
def plot_num_unique_ind_by_omega(x_label, y_label, arr_experiments):
  fig, ax = plt.subplots()
  
  omega_by_gen = {}
  data = []

  for exp in arr_experiments:
    unique_individuals = exp.count_unique_individuals()

    try:
      omega_by_gen[str(exp.omega)] += unique_individuals
    except:
      omega_by_gen.update({str(exp.omega): unique_individuals})

  #for k,v in omega_by_gen.items():
  #  data.append(v)

  keys = omega_by_gen.keys()
  values = omega_by_gen.values()

  plt.bar(keys, values)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  fig.tight_layout()

  plt.show()


def plot_num_unique_ind_by_gen_pop_fixed_omega(omega_val, num_generations, population_size, arr_experiments):
  
  data = np.zeros((len(num_generations),len(population_size)))
  i=-1

  for gen in num_generations:
    i+=1
    j=0
    for pop in population_size:
      for exp in arr_experiments:
        if exp.num_generations == gen and exp.population_size == pop and exp.omega == omega_val:
          data[i][j] = exp.count_unique_individuals()
          j+=1

  print(data)
  

def plot_num_unique_gene_fixed_omega(omega_val, num_generations, population_size, arr_experiments):
  
  data = np.zeros((len(num_generations),len(population_size)))
  i=-1

  for gen in num_generations:
    i+=1
    j=0
    for pop in population_size:
      for exp in arr_experiments:
        if exp.num_generations == gen and exp.population_size == pop and exp.omega == omega_val:
          data[i][j] = exp.count_unique_genes()
          j+=1

  print(data)


def plot_mock_data(arr_experiments):
  # Create some mock data
  t = np.arange(0.01, 10.0, 0.01)
  data1 = np.exp(t)
  data2 = np.sin(2 * np.pi * t)

  arr_omega = []
  arr_num_generations = []
  arr_population = []
  arr_unique_individuals = []

  for exp in arr_experiments:
    arr_omega.append(exp.omega)
    arr_num_generations.append(exp.num_generations)
    arr_population.append(exp.population_size)
    arr_unique_individuals.append(exp.count_unique_individuals())

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('w')
  ax1.set_ylabel('Generations', color=color)
  ax1.plot(arr_omega, arr_num_generations, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = 'tab:blue'
  ax2.set_ylabel('Unique individuals', color=color)  # we already handled the x-label with ax1
  ax2.plot(arr_omega, arr_unique_individuals, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()


def print_individuals(omega_val, num_generations, population_size, arr_experiments):
  arr_results = []
  path_plots = 'evolutionary_results/individuals.txt'

  for gen in num_generations:
    for pop in population_size:
      for exp in arr_experiments:
        if exp.num_generations == gen and exp.population_size == pop and exp.omega == omega_val:
          txt = '=================\nGenerations; Population size; Generated individuals\n {};{};{};\n'.format(
            gen, pop, exp.individuals)
          arr_results.append(txt)
  
  np.savetxt(path_plots, arr_results, delimiter=',',fmt='%s')
  

# How many hazards are produced when we change the smoothness term, the size of the initial population, and the amount of generations?
def num_hazards_by_gen_popsize_omega():
  pass


if __name__ == '__main__':

  #GA parameters
  arr_omega = [0.01, 0.1, 0.25, 0.5, 0.66, 0.75, 0.99] ## smooth term for the fitness function
  num_generations = [10, 20, 30, 50] ## max num of iterations performed by the GA 
  K = [10, 20, 30, 50] ## population size

  
  
  population_size_dict = {}

  path_load_results = 'evolutionary_results/population_size_{}/omega_{}/num_generations_{}'

  arr_experiments = []

  for pop_size in K:
    
    for omega in arr_omega:
      for num_gen in num_generations:
        a = np.load('{}.npy'.format(path_load_results.format(pop_size, omega, num_gen)))
        exp = Experiment(pop_size, num_gen, omega)
        exp.individuals = a
        arr_experiments.append(exp)
        #print('Omega:',omega,'Num of generations:',num_gen,'Selected population:\n',a)

  #print(arr_experiments[0].count_unique_individuals())
  #print(arr_experiments[0].individuals)

  #plot_num_unique_ind_by_omega('ω', 'Unique individuals', arr_experiments) #best omega = 0.5
  #plot_num_unique_ind_by_gen_pop_fixed_omega(0.5, num_generations, K, arr_experiments)
  #print_individuals(0.5, num_generations, K, arr_experiments)
  #plot_num_unique_gene_fixed_omega(0.5, num_generations, K, arr_experiments)



  ### Analysis by fixed omega, pop size and generations
  runs = 10
  path_load_results = 'evolutionary_results/population_size_{}/omega_{}/run_{}/num_generations_{}'

  arr_experiments = []

  fixed_pop_size = 20
  fixed_gen_size = 20
  fixed_omega = 0.5

  arr_results = []
  path_plots = 'evolutionary_results/population_size_{}/omega_{}/individuals_by_run.txt'.format(fixed_pop_size,fixed_omega)

  for r in range(runs):
    a = np.load('{}.npy'.format(path_load_results.format(fixed_pop_size, fixed_omega, r, fixed_gen_size)))
    exp = Experiment(fixed_pop_size, fixed_gen_size, fixed_omega)
    exp.individuals = a
    #print(exp.count_unique_genes(),'\n')

    txt = '=================\nGenerations; Generated individuals\n {};{};\n'.format(
            fixed_gen_size, exp.individuals)
    arr_results.append(txt)
  
  np.savetxt(path_plots, arr_results, delimiter=',',fmt='%s')