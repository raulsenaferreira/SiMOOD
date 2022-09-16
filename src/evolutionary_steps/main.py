import numpy as np
import calculate_fitness_score
import data_converter
from numpy.linalg import norm 
import random
import copy



## GLOBALS

#simple counters for statistics
COUNTERS = [0, 0] #num_crossover, num_mutation


def multi_delete(list_, args):
	print(args)
	indexes = sorted(args, reverse=True)
	for index in indexes:
		del list_[index]
	return list_

def generate_individual(pool_of_genes_and_severity_lvls, individual_size=2):
	individual = []
	num_pool_genes = len(pool_of_genes_and_severity_lvls)

	rng = np.random.default_rng()
	#ood transformation
	rints_4_techniques = rng.integers(low=0, high=num_pool_genes, size=individual_size)

	#rints_4_techniques.sort()

	for t in rints_4_techniques:
		keys_pool_of_genes = list(pool_of_genes_and_severity_lvls)
		technique = keys_pool_of_genes[t]

		severity_lvls = pool_of_genes_and_severity_lvls[technique]
		lvl = round(random.choice(severity_lvls), 2)
		
		# this helps to avoid possible repeated genes
		repeated = technique in str(individual)
		while repeated:
			randindex = rng.integers(low=0, high=num_pool_genes, size=1)[0]
			technique = keys_pool_of_genes[randindex]
			severity_lvls = pool_of_genes_and_severity_lvls[technique]
			lvl = round(random.choice(severity_lvls), 2)
			repeated = technique in str(individual)

		individual.append((technique, lvl))

	return individual


def generate_initial_population(pool_of_genes_and_severity_lvls, individual_size, population_size):
	population = []

	for p in range(population_size):
		individual = generate_individual(pool_of_genes_and_severity_lvls, individual_size)
		population.append(individual)

	return population


def apply_transformation(data_path, selected_population):
	
	labels_path = '../datasets/coco128/labels/train2017'

	num_individual = 0
	path_generated_datasets = []

	for individual in selected_population:
		num_individual+=1
		#print('num_individual', num_individual)

		modified_data_path = None
		#print('modified_data_path',modified_data_path)
		print('individual',individual)

		for gene in individual:
			(technique, lvl) = gene
			#print('technique, lvl',technique, lvl)
			
			modified_data_path = data_converter.transform_images(data_path, labels_path,
			 num_individual, technique, lvl) #save/return yaml

		path_generated_datasets.append(modified_data_path)

	return np.asarray(path_generated_datasets)


def get_fitness_score(arr_data_path, ml_weights, selected_population, pool_of_genes_and_severity_lvls, ω, score_type='map'):
	
	# fitness score given by the below equation
	# α + ρ = α + ((∥v∥ − min(∥v∥)) / (max(∥v∥) − min(∥v∥))) ∗ ω

	def normalize_data(val, max_val, min_val):
		normalized_val = (val - min_val) / (max_val - min_val)
		return normalized_val

	# checks
	if ω >=1:
		ω = 0.99
	elif ω <=0:
		ω = 0
	
	arr_fitness_score = []

	for data_path, individual in zip(arr_data_path, selected_population):
		normalized_severity_lvls = []
		
		for gene in individual:
			technique = gene[0]
			lvl = gene[1]

			array_lvls_of_tech = pool_of_genes_and_severity_lvls[technique]

			min_lvl = array_lvls_of_tech[0]
			max_lvl = array_lvls_of_tech[-1]

			normalized_severity_lvls.append(normalize_data(lvl, max_lvl, min_lvl)) # (technique, severity_lvl)
		
		results, maps_by_class, time = calculate_fitness_score.init_task(data_path, weights=ml_weights)
		#results => [precision, recall, map@.5, map@.5: .95: 10, and 3 losses values]
		#time => pre-processing, inference, NMS per image

		α = None # ML score

		if score_type == 'precision':
			α = results[0]
		elif score_type == 'recall':
			α = results[1]
		elif score_type == 'map':
			α = results[2]
		elif score_type == 'map95':
			α = results[3]

		print('ML score', α)
		
		#norm_v = norm(normalized_severity_lvls,2)
		#ρ = norm_v * ω
		n = np.sum(normalized_severity_lvls)/len(normalized_severity_lvls)
		ρ = n * ω

		fitness_score = α + ρ

		print('Fitness score', fitness_score)

		arr_fitness_score.append(fitness_score)

	return np.array(arr_fitness_score)


def perform_selection(arr_fitness_score, selection_criteria='worst', num_selections=2):
	#print('arr_fitness_score',arr_fitness_score)
	if selection_criteria == 'worst':
		#print('selected worst individuals', np.argpartition(arr_fitness_score, -2)[-2:])
		selected = (arr_fitness_score).argsort()[:num_selections].astype(int)
		rejected = (-arr_fitness_score).argsort()[:num_selections].astype(int)

		#print("\nselected individuals", selected,"\n")
		#print("\nrejected individuals", rejected,"\n")
		#print('arr_fitness_score after sort',arr_fitness_score)
		
		return selected, rejected
		
	elif selection_criteria == 'best':
		#print('selected best individuals', np.argpartition(arr_fitness_score, 2)[2:])
		selected = (-arr_fitness_score).argsort()[:num_selections]
		rejected = (arr_fitness_score).argsort()[:num_selections]

		return  selected, rejected


def perform_crossover(selected_individual_1, selected_individual_2, pool_of_genes_and_severity_lvls):
	len_indiv = len(selected_individual_1) #len of ind 1 = len of ind 2
	
	evolved_1 = selected_individual_1#[0:len_indiv]
	evolved_2 = selected_individual_2#[0:len_indiv]

	random_i = random.randint(0, len(evolved_1)-1) # evolved_1==evolved_2
	
	#if evolved_2[random_i][0] not in str(evolved_1[random_i]) and evolved_1[random_i][0] not in str(evolved_2[random_i]):
	if evolved_2[random_i][0] not in str(evolved_1) and evolved_1[random_i][0] not in str(evolved_2):	
		print('\n\nperforming crossover')
		COUNTERS[0]+=1
		temp = evolved_1[random_i][0:len_indiv]
		evolved_1[random_i] = evolved_2[random_i][0:len_indiv]
		evolved_2[random_i] = temp

		return evolved_1, evolved_2

	else:
		return perform_mutation(evolved_1, evolved_2, pool_of_genes_and_severity_lvls)


def perform_crossover_2(selected_individual_1, selected_individual_2, crossover_prob, counters):
	
	if random.randint(1, 100) <= crossover_prob:
		evolved_1, evolved_2 = selected_individual_1, selected_individual_2

		rng = np.random.default_rng()
		crossover_point_1 = rng.integers(low=0, high=len(evolved_1)-1, size=1)[0]
		crossover_point_2 = rng.integers(low=0, high=len(evolved_2)-1, size=1)[0]

		signal = crossover_point_1 - crossover_point_2

		if signal < 0: # start swapping from cp_1
			for i in range(crossover_point_1, crossover_point_2+1):
				#avoiding puting repeated genes
				#if evolved_2[i][0] not in str(evolved_1[i]) and evolved_1[i][0] not in str(evolved_2[i]):
					
				counters[0]+=1
				temp = evolved_1[i]
				evolved_1[i] = evolved_2[i]
				evolved_2[i] = temp

		elif signal > 0: # start swapping from cp_2
			for i in range(crossover_point_2, crossover_point_1+1):
				#avoiding puting repeated genes
				#if evolved_2[i][0] not in str(evolved_1[i]) and evolved_1[i][0] not in str(evolved_2[i]):
					
				counters[0]+=1
				temp = evolved_1[i]
				evolved_1[i] = evolved_2[i]
				evolved_2[i] = temp

		else: # randomly chose the same indice, just swap this indice
			#avoiding puting repeated genes
			i = crossover_point_1
			#if evolved_2[i][0] not in str(evolved_1[i]) and evolved_1[i][0] not in str(evolved_2[i]):
				
			counters[0]+=1
			temp = evolved_1[i]
			evolved_1[i] = evolved_2[i]
			evolved_2[i] = temp

		return evolved_1, evolved_2, counters

	else:
		return selected_individual_1, selected_individual_2, counters


def perform_mutation(evolved_1, evolved_2, pool_of_genes_and_severity_lvls):

	def generate_random_gene(pool_of_genes_and_severity_lvls):
		random_index_for_technique = random.randint(0, len(pool_of_genes_and_severity_lvls)-1)
		keys_pool_of_genes = list(pool_of_genes_and_severity_lvls)
		random_technique = keys_pool_of_genes[random_index_for_technique]
		severity_lvls = pool_of_genes_and_severity_lvls[random_technique]
		random_severityLvl = round(random.choice(severity_lvls), 2)

		return (random_technique, random_severityLvl)
	
	random_index_for_mutation = random.randint(0, len(evolved_1)-1) # evolved_1==evolved_2

	technique_evolved_1 = evolved_1[random_index_for_mutation][0]
	technique_evolved_2 = evolved_2[random_index_for_mutation][0]

	new_gene = generate_random_gene(pool_of_genes_and_severity_lvls)
	
	#if new_gene[0] != technique_evolved_1:
	if new_gene[0] not in str(evolved_1):
		evolved_1[random_index_for_mutation] = new_gene
	else:
		# here we avoid that the mutation is equal to what we already have in individual 1
		while new_gene[0] == technique_evolved_1:
			new_gene = generate_random_gene(pool_of_genes_and_severity_lvls)
			random_technique = new_gene[0]
			evolved_1[random_index_for_mutation] = new_gene

	new_gene = generate_random_gene(pool_of_genes_and_severity_lvls)
	#if new_gene[0] != technique_evolved_2:
	if new_gene[0] not in str(evolved_2): 
		evolved_2[random_index_for_mutation] = new_gene
	else:
		# now the same check for the individual 2
		while new_gene[0] == technique_evolved_2:
			new_gene = generate_random_gene(pool_of_genes_and_severity_lvls)
			random_technique = new_gene[0]
			evolved_2[random_index_for_mutation] = new_gene

	return evolved_1, evolved_2


def perform_genetic_algorithm(num_generations, pool_of_genes_and_severity_lvls, M, K, data_path,
 ml_weights, omega, score_type, selection_criteria, crossover_prob, mutation_prob, path_save_results, counters, save=True, stop_condition=None):

	selected_population = generate_initial_population(pool_of_genes_and_severity_lvls, M, K)
	#print("\n\nsize initial_population", len(selected_population),"\n\n")

	for generation in range(0, num_generations):
		print("generation", generation+1)

		generated_datasets_per_OOD_type = apply_transformation(data_path, selected_population)
		#print('generated_datasets_per_OOD_type', generated_datasets_per_OOD_type)

		arr_fitness_score = get_fitness_score(generated_datasets_per_OOD_type, ml_weights, selected_population, pool_of_genes_and_severity_lvls, omega, score_type)
		ind_selected_individuals, ind_rejected_individuals = perform_selection(arr_fitness_score, selection_criteria)
		
		#pair of selected individuals
		selected_individual_1 = selected_population[ind_selected_individuals[0]]
		#fitness_score_individual_1 = arr_fitness_score[ind_selected_individuals[0]]
		
		selected_individual_2 = selected_population[ind_selected_individuals[1]]
		#fitness_score_individual_2 = arr_fitness_score[ind_selected_individuals[1]]

		evolved_1, evolved_2, counters = perform_crossover(copy.deepcopy(selected_individual_1), copy.deepcopy(selected_individual_2), crossover_prob, counters)
		
		evolved_1, evolved_2, counters = perform_mutation(evolved_1, evolved_2, mutation_prob, pool_of_genes_and_severity_lvls, counters)

		#update the population by removing the two worst individuals
		del selected_population[ind_rejected_individuals[0]]
		del selected_population[ind_rejected_individuals[1]]
		#print("\n\nsize selected_population AFTER REMOVING", len(selected_population),"\n\n")
		
		# we add two new individuals to the population
		selected_population.append(evolved_1)
		selected_population.append(evolved_2)
		#print("\n\nsize selected_population AFTER ADDING", len(selected_population),"\n\n")

	if save:
		print("saving results...")
		np.save(path_save_results, selected_population)

	return selected_population


def perform_genetic_algorithm_2(num_generations, pool_of_genes_and_severity_lvls, M, K, data_path,
 ml_weights, omega, score_type, selection_criteria, num_selections, crossover_prob, mutation_prob, path_save_results, save=True, stop_condition=None):

	print("generation 1")
	selected_population = generate_initial_population(pool_of_genes_and_severity_lvls, M, K)
	#print("\n\nsize initial_population", len(selected_population),"\n\n")

	generated_datasets_per_OOD_type = apply_transformation(data_path, selected_population)
	#print('generated_datasets_per_OOD_type', generated_datasets_per_OOD_type)

	arr_fitness_score = get_fitness_score(generated_datasets_per_OOD_type, ml_weights, selected_population, pool_of_genes_and_severity_lvls, omega, score_type)
	ind_selected_individuals, ind_rejected_individuals = perform_selection(arr_fitness_score, selection_criteria, num_selections)

	#pair of selected individuals
	selected_individual_1 = selected_population[ind_selected_individuals[0]]
	#fitness_score_individual_1 = arr_fitness_score[ind_selected_individuals[0]]		
	selected_individual_2 = selected_population[ind_selected_individuals[1]]
	#fitness_score_individual_2 = arr_fitness_score[ind_selected_individuals[1]]

	for generation in range(1, num_generations):
		print("generation {} out of {}".format(generation+1, num_generations))		

		#print("\nsize selected_population and fitness scores BEGIN GENERATION", len(selected_population), len(arr_fitness_score), "\n", selected_population, arr_fitness_score)

		# new individuals
		if random.randint(1, 100) <= crossover_prob:
			evolved_1, evolved_2 = perform_crossover(copy.deepcopy(selected_individual_1), copy.deepcopy(selected_individual_2), pool_of_genes_and_severity_lvls)
		if random.randint(1, 100) <= mutation_prob:
			print('\n\nperforming mutation')
			COUNTERS[1]+=1
			evolved_1, evolved_2 = perform_mutation(evolved_1, evolved_2, pool_of_genes_and_severity_lvls)

		new_individuals = [evolved_1, evolved_2]

		# we add two new individuals to the population and their fitness scores
		#selected_population = np.append(selected_population, new_individuals)
		selected_population.append(evolved_1)
		selected_population.append(evolved_2)

		generated_datasets_per_OOD_type = apply_transformation(data_path, new_individuals)
		#print('generated_datasets_per_OOD_type', generated_datasets_per_OOD_type)

		new_arr_fitness_score = get_fitness_score(generated_datasets_per_OOD_type, ml_weights, new_individuals, pool_of_genes_and_severity_lvls, omega, score_type)
		arr_fitness_score = np.append(arr_fitness_score, new_arr_fitness_score)

		#print("\nsize selected_population and fitness scores AFTER FITNESS CALC", len(selected_population), len(arr_fitness_score), "\n", selected_population, arr_fitness_score)

		# We obtain the index of individuals that will be used to generate new individuals (selected) and the ones that will be discarded (rejected)
		ind_selected_individuals, ind_rejected_individuals = perform_selection(arr_fitness_score, selection_criteria, num_selections)

		#pair of selected individuals
		selected_individual_1 = selected_population[ind_selected_individuals[0]]
		#fitness_score_individual_1 = arr_fitness_score[ind_selected_individuals[0]]		
		selected_individual_2 = selected_population[ind_selected_individuals[1]]
		#fitness_score_individual_2 = arr_fitness_score[ind_selected_individuals[1]]

		#update the population by removing the two worst individuals
		#del selected_population[ind_rejected_individuals[0]]
		#del selected_population[ind_rejected_individuals[1]]
		#selected_population = np.delete(selected_population, ind_rejected_individuals)
		selected_population = multi_delete(selected_population, ind_rejected_individuals)
		arr_fitness_score = np.delete(arr_fitness_score, ind_rejected_individuals)
		#print("\nsize selected_population and fitness scores END GENERATION", len(selected_population), len(arr_fitness_score), "\n", selected_population, arr_fitness_score)

	if save:
		print("saving results...")
		np.save(path_save_results, selected_population)

	return selected_population


def run_simood(num_generations, K, M, selection_criteria, num_selections, score_type, omega, crossover_prob, mutation_prob, stop_condition, path_save_results):
		
		#COCO dataset
		data_path = '../datasets/coco128/images/train2017'
		# YOLO
		ml_weights = 'yolov5s.pt'

		## arrays of genes
		odd_GB = np.linspace(1, 46, num=46, dtype=int)
		odd_GB = odd_GB[odd_GB%2==1] #gaussian blur only accepts odd numbers
		
		pool_of_genes_and_severity_lvls = {
			'shifted_pixel': np.linspace(0.5, 0.05, num=10), #10 levels
			#'row_add_logic': np.linspace(0.05, 0.01, num=5), #5 levels
			'gaussian_noise': np.linspace(0.1, 0.7, num=7), #7 levels
			'gaussian_blur': odd_GB, #23 levels
			'grid_dropout': np.linspace(1, 8, num=8, dtype=int), #8 levels
			'coarse_dropout': np.linspace(10, 40, num=30, dtype=int), #30 levels
			'channel_dropout': [1, 2], #2 levels
			#'channel_shuffle': [1], #1 level
			'snow': np.linspace(1, 5, num=5, dtype=int), #5 levels
			'broken_lens': np.linspace(0.3, 0.8, num=10), #10 levels
			'dirty': np.linspace(0.3, 0.8, num=10), #10 levels
			'condensation': np.linspace(0.1, 0.6, num=10), #10 levels
			'sun_flare': np.linspace(400, 900, num=10, dtype=int), #10 levels
			'brightness': np.linspace(0.3, 0.8, num=10), #10 levels
			'contrast': np.linspace(0.2, 0.7, num=10), #10 levels
			'rain': np.linspace(1, 5, num=5, dtype=int), #5 levels
			'smoke': np.linspace(0.03, 0.3, num=10) #10 levels
			}

		# adding level 0 = no transformation; Doing it here separately to make this information explicit
		#print(len(pool_of_genes_and_severity_lvls['smoke']))#10

		pool_of_genes_and_severity_lvls['shifted_pixel'] = np.insert(pool_of_genes_and_severity_lvls['shifted_pixel'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['gaussian_noise'] = np.insert(pool_of_genes_and_severity_lvls['gaussian_noise'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['gaussian_blur'] = np.insert(pool_of_genes_and_severity_lvls['gaussian_blur'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['grid_dropout'] = np.insert(pool_of_genes_and_severity_lvls['grid_dropout'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['coarse_dropout'] = np.insert(pool_of_genes_and_severity_lvls['coarse_dropout'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['channel_dropout'] = np.insert(pool_of_genes_and_severity_lvls['channel_dropout'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['snow'] = np.insert(pool_of_genes_and_severity_lvls['snow'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['broken_lens'] = np.insert(pool_of_genes_and_severity_lvls['broken_lens'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['dirty'] = np.insert(pool_of_genes_and_severity_lvls['dirty'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['condensation'] = np.insert(pool_of_genes_and_severity_lvls['condensation'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['sun_flare'] = np.insert(pool_of_genes_and_severity_lvls['sun_flare'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['brightness'] = np.insert(pool_of_genes_and_severity_lvls['brightness'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['contrast'] = np.insert(pool_of_genes_and_severity_lvls['contrast'], 0, 0.0, axis=0)
		pool_of_genes_and_severity_lvls['rain'] = np.insert(pool_of_genes_and_severity_lvls['rain'], 0, 0, axis=0)
		pool_of_genes_and_severity_lvls['smoke'] = np.insert(pool_of_genes_and_severity_lvls['smoke'], 0, 0.0, axis=0)

		#print(pool_of_genes_and_severity_lvls['smoke'])
		#print(pool_of_genes_and_severity_lvls['channel_dropout'])
		#print(len(pool_of_genes_and_severity_lvls['smoke']))#11
		#print(len(pool_of_genes_and_severity_lvls['channel_dropout']))#3

		selected_individuals_per_OOD_type = {}
		generated_datasets_per_OOD_type = None

		# GA 
		selected_population = perform_genetic_algorithm_2(num_generations, pool_of_genes_and_severity_lvls, M, K, data_path,
		ml_weights, omega, score_type, selection_criteria, num_selections, crossover_prob, mutation_prob, path_save_results)

		print('num experiments', (num_selections*num_generations-1)+K)
		print('num_crossover', COUNTERS[0])
		print('num_mutation', COUNTERS[1])



if __name__ == '__main__':
	#arr_omega = [0.01, 0.1, 0.25, 0.5, 0.66, 0.75, 0.99] ## smooth term for the fitness function
	arr_omega = [0.5]
	#GA parameters
	#num_generations = [10, 20, 30, 50] ## max num of iterations performed by the GA 
	num_generations = [20]
	stop_condition = 0.1 ## threshold for terminating GA iterations (not used in this moment)
	#arr_pop_size = [10, 20, 30, 50]# population size 
	arr_pop_size = [20]
	M = 2 ## individual size = number of different transformations for each individual
	selection_criteria = 'worst' # or 'best' ## worst will consider the worst scores for selection and best otherwise
	num_selections = 2 # number of individuals to be selected/replaced for each generation
	score_type = 'map' ## ML metric used as fitness score
	crossover_prob = 100 # 1 to 100 percent ## probability to perform a crossover
	mutation_prob = 10 # 1 to 100 percent ## probability to perform mutation after a crossover

	path_save_results = 'evolutionary_results/population_size_{}/omega_{}/run_{}/num_generations_{}'
	for i in range(10):
		print('RUN NUMBER ', i)
		for K in arr_pop_size: 
			for num_gen in num_generations:
				for omega in arr_omega:
					run_simood(num_gen, K, M, selection_criteria, num_selections, score_type, omega, crossover_prob, mutation_prob, stop_condition, path_save_results.format(K, omega, i, num_gen))