import calculate_fitness_score as run

results, maps_by_class, time = run.init_task()

print("aqui estao os resultados")
print(results) #precision, recall, map@.5, map@.5: .95: 10, and 3 losse values
print(len(maps_by_class)) #
print(time) #time of pre-processing, time of inference, time of NMS per image
