    import numpy as np
    import random
    import matplotlib.pyplot as plt
    
    num_clients = 10
    num_chromosomes = 100
    num_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.2
    
    clients = [f'K{i}' for i in range(1, num_clients + 1)]
    
    population = [random.sample(clients, num_clients) for _ in range(num_chromosomes)]
    def calculate_total_cost(chromosome, cost_matrix):
        total_cost = 0
        for i in range(num_clients - 1):
            client_from, client_to = chromosome[i], chromosome[i + 1]
            total_cost += cost_matrix[clients.index(client_from)][clients.index(client_to)]
        return total_cost
    
    # Przykładowa macierz kosztów (do dostosowania do rzeczywistych danych)
    cost_matrix = np.random.randint(1, 10, size=(num_clients, num_clients))
    
    # 4. Operatory genetyczne
    def selection(population, cost_matrix):
        selected = random.sample(population, k=2)
        return min(selected, key=lambda x: calculate_total_cost(x, cost_matrix))
    
    def crossover(parent1, parent2):
        crossover_point = random.randint(0, num_clients - 1)
        child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
        return child
    
    def mutation(chromosome):
        mutated_chromosome = chromosome.copy()
        idx1, idx2 = random.sample(range(num_clients), k=2)
        mutated_chromosome[idx1], mutated_chromosome[idx2] = mutated_chromosome[idx2], mutated_chromosome[idx1]
        return mutated_chromosome
    
    for generation in range(num_generations):
        new_population = []
    
        for _ in range(num_chromosomes // 2):
            parent1 = selection(population, cost_matrix)
            parent2 = selection(population, cost_matrix)
    
            if random.random() < crossover_rate:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
            else:
                child1, child2 = parent1, parent2
    
            if random.random() < mutation_rate:
                child1 = mutation(child1)
            if random.random() < mutation_rate:
                child2 = mutation(child2)
    
            new_population.extend([child1, child2])
    
        population = new_population
    
    best_solution = None
    best_cost = float('inf')
    
    for population_size in [50, 100, 150]:
        for crossover_rate in [0.6, 0.8, 1.0]:
            for mutation_rate in [0.1, 0.2, 0.3]:
                population = [random.sample(clients, num_clients) for _ in range(population_size)]
    
                for generation in range(num_generations):
                    new_population = []
    
                    for _ in range(population_size // 2):
                        parent1 = selection(population, cost_matrix)
                        parent2 = selection(population, cost_matrix)
    
                        if random.random() < crossover_rate:
                            child1 = crossover(parent1, parent2)
                            child2 = crossover(parent2, parent1)
                        else:
                            child1, child2 = parent1, parent2
    
                        if random.random() < mutation_rate:
                            child1 = mutation(child1)
                        if random.random() < mutation_rate:
                            child2 = mutation(child2)
    
                        new_population.extend([child1, child2])
    
                    population = new_population
    
                # Znajdź najlepsze rozwiązanie z populacji
                best_route = min(population, key=lambda x: calculate_total_cost(x, cost_matrix))
                current_cost = calculate_total_cost(best_route, cost_matrix)
    
                # Aktualizacja najlepszego wyniku
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = {
                        'population_size': population_size,
                        'crossover_rate': crossover_rate,
                        'mutation_rate': mutation_rate,
                        'generation': generation,
                        'best_route': best_route,
                        'best_cost': best_cost
                    }
    def plot_route(route, cost_matrix):
        x = [cost_matrix[clients.index(client)][0] for client in route]
        y = [cost_matrix[clients.index(client)][1] for client in route]
    
        plt.plot(x, y, marker='o')
        plt.title('Trasa Dostaw')
        plt.xlabel('Szerokość geograficzna')
        plt.ylabel('Długość geograficzna')
        plt.show()
    # Wydrukuj najlepsze wyniki
    print("Najlepsze rozwiązanie:")
    print(f"Rozmiar populacji: {best_solution['population_size']}")
    print(f"Współczynnik krzyżowania: {best_solution['crossover_rate']}")
    print(f"Współczynnik mutacji: {best_solution['mutation_rate']}")
    print(f"Ilość pokoleń: {best_solution['generation']}")
    print(f"Najlepsza trasa: {best_solution['best_route']}")
    print(f"Koszt: {best_solution['best_cost']}")
    
    plot_route(best_solution['best_route'], cost_matrix)