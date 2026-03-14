
class EvolutionaryCodeGenerator:
    def __init__(self, problem_specification):
        self.problem_spec = problem_specification
        self.population_size = 50
        self.generation_count = 0
        self.population = self.generate_initial_population()
    
    def evolve_solution(self, max_generations=100):
        """Evolve code solutions through genetic programming"""
        
        for generation in range(max_generations):
            self.generation_count += 1
            
            # Evaluate fitness of all individuals
            fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
            
            # Selection: Choose best performers
            parents = self.select_parents(self.population, fitness_scores)
            
            # Crossover: Combine successful solutions
            offspring = self.crossover(parents)
            
            # Mutation: Introduce variations
            mutated_offspring = self.mutate(offspring)
            
            # Validation: Ensure code compiles and runs
            valid_offspring = [ind for ind in mutated_offspring if self.is_valid_code(ind)]
            
            # Survival: Select next generation
            self.population = self.select_survivors(self.population + valid_offspring, fitness_scores)
            
            # Check for convergence
            best_fitness = max(fitness_scores)
            if best_fitness >= self.problem_spec['target_fitness']:
                break
            
            if generation % 10 == 0:
                print(f"🧬 Generation {generation}: Best fitness = {best_fitness:.3f}")
        
        # Return best solution
        final_fitness = [self.evaluate_fitness(ind) for ind in self.population]
        best_individual = self.population[final_fitness.index(max(final_fitness))]
        
        return {
            'solution': best_individual,
            'fitness': max(final_fitness),
            'generations': self.generation_count,
            'population_size': len(self.population)
        }
    
    def crossover(self, parents):
        """Combine code from successful parents"""
        offspring = []
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                
                # AST-based crossover
                child1, child2 = self.ast_crossover(parent1, parent2)
                offspring.extend([child1, child2])
        
        return offspring
    
    def mutate(self, individuals):
        """Introduce random variations in code"""
        mutated = []
        
        for individual in individuals:
            if random.random() < self.mutation_rate:
                mutated_individual = self.apply_mutation(individual)
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)
        
        return mutated
