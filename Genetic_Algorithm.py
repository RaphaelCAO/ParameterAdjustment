import numpy as np
from random import shuffle


class Gene(object):
    def __init__(self, init_gene=None):
        """
        constructor of a gene

        :param init_gene: initial value of the gene
        """
        if init_gene is None:
            self.gene = np.random.rand()*5
        else:
            self.gene = float(init_gene)

    def mutate(self, rate):
        """
        value of the gene will mutate randomly in a certain range. The range depends
        on the quantity of itself

        :param rate: mutation rate
        :return: None
        """
        sign = np.sign(np.random.rand() - 0.5)
        self.gene *= (1+rate * np.random.random()) ** sign


class Chromosome(object):
    def __init__(self, genes_num, genes_val=None):
        """
        constructor of a chromosome

        :param genes_num: amount of genes in the chromosome
        :param genes_val: initial value of the genes
        """
        self.genes_num = genes_num
        self.mutation_num_rate = 0.5
        self.mutation_val_rate = 2
        if genes_val is None:
            self.genes = [Gene() for i in range(genes_num)]
        else:
            if len(genes_val) != genes_num:
                raise Exception("Genes number does not match values")
            self.genes = [Gene(val) for val in genes_val]

    def mutate(self):
        """
        random number of genes will mutate

        :return:  None
        """
        mutation_number = np.ceil(self.mutation_num_rate * self.genes_num)
        mutation_number = mutation_number.astype(np.int32)
        mutated_genes = np.random.choice(self.genes, mutation_number, replace=False)
        [gene.mutate(self.mutation_val_rate) for gene in mutated_genes]

    def get_gene_val_list(self):
        """
        show the values of the genes by a list

        :return: a list of values for the genes
        """
        return [gene.gene for gene in self.genes]


class Individual(object):
    def __init__(self, genes_number, genes_val=None, generation=0):
        """
        constructor of an individual

        :param genes_number: amount of genes for the individual
        :param genes_val: initial gene value of the individual
        :param generation: initial generation of the individual
        """
        self.chromosome = Chromosome(genes_number, genes_val)
        self.generation = generation
        self.inheritance_rate = 0.3
        self.performance = self.evaluate()


    def evaluate(self):
        """
        evaluate the performance of the individual

        :return: performance of the individual
        """
        genes = self.chromosome.get_gene_val_list()
        score = 1
        for gene in genes:
            score *= gene

        if score > 10:
            score = 0   # explode

        return score

    def match(self, another_individual):
        """
        Match another individual to create a new individual that inherits genes randomly
        from its parents and mutates the genes randomly.

        :param another_individual:
        :return: a new individual
        """
        genes1 = self.chromosome.get_gene_val_list()
        genes2 = another_individual.chromosome.get_gene_val_list()

        if len(genes1) != len(genes2):
            raise Exception("The genes number of the two individual are not the same.")
        genes_num = self.chromosome.genes_num

        total_indexes = [i for i in range(genes_num)]
        half_indexes = [i for i in range(0, genes_num, 2)]
        partial_indexes = np.random.choice(half_indexes,
                        np.maximum(int(genes_num*self.inheritance_rate), 1), replace=False)

        new_gene = []
        for index in total_indexes:
            if index in partial_indexes:
                new_gene.append(genes2[index])
            else:
                new_gene.append(genes1[index])

        generation = np.maximum(self.generation, another_individual.generation) + 1
        child1 = Individual(len(new_gene), genes_val=new_gene, generation=generation)

        # mutation
        child1.chromosome.mutate()

        return child1


class Population(object):
    def __init__(self, individual_num, genes_number, init_genes=None):
        """
        constructor of a population

        :param individual_num: amount of individuals in the population
        :param genes_number: amount of genes for every individual
        :param init_genes: initial genes of the individuals in the first generation.
        """
        self.individual_num = individual_num
        self.genes_number = genes_number
        if init_genes is None:
            self.population = [Individual(genes_number) for i in range(individual_num)]
        else:
            self.population = [Individual(genes_number, genes_val=init_genes)]
            [self.population.append(Individual(genes_number)) for i in range(individual_num - 1)]

    def sort_by_performance(self):
        """
        sort the individuals in the population from the best one to the worst one

        :return: None
        """
        self.population = sorted(self.population, key=lambda x: x.performance, reverse=True)

    def next_generation(self):
        """
        remove the individuals with low performance
        create new individuals that might inherit the strengths from parents
        import new invader

        :return: None
        """
        self.sort_by_performance()
        couple_num = int(self.individual_num/4)
        winner_num = int(self.individual_num/2)

        self.population = self.population[:winner_num]

        shuffled_index = [i for i in range(winner_num)]
        shuffle(shuffled_index)
        for i in range(0, couple_num, 2):
            parent1 = self.population[shuffled_index[i]]
            parent2 = self.population[shuffled_index[i+1]]
            child1 = parent1.match(parent2)
            self.population.append(child1)     # add a child who inherits the properties of parents
            self.population.append(Individual(self.genes_number))   # add a new random individual

    def get_best_individual(self):
        """
        find the individual with the highest performance

        :return: the best individual
        """
        self.sort_by_performance()
        return self.population[0]


if __name__ == '__main__':
    """
    For instance, there is are 5 numbers. The task is find a combination of any 5 numbers whose
    product is as close as possible to 10. However, if the product is larger than 10, the task 
    fails.
    """
    population = Population(32, 5)
    best = population.get_best_individual()
    print("initial performace: {}\nGenes:{}\n".format(best.performance, best.chromosome.get_gene_val_list()))

    for i in range(100):
        population.next_generation()
        best = population.get_best_individual()
        print("Generation {}\t performace: {}\nGenes:{}\n".format(i, best.performance, best.chromosome.get_gene_val_list()))