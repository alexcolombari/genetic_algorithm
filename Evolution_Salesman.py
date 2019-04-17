'''
    Evolution of a Salesman problem resolution using Genetic Algorithm with Machine Learning;
    Coded following this tutorial: https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35
    @Author: Alex Colombari
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np, random, operator

class City:  # In genetic algorithm, this is our representation of Gene.
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance (self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

class Fitness:  #  This is our function that tell us how good each route is (how short the distance is).
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0 , len(self.route)):
                fromCity = self.route[i]
                toCity = None
                
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    tocity = self.route[0]

                pathDistance += fromCity.distance(toCity)
                self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# ---------- Genetic Algorithm start ----------

def createRoute(cityList): # Creates an one individual.
    route = random.sample(cityList, len(cityList))
    return route

# Loop over createRoute def to create many routes for our population.

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

    # Determinate the fitness. Simulate the "survival of the fittest";
    # Use the Fitness to rank each individual in the population;
    # The output will be an ordered list with the route IDs and each associated fitness score.

def rankRoutes(population):
    fitnessResult = {}
    for i in range(0, len(population)):
        fitnessResult[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResult.items(), key = operator.itemgetter(1), reverse = True)

    # SELECT THE MATING POOL:
        # Select the parents that will be used to create the next generation

def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns = ["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cum_sum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

    # BREED FUNCTION:
        # With our mating pool created, we can create the next generation in a process called "crossover"

def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

        childP2 = [item for item in parent2 if item not in childP1]

        child = childP1 + childP2
        return child

def breedPopulation(matingpool, eliteSize):
    children = []
    lenght = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, lenght):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children

    # MUTATE:

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city1
            individual[swapWith] = city2

    return individual

def mutatePopulation(population, mutationRate):
    mutatePop = []
    for i in range(0, len(population)):
        mutateInd = mutate(population[ind], mutationRate)
        mutatePop.append(mutateInd)

    return mutatePop

    # REPEAT:
        # Function that produces a new generation.
        # First, we rank the routes in the current generation using "rankRoutes".
        # We then determine our potential parents by running the "selection" function,
        # which allows us to create the mating pool using the "matingPool" function.
        # Finally, we then create our new generation using the "breedPopulation" function
        # and then applying mutation using the "mutatePopulation" function.

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute


cityList = []
for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])
    
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.show()
