#   Travelling Umpire Problem
#	Jakub Svoboda
#   xsvobo0z@stud.fit.vutbr.cz
#	21.4.2020
#	Brno University of Technology, Faculty of Informatics


import time
import datetime
import os
import numpy as np
import sys
import argparse
import csv
import re
import random
import itertools
import statistics
import numpy as np
import cProfile
from multiprocessing import Process
import copy


class Problem:
	def __init__(self, nTeams, dists, opponents, args):
		self.nTeams = nTeams
		self.dists = dists
		self.opponents = opponents
		self.d1 = args.d1
		self.d2 = args.d2
		self.createTournament()
		self.constraintPenalty = 100000

	def createTournament(self):
		self.tournament = []								#Create the tournament object with proper size
		for _ in range(0, 4*(self.nTeams//2)-2):
			self.tournament.append([])
		
		for slotIdx, _  in enumerate(self.tournament):		#Fill in the data
			for recordIdx in range(0, self.nTeams):
				gamePair = [recordIdx+1, self.opponents[slotIdx][recordIdx]]
				if gamePair[0] > gamePair[1]:				#home team goes first
					gamePair = [gamePair[1], gamePair[0]]
				gamePair[0]	= abs(gamePair[0])
				gamePair[1]	= abs(gamePair[1])
				
				if gamePair not in self.tournament[slotIdx] and [gamePair[1], gamePair[0]] not in self.tournament[slotIdx]:
					self.tournament[slotIdx].append(gamePair)

class Solution:

	def __init__(self, problem, create=True):
		self.problem = problem
		self.umpires = []
		if create:
			self.createUmpires()

	def mutate(self):
		slotIndex = random.randint(0, len(self.umpires)-1)
		mutatedIndex = random.randint(0, len(self.umpires[0])-1)
		mutatedIndex2 = random.randint(0, len(self.umpires[0])-1)
		while mutatedIndex == mutatedIndex2:
			mutatedIndex2 = random.randint(0, len(self.umpires[0])-1)
		slot = self.umpires[slotIndex]
		game1 = slot[mutatedIndex]
		game2 = slot[mutatedIndex2]
		slot[mutatedIndex2] = game1
		slot[mutatedIndex] = game2
		self.umpires[slotIndex] = slot

	def myMutate(self):
		mutatedIndex = random.randint(0, len(self.umpires)-1)
		mutatedIndex2 = random.randint(0, len(self.umpires)-1)
		tmp = self.umpires[mutatedIndex]
		self.umpires[mutatedIndex] = self.umpires[mutatedIndex2]
		self.umpires[mutatedIndex2] = tmp
		

	def cost(self, skipDist=False):
		distanceCost = 0
		if not skipDist:
			distanceCost = self.getDistanceCost()	
		rule3cost = self.getRule3Violations()	
		rule4cost = self.getRule4Violations()
		rule5cost = self.getRule5Violations()
		self.myCost = distanceCost+rule3cost+rule4cost+rule4cost+rule5cost
		return self.myCost

	def getRule3Violations(self):
		cost = 0
		for umpireIdx in range(0,len(self.umpires[0])):					#calcualte cost for each umpire
			myGames = []													#get list of home venues:
			for slotIdx in range(0,len(self.umpires)):				
				myGames.append(self.problem.tournament[slotIdx][self.umpires[slotIdx][umpireIdx]])
			myHomeGames = column(myGames,0)
			homeGamesSet = set(myHomeGames)
			unvisited = self.problem.nTeams - len(homeGamesSet)
			cost += self.problem.constraintPenalty * unvisited
		return cost

	def getRule4Violations(self):
		violations = 0 
		lst = self.problem.tournament
		slotsWidth = self.problem.nTeams//2 - self.problem.d1
		for umpireIdx in range(0,len(self.umpires[0])):				#for each umpire
			myGames = []
			for slotIdx, _ in enumerate(self.umpires):				#for each slot	
				myGames.append(lst[slotIdx][self.umpires[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)	
			for i in range(0, len(self.umpires)-slotsWidth+1):
				sublist = myHomeTeams[i:i+slotsWidth]
				violations += len(sublist) - len(set(sublist))
		return violations * self.problem.constraintPenalty		

	def getRule5Violations(self):
		violations = 0 
		lst = self.problem.tournament
		consecutiveAllowed = self.problem.nTeams//4 - self.problem.d2
		for umpireIdx in range(0,len(self.umpires[0])):					#for each umpire
			myGames = []
			for slotIdx, _ in enumerate(self.umpires):				#for each slot	
				myGames.append(lst[slotIdx][self.umpires[slotIdx][umpireIdx]])
			teamsEncountered = list(itertools.chain(*myGames))
			for i in range(0, (len(self.umpires)-consecutiveAllowed)*2+1, 2):
				sublist = teamsEncountered[i:i+(consecutiveAllowed)*2]
				violations += len(sublist) - len(set(sublist))
		return violations * self.problem.constraintPenalty


	def getDistanceCost(self):
		distanceCost = 0
		for umpireIdx in range(0,len(self.umpires[0])):					#calcualte cost for each umpire
			myGames = []													#get list of home venues:
			for slotIdx in range(0,len(self.umpires)):				
				myGames.append(self.problem.tournament[slotIdx][self.umpires[slotIdx][umpireIdx]])
			myHomeGames = column(myGames,0)
			lastVenue = None
			for venueIdx,venue in enumerate(myHomeGames):
				if(venueIdx == 0):
					lastVenue = venue
					continue
				distanceCost += self.problem.dists[venue-1][lastVenue-1]
				lastVenue = venue	
		return distanceCost	


	def createUmpires(self):
		self.umpires = []
		alloptions = []
		for i in range(0, 4*(self.problem.nTeams//2)-2):
			perms = list(itertools.permutations(list(range(0,self.problem.nTeams//2))))
			perms = list(map(list, perms))
			random.shuffle(perms)
			alloptions.append(perms)

		i = 0
		while i < 4*(self.problem.nTeams//2)-2:				#for each time slot
			if i == 0:											#first slot generate randomly
				self.umpires.append(alloptions[i].pop())
				i+=1
			else:												#following games generation:
				while(True):
					if len(alloptions[i]) == 0:
						perms = list(itertools.permutations(list(range(0,self.problem.nTeams//2))))
						perms = list(map(list, perms))
						alloptions[i] = perms
						self.umpires.pop()
						i-=1	
						if i<0:
							exit(1)
						continue
					slot = alloptions[i].pop()
					if not self.breaksD1Creation(self.umpires + [slot]) and not self.breaksD2Creation(self.umpires + [slot]):
						self.umpires.append(slot)
						i+=1
						break


	def breaksD1Creation(self, lst):	
		maxHomeRunAllowed = self.problem.nTeams//2 - self.problem.d1
		for umpireIdx in range(0,len(lst[0])):		#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(lst):
				myGames.append(self.problem.tournament[slotIdx][lst[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)	
			for i in range(0, (len(lst)-maxHomeRunAllowed)*2+1, 2):
				sublist = myHomeTeams[i:i+(maxHomeRunAllowed)]
				if len(sublist) != len(set(sublist)):
					return True	
		return False

	def breaksD2Creation(self, lst):
		maxSequenceAllowed = (self.problem.nTeams//4) - self.problem.d2
		for umpireIdx in range(0,len(lst[0])):		#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(lst):
				myGames.append(self.problem.tournament[slotIdx][lst[slotIdx][umpireIdx]])
			teamsEncountered = []
			for sublist in myGames:
				for item in sublist:
					teamsEncountered.append(item)
			for i in range(0, (len(lst)-maxSequenceAllowed)*2+1, 2):
				sublist = teamsEncountered[i:i+(maxSequenceAllowed)*2]
				if len(sublist) != len(set(sublist)):
					return True	
		return False

	def printGames(self):
		for slotIdx, slot in enumerate(self.umpires):
			print(slotIdx, end=":")
			for _, umpire in enumerate(slot):
				print( "  \t(", self.problem.tournament[slotIdx][umpire], ")", end="")	
			print("")	


# Check if given list contains any duplicates 
def checkIfDuplicates(listOfElems):
	if len(listOfElems) == len(set(listOfElems)):
		return False
	else:
		return True

def column(matrix, i):
	return [row[i] for row in matrix]	


#check the validity of arguments
def checkArgs(args):	
	if args.d1 < 0: 
		raise ValueError("D1 must be >= 0.")
	if args.d2 < 0: 
		raise ValueError("D2 must be >= 0.")
	if not os.path.isfile(args.inputPath):				
		raise ValueError("The file " + args.inputPath + " does not exist.")
	return args	

#sets up an argument parser	
def setArguments(args):
	parser = argparse.ArgumentParser(description = "GA algorithm solver for the travelling umpire problem.")
	parser.add_argument("-d1", "--d1", "-D1", "--D1", help = "The D1 constraint parameter.", default = 0, type = int)
	parser.add_argument("-d2", "--d2", "-D2", "--D2", help = "The D2 constraint parameter.", default = 0, type = int)
	parser.add_argument("-i", "--inputPath", help = "File in which input dataset is stored.", default = "dataset/umps8.txt", type = str)

	args = parser.parse_args()
	args = checkArgs(args)
	return args


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

#Parses the really odd input format, because csv files are too difficult for some people I guess
def parseInput(path):
	nTeams = dists = opponents = None
	f = open(path,"r")
	lines = f.readlines() 
	concatenated = "".join(lines) 	
	nTeams = int(re.search("nTeams=(\d*|\s*)*;", concatenated)[0][7:-1])

	dists = re.search("dist(\s*)=(.*?|\n*?|\s*)*;", concatenated)
	dists = dists.group(0)[6:-2]
	dists = dists.replace("[", "")
	dists = dists.replace("]","")
	dists = dists.replace("\n","")
	dists = list(map(int, dists.split()))
	dists = list(chunks(dists,nTeams))

	opponents = re.search("opponents(\s*)=(.*?|\n*?|\s*|-*)*;", concatenated)
	opponents = opponents.group(0)[10:-2]
	opponents = opponents.replace("[", "")
	opponents = opponents.replace("]","")
	opponents = opponents.replace("\n","")
	opponents = list(map(int, opponents.split()))
	opponents = list(chunks(opponents,nTeams))
	return nTeams, dists, opponents


def run(nTeams, dists, opponents, args):
	popSize = 500
	mutationChance = 5									#5% is the textbook mutation chance
	problem = Problem(nTeams, dists, opponents, args)
	population = []
	for i in range(0, popSize):							#500 initial population
		print("Generating population:", i+1,"of 500\r", end="")
		population.append(Solution(problem))
	print("")
	bestSolution = copy.deepcopy(population[0])

	for epochNum in range(1,10000):
		print("Epoch:", epochNum, end = '')
		costs = [p.cost() for p in population]
		minindex = costs.index(min(costs))
		if bestSolution.cost() > population[minindex].cost():	#save best solution
			bestSolution = copy.deepcopy(population[minindex])
		costsAverage = statistics.mean(costs)	
		umpires = []
		for j in range (0,popSize):
			umpires.append(population[j].umpires)
		umpires = set(str(x) for x in umpires) 

		print("\tBest:", bestSolution.cost(), "\tR3 cost:", bestSolution.getRule3Violations(), "\tR4 cost:", 
		bestSolution.getRule4Violations(), "\tR5 cost:", bestSolution.getRule5Violations(), "\tAverage:", int(costsAverage), len(umpires))	
		if epochNum % 10 == 0:
			print("Best schedule found:  <Umpire1> ... <UmpireN> ")
			bestSolution.printGames()
		population.sort(key=lambda x: x.myCost)	
		parents = population[0: popSize//2]
		eliminates = population[popSize//2:popSize]
		random.shuffle(eliminates)
		random.shuffle(parents)		
		#children = [None for _ in range(popSize//2)]
		for i in range(0, popSize//4):			
			eliminates[i*2] = crossover(parents[i*2], parents[i*2+1], eliminates[i*2], umpires)
			eliminates[i*2+1] = crossover(parents[i*2], parents[i*2+1], eliminates[i*2+1], umpires)
		#population = parents + children			
		#print(len(parents), len(eliminates))
		population = parents + eliminates
		for i in range(0, popSize):
			#print(i, len(population))
			rand = random.randint(0,100)
			if rand < mutationChance:
				population[i].mutate()
	return None

def crossover(parent1,parent2, eliminate, umpires):
	perms = list(itertools.permutations(list(range(0, parent1.problem.nTeams//2))))
	perms = list(map(list, perms))
	split = random.randint(1, len(parent1.umpires)-1)	
	start = parent1.umpires[0:split] 
	end = parent2.umpires[split:]
	candidates = []
	for p in perms:
		npa = np.asarray(end, dtype=np.int32)
		rearranged = npa[:,p]		
		candidates.append(rearranged.tolist())
		
	pop = [Solution(parent1.problem, create=False) for x in list(range(0, len(perms)))]

	for idx,_ in enumerate(pop):
		pop[idx].umpires = start + candidates[idx]

	[p.cost() for p in pop]
	pop.sort(key=lambda x:x.myCost)
	for p in pop:
		umpStr = str(p.umpires)
		if umpStr in umpires:			#it is an already existing child
			continue
		else:
			return p
	return eliminate		#if no unique child is found, eliminate survives

def crossover_old(parent1, parent2):
	split = random.randint(1, len(parent1.umpires)-1)	
	start = parent1.umpires[0:split] 
	end = parent2.umpires[split:]

	perms = list(itertools.permutations(list(range(0, parent1.problem.nTeams//2))))
	perms = list(map(list, perms))

	candidates = []

	for p in perms:
		npa = np.asarray(end, dtype=np.int32)
		rearranged = npa[:,p]		
		candidates.append(rearranged.tolist())
		
	pop = [Solution(parent1.problem, create=False) for x in list(range(0, len(perms)))]

	for idx,_ in enumerate(pop):
		pop[idx].umpires = start + candidates[idx]

	costs = [p.cost() for p in pop]
	minindex = costs.index(min(costs))
	return pop[minindex]

def crossoverExperimental(parent1, parent2):
	split = random.randint(1, len(parent1.umpires)-1)
	child = Solution(parent1.problem, create=False)
	umpires = parent1.umpires[0:split] + parent2.umpires[split:]
	child.umpires = umpires 
	return child


def main(args=None):
	if args is None:				
		args = sys.argv[1:]
	args = setArguments(args)

	random.seed(42)

	nTeams, dists, opponents = parseInput(args.inputPath)

	start = time.time()
	run(nTeams, dists, opponents, args)
	end = time.time()
	print("Time:", end-start)

if __name__== "__main__":
	main()