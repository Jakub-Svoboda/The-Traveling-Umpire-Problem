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
		#Create the tournament object with proper size
		#tmpGame = [None] * (self.nTeams//2)
		self.tournament = []
		for _ in range(0, 4*(self.nTeams//2)-2):
			self.tournament.append([])
		#Fill in the data
		for slotIdx, _  in enumerate(self.tournament):
			for recordIdx in range(0, self.nTeams):
				gamePair = [recordIdx+1, self.opponents[slotIdx][recordIdx]]
				if gamePair[0] > gamePair[1]:						#home team goes first
					gamePair = [gamePair[1], gamePair[0]]
				gamePair[0]	= abs(gamePair[0])
				gamePair[1]	= abs(gamePair[1])
				
				if gamePair not in self.tournament[slotIdx] and [gamePair[1], gamePair[0]] not in self.tournament[slotIdx]:
					#print(gamePair, "not in ", self.tournament)
					self.tournament[slotIdx].append(gamePair)
				#self.tournament[slotIdx][recordIdx] = [recordIdx+1, self.opponents[slotIdx][recordIdx]]
		#print(self.tournament)

class Solution:

	def __init__(self, problem, create=True):
		self.problem = problem
		self.umpires = []
		if create:
			self.createUmpires()

	def cost(self):
		distanceCost = self.getDistanceCost()	
		#print("Distance cost", distanceCost)
		rule3cost = self.getRule3Violations()	
		#print("rule 3 cost", rule3cost)
		rule4cost = self.getRule4Violations()
		#print("Rule 4 cost:", rule4cost)
		rule5cost = self.getRule5Violations()
		#print("Rule 5 cost:", rule5cost)
		#print("Total cost:", distanceCost+rule3cost+rule4cost+rule4cost+rule5cost)
		self.myCost = distanceCost+rule3cost+rule4cost+rule4cost+rule5cost
		return distanceCost+rule3cost+rule4cost+rule4cost+rule5cost

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
			for slotIdx, slot in enumerate(self.umpires):				#for each slot	
				myGames.append(lst[slotIdx][self.umpires[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)	
			#print(myHomeTeams)
			for i in range(0, len(self.umpires)-slotsWidth+1):
				sublist = myHomeTeams[i:i+slotsWidth]
				#print(sublist)
				#print(i, slotsWidth, i+slotsWidth)
				violations += len(sublist) - len(set(sublist))
		return violations * self.problem.constraintPenalty		

	def getRule5Violations(self):
		violations = 0 
		lst = self.problem.tournament
		consecutiveAllowed = self.problem.nTeams//4 - self.problem.d2
		for umpireIdx in range(0,len(self.umpires[0])):					#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(self.umpires):				#for each slot	
				myGames.append(lst[slotIdx][self.umpires[slotIdx][umpireIdx]])
			teamsEncountered = []
			for sublist in myGames:
				for item in sublist:
					teamsEncountered.append(item)
			#print("Allowed",consecutiveAllowed)
			#self.printGames()
			for i in range(0, (len(self.umpires)-consecutiveAllowed)*2+1, 2):
				sublist = teamsEncountered[i:i+(consecutiveAllowed)*2]
				#print(i, sublist)
				violations += len(sublist) -  len(set(sublist))
		return violations * self.problem.constraintPenalty


	def getDistanceCost(self):
		distanceCost = 0
		for umpireIdx in range(0,len(self.umpires[0])):					#calcualte cost for each umpire
			myGames = []													#get list of home venues:
			for slotIdx in range(0,len(self.umpires)):				
				myGames.append(self.problem.tournament[slotIdx][self.umpires[slotIdx][umpireIdx]])
			myHomeGames = column(myGames,0)

			#print(myGames)
			#print(myHomeGames)

			lastVenue = None
			for venueIdx,venue in enumerate(myHomeGames):
				if(venueIdx == 0):
					lastVenue = venue
					continue

				#print("From:", venue, "to", lastVenue, "it is", self.problem.dists[venue-1][lastVenue-1])
				distanceCost += self.problem.dists[venue-1][lastVenue-1]
				lastVenue = venue
		return distanceCost	


	def createUmpires(self):
		self.umpires = []
		alloptions = []
		for i in range(0, 4*(self.problem.nTeams//2)-2):
			perms = list(itertools.permutations(list(range(0,self.problem.nTeams//2))))
			perms = list(map(list, perms))
			#print(perms)
			random.shuffle(perms)
			alloptions.append(perms)
		#print(alloptions)	

		i = 0
		while i < 4*(self.problem.nTeams//2)-2:				#for each time slot

			#print("LEVEL:", i)
			if i == 0:											#first slot generate randomly
				self.umpires.append(alloptions[i].pop())
				i+=1
			else:												#following games generation:
				while(True):
					if len(alloptions[i]) == 0:
						#print("level",i, "exhausted, backtracking")
						perms = list(itertools.permutations(list(range(0,self.problem.nTeams//2))))
						perms = list(map(list, perms))
						alloptions[i] = perms
						self.umpires.pop()
						i-=1	
						if i<0:
							exit(1)
						continue
					slot = alloptions[i].pop()

					#print(slot)
					'''for slotIdx, s in enumerate(self.umpires + [slot]):
						print(slotIdx, s, end="")
						for _, umpire in enumerate(s):
							print( "  \t(", self.problem.tournament[slotIdx][umpire], ")", end="")	
						print("")
					'''

					if not self.breaksD1Creation(self.umpires + [slot]) and not self.breaksD2Creation(self.umpires + [slot]):
						self.umpires.append(slot)
						#print("Appended", i, slot )
						#print("Total:", self.umpires)
						#self.printGames()
						#print("-------")
						i+=1
						break


		#self.printGames()
		#exit()

	def breaksD1Creation(self, lst):
	#	for slotIdx, slot in enumerate(lst):
		#	print(slotIdx, slot, end="")
	#		for _, umpire in enumerate(slot):
	#			print( "  \t(", self.problem.tournament[slotIdx][umpire], ")", end="")	
	#		print("")	
		maxHomeRunAllowed = self.problem.nTeams//2 - self.problem.d1
		for umpireIdx in range(0,len(lst[0])):		#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(lst):
				myGames.append(self.problem.tournament[slotIdx][lst[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)	
			for i in range(0, (len(lst)-maxHomeRunAllowed)*2+1, 2):
				sublist = myHomeTeams[i:i+(maxHomeRunAllowed)]
				#print(maxHomeRunAllowed)
				#print(umpireIdx,i,"sublists:", sublist)
				if len(sublist) != len(set(sublist)):
					#print("D1 broken")
					return True	
		return False

	def breaksD2Creation(self, lst):
	#	for slotIdx, slot in enumerate(lst):
	#		print(slotIdx, slot, end="")
	#		for _, umpire in enumerate(slot):
	#			print( "  \t(", self.problem.tournament[slotIdx][umpire], ")", end="")	
	#		print("")
		

		#print(lst)
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
				#print(umpireIdx,i,"sublists:", sublist)
				if len(sublist) != len(set(sublist)):
					#print("----D2 broken")
					return True	

		return False

	def printGames(self):
		for slotIdx, slot in enumerate(self.umpires):
			print(slotIdx, slot, end="")
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
		raise ValueError("The file " + args.weights + " does not exist.")
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
	#print(re.search("nTeams=(\d*|\s*)*;", concatenated)[0][7:-1])
	nTeams = int(re.search("nTeams=(\d*|\s*)*;", concatenated)[0][7:-1])
	#print(nTeams)

	dists = re.search("dist(\s*)=(.*?|\n*?|\s*)*;", concatenated)
	dists = dists.group(0)[6:-2]
	dists = dists.replace("[", "")
	dists = dists.replace("]","")
	dists = dists.replace("\n","")
	dists = list(map(int, dists.split()))
	dists = list(chunks(dists,nTeams))
	#print(list(chunks(dists,nTeams)))

	opponents = re.search("opponents(\s*)=(.*?|\n*?|\s*|-*)*;", concatenated)
	opponents = opponents.group(0)[10:-2]
	opponents = opponents.replace("[", "")
	opponents = opponents.replace("]","")
	opponents = opponents.replace("\n","")
	opponents = list(map(int, opponents.split()))
	opponents = list(chunks(opponents,nTeams))
	#print(list(chunks(opponents,nTeams)))

	return nTeams, dists, opponents



def run(nTeams, dists, opponents, args):
	popSize = 500
	mutationChance = 5		# int/100
	problem = Problem(nTeams, dists, opponents, args)
	population = []
	for i in range(0, popSize):							#500 initial population
		print("Generating population:", i+1,"of 500\r", end="")
		population.append(Solution(problem))
	print("")
	bestSolution = population[0]

	for epochNum in range(1,1000):
		print("Epoch:", epochNum, end = '')
		population.sort(key=Solution.cost)
		if bestSolution.myCost > population[0].myCost:	#save best solution
			bestSolution = population[0]
		costsAverage = statistics.mean([p.myCost for p in population])	
		#unique = []
		#for j in range (0,popSize):
		#	unique.append(population[j].umpires)
		#unique = set(str(x) for x in unique) 

		print("\tBest:", bestSolution.myCost, "\tR3 cost:", bestSolution.getRule3Violations(), "\tR4 cost:", 
		bestSolution.getRule4Violations(), "\tR5 cost:", bestSolution.getRule5Violations(), "\tAverage:", int(costsAverage))	
		if epochNum % 100 == 0:
			print(bestSolution.printGames())
		parents = population[0: popSize//2]
		random.shuffle(parents)		
		children = 250 * [None]
		for i in range(0, popSize//4):			
			#print(i)
			children[i*2] = crossover(parents[i*2], parents[i*2+1])
			children[i*2+1] = crossover(parents[i*2], parents[i*2+1])
		population = parents + children			
		for i in range(0, popSize):
			rand = random.randint(0,100)
			if rand < mutationChance:
				mutate(population[i])
	return None

def crossover(parent1, parent2):
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
		
	pop = []
	for i in range(0, len(perms)):
		pop.append(Solution(parent1.problem, create=False))	

	for idx,p in enumerate(pop):
		pop[idx].umpires = start + candidates[idx]
	#costs = [c.cost() for c in pop]	
	pop.sort(key=Solution.cost)
	pop[0].umpires = start + end
	return pop[0]

def mutate(solution):
	mutatedIndex = random.randint(0, len(solution.umpires)-1)
	mutatedIndex2 = random.randint(0, len(solution.umpires)-1)
	tmp = solution.umpires[mutatedIndex]
	solution.umpires[mutatedIndex] = solution.umpires[mutatedIndex2]
	solution.umpires[mutatedIndex2] = tmp


def main(args=None):
	if args is None:				
		args = sys.argv[1:]
	args = setArguments(args)

	random.seed(42)

	nTeams, dists, opponents = parseInput(args.inputPath)

	start = time.time()
	bestSolution = run(nTeams, dists, opponents, args)
	end = time.time()
	print("Time:", end-start)





if __name__== "__main__":
	main()