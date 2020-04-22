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

	def __init__(self, problem):
		self.problem = problem
		self.createUmpires()

	def cost(self):
		distanceCost = self.getDistanceCost()	
		print(distanceCost)
		rule3cost = self.getRule3Violations()	
		print(rule3cost)
		rule4cost = self.getRule4Violations()
		print(rule4cost)

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
		maxHomeRunAllowed = self.problem.nTeams//2 - self.problem.d1
		for umpireIdx in range(0,len(self.umpires[0])):				#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(self.umpires):				#for each slot	
				myGames.append(lst[slotIdx][self.umpires[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)	
			print(myHomeTeams)
			for i in range(0, len(self.umpires)-maxHomeRunAllowed):
				sublist = myHomeTeams[i:i+maxHomeRunAllowed+1]
				print(sublist)
				if len(set(sublist)) == 1:
					violations+=1	
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
		for i in range(0, 4*(self.problem.nTeams//2)-2):		#for each time slot
			if i == 0:											#first slot generate randomly
				slot = list(range(0,4))
				random.shuffle(slot)
				self.umpires.append(slot)
			else:												#following games generation:
				while(True):
					slot = list(range(0,4))
					random.shuffle(slot)
					if not self.breaksD1Creation(self.umpires + [slot]) and not self.breaksD2Creation(self.umpires + [slot]):
						self.umpires.append(slot)
						break

		self.printGames()

	def breaksD1Creation(self, lst):
		maxHomeRunAllowed = self.problem.nTeams//2 - self.problem.d1
		for umpireIdx in range(0,len(lst[0])):		#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(lst):
				myGames.append(self.problem.tournament[slotIdx][lst[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)	
			longestHomeRun = longest_repetition(myHomeTeams)
			if longestHomeRun > maxHomeRunAllowed:
				return True	
		return False

	def breaksD2Creation(self, lst):
		maxSequenceAllowed = (self.problem.nTeams//2) - self.problem.d2
		for umpireIdx in range(0,len(lst[0])):		#for each umpire
			myGames = []
			for slotIdx, slot in enumerate(lst):
				myGames.append(self.problem.tournament[slotIdx][lst[slotIdx][umpireIdx]])
			myHomeTeams = column(myGames,0)
			myAwayTeams = column(myGames,1)	
			longestHomeRun = longest_repetition(myHomeTeams)
			longestAwayRun = longest_repetition(myAwayTeams)
			if longestHomeRun > maxSequenceAllowed or longestAwayRun > maxSequenceAllowed:
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

def longest_repetition(iterable):
    """
    Return the number of longest cansecutive repetitions in `iterable`.
    If there are multiple such items, return the first one.
    If `iterable` is empty, return `None`.
    """
    longest_element = current_element = None
    longest_repeats = current_repeats = 0
    for element in iterable:
        if current_element == element:
            current_repeats += 1
        else:
            current_element = element
            current_repeats = 1
        if current_repeats > longest_repeats:
            longest_repeats = current_repeats
            longest_element = current_element
    return longest_repeats		

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
	problem = Problem(nTeams, dists, opponents, args)
	solution = Solution(problem)
	solution.cost()
	#solution2 = Solution(problem)

	return None


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