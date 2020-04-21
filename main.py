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
	print(nTeams)

	dists = re.search("dist=(.*?|\n*?|\s*)*;", concatenated)
	dists = dists.group(0)[6:-2]
	dists = dists.replace("[", "")
	dists = dists.replace("]","")
	dists = dists.replace("\n","")
	dists = list(map(int, dists.split()))
	print(list(chunks(dists,nTeams)))

	opponents = re.search("opponents=(.*?|\n*?|\s*|-*)*;", concatenated)
	opponents = opponents.group(0)[10:-2]
	opponents = opponents.replace("[", "")
	opponents = opponents.replace("]","")
	opponents = opponents.replace("\n","")
	opponents = list(map(int, opponents.split()))
	print(list(chunks(opponents,nTeams)))

	return nTeams, dists, opponents






def main(args=None):
	if args is None:				
		args = sys.argv[1:]
	args = setArguments(args)

	nTeams, dists, opponents = parseInput(args.inputPath)

	out = None





if __name__== "__main__":
	main()