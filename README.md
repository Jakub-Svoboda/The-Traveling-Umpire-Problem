# The-Traveling-Umpire-Problem
## Simulation for the SNT class at FIT VUTBR.

Name: Jakub Svoboda 

Login: xsvobo0z

Email: xsvobo0z@stud.fit.vutbr.cz

Date: 23.4.2020


This project implements the algorithm explained in paper: Michael A. Trick, Hakan Yildiz - Locally Optimized Crossover for the Traveling Umpire Problem

How to use: 

Run: 	

>	python3 main.py		


You can use optional parameters:
>	python3 main.py -d1 \<NumD1\> -d2 \<NumD2\> --i \<path\>

where NumD1 and NumD2 are parameters describing the constraints of the problem (default d1=d2=0) and path is the path to the input txt file (default dataset/umps8.txt).

The program will run for 1000 epochs or until stopped.


The dataset is obtainable from https://benchmark.gent.cs.kuleuven.be/tup/en/ but be aware that the provided zip archive does not contain all tournament data used in the original paper.

