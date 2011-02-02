/*
 *  basicProbability.h
 *  
 *
 *  Created by Arun Ahuja on 3/9/10.
 */

#include<vector>
#include<map>

double computeEntropy(double * dist, int n);
double* mixDistributions(double* dist1, double* dist2, double* mix, int n,
		double lambda);
double klDivergence(double* dist1, double* dist2, int n);
double symmetricKLDivergence(double* dist1, double* dist2, int n);
double jsDivergence(double* dist1, double* dist2, int n);
double lambdaKLDivergence(double* dist1, double* dist2, int n, double lambda);
double euclideanDistance(double* dist1, double* dist2, int n);
double klDivergence(double * dist1, double * dist2,
		std::vector<int> contextList);
double cosineSimilarity(double* dist1, double* dist2, int n);
double computeAUC(std::vector<std::pair<int, double> > rankedList, std::map<
		int, bool> truthValMap);
