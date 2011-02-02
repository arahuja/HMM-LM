/*
 *  basicProbability.cpp
 *  
 *
 *  Created by Arun Ahuja on 3/9/10.
 *
 */

#include"math.h"
#include "basicProbability.h"
#include<vector>
#include<map>

using std::map;
using std::list;
using std::vector;
using std::pair;

double computeEntropy(double * dist, int n)
{
	double sum = 0.0;
	for (int i =0; i < n; i++)
	{
		if ( dist[i]  != 0)
			sum -= dist[i] * log2(dist[i]);
	}
	
	return sum;
}

double euclideanDistance(double * v1, double* v2, int n)
{
	double eucDist =0;
	for ( int i =0; i < n; i++)
	{	
		eucDist += pow((v1[i] - v2[i]), 2);
	}

	return eucDist;
}
double symmetricKLDivergence(double* dist1, double* dist2, int n)
{
	return (klDivergence(dist1, dist2, n) + klDivergence(dist2, dist1, n));
}

double lambdaKLDivergence(double* dist1, double* dist2, int n, double lambda)
{
	double * mixture;
	mixture = new double[n];
	mixDistributions(dist1, dist2, mixture, n, lambda);
	double lambdaDiv1 = lambda*klDivergence(dist1, mixture, n);
	double lambdaDiv2 = (1-lambda) * klDivergence(dist2, mixture, n);
	delete [] mixture;
	
	return lambdaDiv1 + lambdaDiv2;
}

double* mixDistributions(double* dist1, double* dist2, double* mix, int n, double lambda)
{
	for (int i = 0; i < n; i++)
	{
		mix[i] = lambda*dist1[i] + (1-lambda)*(dist2[i]);
	}
}

double jsDivergence(double* dist1, double* dist2, int n)
{
	return lambdaKLDivergence(dist1, dist2, n, .5);
}

double klDivergence(double* p, double* q, int n)
{
	double result = 0;
	for (int i = 0;  i< n; i++)
	{
		if ( dist1[i]  != 0 && dist2[i] != 0)
			result += p[i] * log2(p[i] / q[i]);
	}
	
	return result;
}

double cosineSimilarity(double* p, double* q, int n)
{
	double result = 0;
	double sumDist1 = 0;
	double sumDist2 = 0;
	for (int i = 0;  i< n; i++)
	{
		if ( p[i]  != 0 && q[i] != 0)
		{
			sumDist1 += pow(p[i], 2);
			sumDist2 += pow(q[i], 2);
			result += p[i]*q[i];
		}
	}
	return result / (sqrt(sumDist1) * sqrt(sumDist2));
}


double klDivergence(double * p, double * q, vector<int> contextList)
{
	double result = 0;
	for (int i = 0;  i< contextList.size(); i++)
	{
		int contextId = contextList[i];
		if ( p[contextId]  != 0 && q[contextId] != 0)
			result += p[contextId] * log2(p[contextId] / q[contextId]);
	}
	
	return result;
}

double computeAUC(vector< pair<int, double> > rankedList, map< int, bool> truthValMap)
{
	double score = 0.0;
	int numCorrect = 0;
	double lastPrecision = 1;
	double currentPrecision = 1.0;
	for( int i =0; i < rankedList.size(); i++)
	{
		int extraction = rankedList[i].first;
		bool truthVal = truthValMap[extraction];
		if (truthVal == 1)
		{
			numCorrect++;
			currentPrecision = numCorrect / (i + 1.0);
			score += (currentPrecision + lastPrecision) / 2;
			lastPrecision = currentPrecision;
		}
	}
	
	return score/numCorrect;
}
