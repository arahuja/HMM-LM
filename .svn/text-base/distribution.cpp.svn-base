#include "distribution.h"
#include<stdlib.h>
#include<iostream>

using namespace std;
Distribution::Distribution(double *p, int len)
{
	
	n = len+1;	
	cum_dist = new double[n];

	if (n < 1)
		exit(0);
	
	cum_dist[0] = 0.0;
	
	//p[0]/total;

	for (int i = 1; i < n; i++)
	{
		cum_dist[i] = cum_dist[i-1] + p[i-1];
		//cout << "The dist "<< i-1 << " " <<cum_dist[i] << endl;
	}				
}

Distribution::Distribution()
{


}

Distribution::~Distribution()
{
	delete [] cum_dist;

}

int Distribution::generate_sample()
{

	double s = (rand()) / (RAND_MAX + 1.0);
//	s /= 100001;

/*
	int top = n-1;
	int bottom = 0;
	int k = (top+bottom)/2;
	while ( !(cum_dist[k] >= sample) && !(cum_dist[k+1] <= sample))
	{
		k = ( top + bottom ) / 2;
		if ( cum_dist[k] <= sample )
			bottom  = k + 1;
		else
			top = k - 1;
	}
*/

	for (int i = 1; i < n; i++)
	{
		if ((cum_dist[i-1] <= s) && (cum_dist[i] >= s))
			return i-1;
	}
	return n-2;
}


void Distribution::set_distribution(double* p, int n){

}
	


