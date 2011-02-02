#include "hmm.h"
#include <iostream>
#include<stdlib.h>
#include<math.h>
#include<fstream>
#include<sstream>
#include<cstring>

int main(int argc, char** argv){
	HMM h(argv[1]);
	double entropy =0;
	double perplexity;
	string line;
	ifstream testFile(argv[2]);
	int tokenCount = 0;
	while(getline(testFile, line)){
		vector<int> words;
		stringstream ss(line);
		string buf;
		while ( ss >> buf){
			words.push_back(atoi(buf.c_str()));
			tokenCount++;	
		}
		double p = h.forwardAlgorithmScaled(words);
		entropy += p;
		if (p != p){	
			cout << "ERR : NUMERICAL ERROR p = "<< p  << HMMFile << endl;
			return 0;
		}
		cout << p << endl;
	}		
	perplexity = pow(2.0, (-1.0/tokenCount)*entropy);
	//cout << "The perplexity of file: " << argv[2] << " is " << perplexity <<endl;
	cout << perplexity <<endl;

	return 0;
}
