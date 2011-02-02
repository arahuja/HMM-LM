#include "hmm.h"
#include <iostream>
#include<stdlib.h>
#include<math.h>
#include<fstream>
#include<sstream>

int main(int argc, char** argv){
	if (argc < 3) {
		cout << "expected 2 arguments : ./testHMM <HMMfile> <testFile>" << endl;
	} else {
		char * HMMFile = argv[1];
		HMM h(HMMFile);
		double entropy =0;
		double perplexity;
		string line;
		ifstream testFile(argv[2]);
		int tokenCount = 0;
		while(getline(testFile, line)){
			cout << line << " ";
			vector<int> words;
			stringstream ss(line);
			string buf;
			while ( ss >> buf){
				words.push_back(atoi(buf.c_str()));
				tokenCount++;
				
			}
			double p = h.forwardAlgorithmScaled(words);
			
			cout << p << " " << p/words.size() << endl;
			entropy += p;
			if (p != p)
				return 0;
		}		
		perplexity = pow(2.0, (-1.0/tokenCount)*entropy);
		cout << "The perplexity of file: " << argv[2] << " is " << perplexity <<endl;
	}

	return 0;
}
