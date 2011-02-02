#include "hmm.h"
#include <iostream>
#include<stdlib.h>
#include<math.h>
#include<fstream>
#include<sstream>
#include<cstring>

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		cout << "Too few argument" << endl;

	}
	else
	{
		int count = 0;
		int step = 1;
		if (argc == 5)
		{
			count = atoi(argv[4]);
		}
		if (argc == 6)
		{
			step = atoi(argv[5]);
		}
		int iterations = atoi(argv[3]);
		char * HMMFile = argv[1];
		char buf[5];
		string iter = "-iter";
		while (count < iterations)
		{
			count += step;
			stringstream s;
			s << count;
			string str = s.str();
			string name = HMMFile + iter + str;
			char* file = new char[name.size() + 1];
			strcpy(file, name.c_str());
			//cout << "Loading HMM from file: " << HMMFile << endl;
			HMM h(file);
			double entropy = 0;
			double perplexity;
			string line;
			ifstream testFile(argv[2]);
			int tokenCount = 0;
			while (getline(testFile, line))
			{
				vector<int> words;
				stringstream ss(line);
				string buf;
				while (ss >> buf)
				{
					words.push_back(atoi(buf.c_str()));
					tokenCount++;
				}
				double p = h.forwardAlgorithmScaled(words);
				entropy += p;
				if (p != p)
				{
					cout << "ERR : NUMERICAL ERROR p = " << p << HMMFile
							<< endl;
					return 0;
				}
			}
			perplexity = pow(2.0, (-1.0 / tokenCount) * entropy);
			//cout << "The perplexity of file: " << argv[2] << " is " << perplexity <<endl;
			cout << perplexity << endl;
		}
	}

	return 0;
}
