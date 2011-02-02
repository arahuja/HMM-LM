#include "hmm.h"
#include <iostream>
#include<stdlib.h>
#include<math.h>
#include<fstream>
#include<sstream>
#include<cstring>
#include "utilities.h"
#include<vector>
#include<map>

#include "basicProbability.h"

int main(int argc, char** argv){
		

	char * HMMFile = argv[1];
	char * vocabFile = argv[2];
	int nearestK = atoi(argv[3]);
	ifstream wordFile(argv[4]);
	map<string, int> stringToIntvocabMap;
	map<int, string> intToStringvocabMap;
	map<int, int> wordCount;
	loadVocabFromFile(vocabFile, &stringToIntvocabMap, &intToStringvocabMap, &wordCount);	

	HMM h(argv[1]);
	string line;
	while(getline(wordFile, line))
	{
		int word = stringToIntvocabMap[line];
		vector<pair<int, double> > neighbors;
		for (int neighbor = 0; neighbor < h.getNumObs(); neighbor++)
		{
			if (neighbor != word)
			{
				double distance = jsDivergence(h.condstate_dist[word], h.condstate_dist[neighbor], h.getNumStates());
				neighbors.push_back(pair<int, double>(neighbor, distance));
			}
		}
		std::sort(neighbors.begin(), neighbors.end(), sort_scorepairLESSTHAN);
		cout << intToStringvocabMap[word] << " " << wordCount[word] << endl;
		for (int i = 0; i < nearestK; i++)
		{
			cout << intToStringvocabMap[neighbors[i].first] << " " << neighbors[i].second << " ";
		}
		cout << endl;
	
	}	
	wordFile.close();
		
	return 0;
}
