
#include <map>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <fstream>
#include<iostream>
#include "utilities.h"

using namespace std;

bool sort_scorepairGREATER(const ScorePair& left, const ScorePair& right)
{
	return left.second > right.second;
}

bool sort_scorepairLESSTHAN(const ScorePair& left, const ScorePair& right)
{
	return left.second < right.second;
}

double** createMatrix(int m, int n)
{
	double** matrix;
	matrix = new double*[m];
	for (int i = 0; i < m; i++)
	{
		matrix[i] = new double[n];
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = 0;
		}
	}
	return matrix;
}

void createVocab(char * vocab, char * vocabOut)
{
	ofstream outputFile(vocabOut);
	ifstream inputFile(vocab);

	map<string, int> vocabMap;
	map<string, int> wordCount;
	int numWords = 0;
	string line;
	while (getline(inputFile, line))
	{ // for every sentence
		vector < string > words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(buf);
		}
		int len = words.size();
		for (int i = 0; i < len; i++)
		{
			string s = words[i];
			map<string, int>::iterator iter = vocabMap.find(s);
			if (iter == vocabMap.end())
			{
				numWords++;
				vocabMap.insert(pair<string, int> (s, numWords));
				wordCount.insert(pair<string, int> (s, 1));
			}
			else
			{
				int currCount = wordCount.find(s)->second;
				wordCount[s] = currCount + 1;
			}
		}
	}
	outputFile << numWords + 1 << endl;
	map<string, int>::iterator iter;
	for (iter = vocabMap.begin(); iter != vocabMap.end(); ++iter)
	{
		outputFile << iter->first << " " << iter->second << " " << wordCount[iter->first] << endl;
		//cout << iter->first << " " <<iter->second << wordCount[iter->first] << endl;
	}

	inputFile.close();
	outputFile.close();
}

void translateFile(char * vocab, char * input, char* output)
{
	ifstream vocabFile(vocab);
	ifstream inputFile(input);
	ofstream outputFile(output);

	map<string, int> vocabMap;
	string line;
	getline(vocabFile, line);
	int count = atoi(line.c_str());
	while (getline(vocabFile, line))
	{
		vector < string > words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(buf);
		}
		vocabMap.insert(pair<string, int> (words[0], atoi(words[1].c_str())));
	}

	while (getline(inputFile, line))
	{
		vector < string > words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(buf);
		}
		int len = words.size();
		for (int i = 0; i < len; i++)
		{
			map<string, int>::iterator iter = vocabMap.find(words[i]);
			if (iter != vocabMap.end())
			{
				outputFile << iter->second << " ";
			}
			else
			{
				outputFile << "0 ";
			}
		}
		outputFile << endl;
	}

	vocabFile.close();
	inputFile.close();
	outputFile.close();
}

void loadVocabFromFile(char* vocab, map<string, int> * stringToIntvocabMap, map<int, string> * intToStringvocabMap)
{

	ifstream vocabFile(vocab);

	string line;
	getline(vocabFile, line);
	int count = atoi(line.c_str());
	while (getline(vocabFile, line))
	{
		vector < string > words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(buf);
		}
		stringToIntvocabMap->insert(pair<string, int> (words[0], atoi(words[1].c_str())));
		intToStringvocabMap->insert(pair<int, string> (atoi(words[1].c_str()), words[0]));
	}
	vocabFile.close();
}

void loadVocabFromFile(char* vocab, map<string, int> * stringToIntvocabMap, map<int, string> * intToStringvocabMap,
		map<int, int> * wordCount)
{

	ifstream vocabFile(vocab);

	string line;
	getline(vocabFile, line);
	int count = atoi(line.c_str());
	while (getline(vocabFile, line))
	{
		vector < string > words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(buf);
		}
		stringToIntvocabMap->insert(pair<string, int> (words[0], atoi(words[1].c_str())));
		intToStringvocabMap->insert(pair<int, string> (atoi(words[1].c_str()), words[0]));
		wordCount->insert(pair<int, int> (atoi(words[1].c_str()), atoi(words[2].c_str())));
	}
	vocabFile.close();
}

void renormalize(double * dist, int n)
{

	double sum = 0;

	for (int i = 0; i < n; i++)
	{
		sum += dist[i];
	}

	for (int i = 0; i < n; i++)
	{
		dist[i] /= sum;
	}

}

