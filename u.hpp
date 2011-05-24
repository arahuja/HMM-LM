#include <map>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <map>
#include <fstream>
#include<iostream>

using namespace std;

typedef pair<int, double> ScorePair;

template <class T>
int maxVal(T* arr, int n)
{
	int maxindex = 0;
	double maxval = arr[0];
	for (int i = 1; i < n; i++)
	{
		if (arr[i] > maxval)
		{
			maxval = arr[i];
			maxindex = i;
		}
	}

	return maxindex;
}

template <class T>
void freeMatrix(T** matrix, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		delete[] matrix[i];
	}
	delete[] matrix;
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

template <class T>
void printMatrix(T ** matrix, int m, int n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

template <class T>
void printArray(T* array, int n)
{
	for (int j = 0; j < n; j++)
	{
		cout << array[j] << " ";
	}
	cout << endl;
}

template <class T>
void zeroArray(T * a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = 0;
	}
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
		outputFile << iter->first << " " << iter->second << " "
				<< wordCount[iter->first] << endl;
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

void loadVocabFromFile(char* vocab, map<string, int> * stringToIntvocabMap,
		map<int, string> * intToStringvocabMap)
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
		stringToIntvocabMap->insert(pair<string, int> (words[0], atoi(
				words[1].c_str())));
		intToStringvocabMap->insert(pair<int, string> (atoi(words[1].c_str()),
				words[0]));
	}
	vocabFile.close();
}

void loadVocabFromFile(char* vocab, map<string, int> * stringToIntvocabMap,
		map<int, string> * intToStringvocabMap, map<int, int> * wordCount)
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
		stringToIntvocabMap->insert(pair<string, int> (words[0], atoi(
				words[1].c_str())));
		intToStringvocabMap->insert(pair<int, string> (atoi(words[1].c_str()),
				words[0]));
		wordCount->insert(pair<int, int> (atoi(words[1].c_str()), atoi(
				words[2].c_str())));
	}
	vocabFile.close();
}
bool sort_scorepairGREATER(const ScorePair& left, const ScorePair& right)
{
	return left.second > right.second;
}

bool sort_scorepairLESSTHAN(const ScorePair& left, const ScorePair& right)
{
	return left.second < right.second;
}

template <class T>
int argmax(T * arr, int n)
{
	if (n == 0)
	{
		return 0;
	}
	T max = arr[0];
	int maxIndex = 0;
	for (int i = 1; i < n; i++)
	{
		if (arr[i] > max)
		{
			max = arr[i];
			maxIndex = i;
		}
	}

	return maxIndex;
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
