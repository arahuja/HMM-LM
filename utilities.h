#ifndef UTILITIES_H
#define UTILITIES_H

#include <map>
#include <string>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <fstream>
#include<iostream>

using namespace std;

typedef pair<int, double> ScorePair;

bool sort_scorepairLESSTHAN(const ScorePair& left, const ScorePair& right);
bool sort_scorepairGREATER(const ScorePair& left, const ScorePair& right);
double** createMatrix(int m, int n);
void createVocab(char * vocab, char * vocabOut);
void translateFile(char * vocab, char * input, char* output);
void loadVocabFromFile(char* vocab, map<string, int> * stringToIntvocabMap, map<int, string> * intToStringvocabMap);
void loadVocabFromFile(char* vocab, map<string, int> * stringToIntvocabMap, map<int, string> * intToStringvocabMap);
void renormalize(double * dist, int n);



template<class T>
int maxVal(T* arr, int n)
{
	int maxindex = 0;
	T maxval = arr[0];
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

template<class T>
void freeMatrix(T** matrix, int m, int n)
{

	for (int i = 0; i < m; i++)
	{
		delete[] matrix[i];
	}
	delete[] matrix;
}

template<class T>
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

template<class T>
void printArray(T* array, int n)
{
	for (int j = 0; j < n; j++)
	{
		cout << array[j] << " ";
	}
	cout << endl;
}

template<class T>
void zeroArray(T * a, int n)
{
	for (int i = 0; i < n; i++)
	{
		a[i] = 0;
	}
}

template<class T>
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

#endif
