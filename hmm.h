#ifndef HMM_H
#define HMM_H

#include<string>
#include<vector>
using namespace std;

class HMM
{
protected:
	int _order;
	int _numStates;
	int _maxState;
	int _numObs;
	double **_pObservation;
	double **_pTransition;
	double* _pState;
	double* initial_probability;
	virtual void updateHMM(double **trans, double **emit, double *state_count, double *sequence_count,
			double* init_count, double* obs_count);
	virtual void computeForwardMatrix(vector<int> words, double ** forward, int len);
	virtual void computeBackwardMatrix(vector<int> words, double ** backward, int len);
	virtual void computeForwardMatrixScaled(vector<int> words, double** forward, double * scaleArray, int len);
	virtual void
			computeSubForwardMatrixScaled(vector<int> words, double** forward, double * scaleArray, int len, int k);
	virtual void computeBackwardMatrixScaled(vector<int> words, double** backward, double * scaleArray, int len);

public:
	HMM(int r, int n_states, int n_obs);
	HMM(char * inputFile);
	HMM(HMM &h);
	~HMM();

	void splitModel();

	int getNumObs();
	int getNumStates();

	double ** getTransitionMatrix();
	double ** getEmissionMatrix();
	int sampleStartState();
	int sampleState(int lastState);
	int sampleWord(int state);

	double** condstate_dist;
	void getSimilarWords(int word, vector<pair<int, double> > * similarWords);
	double backwardAlgorithm(vector<int> words);
	double forwardAlgorithm(vector<int> words);
	double subForwardAlgorithmScaled(vector<int> words, int k);
	double forwardAlgorithmScaled(vector<int> words);
	double forward(int* s);
	double backward(int *s);
	void trainFromFile(char* fileName);
	void trainParallel(vector<string> files_list);
	void trainGibbsFromFile(char* fileName);
	void trainGibbsParallel(vector<string> files_list);
	int* viterbi(int *s);
	bool checkDistribution(double * distribution, int n);
	void saveHMM(char* outputName);
	void loadWithStateDist(char*);
	void checkTransition();
	void checkObservation();
	//void loadHMM( char* hmm);

};

#endif
