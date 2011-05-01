/*======================================================
 
 Unsupervised Hidden Markov Model
 
 Implemented here is a variable _order unsupervised Hidden
 Markov Model intended for use in language modeling.  It utilizes MPI
 to distribute the EM task in parallel.
 
 Arun Ahuja, Northwestern University

 Last Modified: December 25, 2009
 
 ========================================================*/
#include "hmm.h"
#include<string>
#include "math.h"
#include <time.h>
#include<stdlib.h>
#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include "utilities.h"
#include<math.h>
#include "mpi.h"
#include "distribution.h"
#include "basicProbability.h"

/* HMM Constructor; create HMM object; takes
 INT r, _order
 INT n_states, number of states
 INT n_obs, vocabulary size */

HMM::HMM(int order_, int numStates_, int numObs_)
{
	_order = order_;
	_numStates = numStates_ + 1; //autoinclude begin/end state
	_numObs = numObs_;
	_maxState = int(pow(_numStates, _order));

	/*Initial HMM parameter matrices */
	_pTransition = createMatrix(_maxState, _numStates);
	_pObservation = createMatrix(_numStates, _numObs);
	condstate_dist = createMatrix(_numObs, _numStates);
	//_initial_probability = new double[_maxState];
	_pState = new double[_numStates];
	srand(time(0));

	int sum;
	for (int i = 0; i < _numObs; i++)
	{
		sum = 0;
		//_pTransition[i][0] = 0;
		for (int j = 0; j < _numStates; j++)
		{
			int r = (rand() % 10000) + 1;
			_pTransition[i][j] = r;
			sum += r;
		}
		for (int j = 0; j < _numStates; j++)
		{
			_pTransition[i][j] /= sum;
		}

		if (!checkDistribution(_pTransition[i], _numStates))
		{
			cout << "Degenerate" << endl;
			exit(0);
		}

	}

	for (int o = 0; o < _numObs; o++)
	{
		_pObservation[0][o] = 0.0;
	}

	for (int i = 1; i < _numStates; i++)
	{
		sum = 0;
		for (int o = 0; o < _numObs; o++)
		{
			int r = (rand() % 10000) + 1;
			_pObservation[i][o] = r;
			sum += r;
		}
		for (int o = 0; o < _numObs; o++)
		{
			_pObservation[i][o] /= sum;
		}
		if (!checkDistribution(_pObservation[i], _numObs))
		{
			cout << "Degenerate" << endl;
			exit(0);
		}
	}

	//	//initial_probability[0] = 0.0;
	//	sum = 0;
	//	for (int i = 1; i < _numStates; i++)
	//	{
	//		int r = (rand() % 10000) + 1;
	//		initial_probability[i] = r;
	//		sum += r;
	//	}
	//	for (int i = 1; i < _numStates; i++)
	//	{
	//		initial_probability[i] /= sum;
	//	}
	//
	//	for (int i = _numStates; i < _maxState; i++)
	//	{
	//		initial_probability[i] = 0.0;
	//	}
}
/* HMM Constructor; create HMM object; takes
 STRING HMM File name
 */
HMM::HMM(char * hmm)
{
	ifstream HMMfile(hmm);
	string line;
	getline(HMMfile, line);
	stringstream ss(line);
	string buf;
	int params[3];
	int i = 0;
	while (ss >> buf)
	{
		params[i] = atoi(buf.c_str());
		i++;
	}
	_order = params[0];
	_numStates = params[1];
	_numObs = params[2];
	_maxState = int(pow(_numStates, _order));

	_pTransition = createMatrix(_maxState, _numStates);
	_pObservation = createMatrix(_numStates, _numObs);
	initial_probability = new double[_maxState];

	condstate_dist = createMatrix(_numObs, _numStates);
	_pState = new double[_numStates];

	for (int state = 0; state < _numStates; state++)
	{
		for (int obs = 0; obs < _numObs; obs++)
		{
			HMMfile >> _pObservation[state][obs];
		}
	}
	for (int i = 0; i < _numObs; i++)
	{
		for (int j = 0; j < _numStates; j++)
		{
			HMMfile >> condstate_dist[i][j];
		}
	}
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		for (int state = 0; state < _numStates; state++)
		{
			HMMfile >> _pTransition[state_sequence][state];
		}
	}

	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		HMMfile >> initial_probability[state_sequence];

	}

	HMMfile.close();
}

int HMM::sampleStartState()
{
	Distribution d(_pTransition[0], _maxState);
	return d.generate_sample();
}
int HMM::sampleState(int lastState)
{
	Distribution d(_pTransition[lastState], _numStates);
	return d.generate_sample();
}
int HMM::sampleWord(int state)
{
	Distribution d(_pObservation[state], _numObs);
	return d.generate_sample();
}

double**
HMM::getEmissionMatrix()
{
	return _pObservation;
}

double**
HMM::getTransitionMatrix()
{
	return _pTransition;
}

int HMM::getNumObs()
{
	return _numObs;
}

int HMM::getNumStates()
{
	return _numStates;
}

HMM::HMM(HMM &h)
{

	//copy constructor
}

/* HMM deconstructor */
HMM::~HMM()
{

	//delete[] initial_probability;
	freeMatrix(_pTransition, _maxState, _numStates);
	freeMatrix(_pObservation, _numStates, _numObs);
	delete[] _pState;
	freeMatrix(condstate_dist, _numObs, _numStates);

}
//void HMM::getSimilarWords(int word, vector<pair<int, double> > * similarWords) {
//
//	for (int i = 0; i < _numObs; i++) {
//		similarWords->push_back(pair<int, double> (i, klDivergence(
//				condstate_dist[word], condstate_dist[i], _numStates)));
//	}
//	sort(similarWords->begin(), similarWords->end(), sort_scorepairLESSTHAN);
//
//}

/*==============STANDARD FORWARD AND BACKWARD ALGORITHMS===================*/
void HMM::computeForwardMatrix(vector<int> words, double** forward, int len)
{

	forward[0][0] = 0;
	for (int state = 1; state < _numStates; state++)
	{
		forward[0][state] = initial_probability[state] * _pObservation[state][words[0]];
	}

	for (int state = _numStates; state < _maxState; state++)
	{
		forward[0][state] = 0;
	}

	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			int state = state_sequence % _numStates;
			double sum = 0;
			double o_prob = _pObservation[state][obs];
			for (int previous_state = 0; previous_state < _numStates; previous_state++)
			{
				int conditional_state = previous_state * int(pow(_numStates, _order - 1)) + state_sequence / _numStates;
				double t = _pTransition[conditional_state][state];
				sum += forward[i - 1][conditional_state] * _pTransition[conditional_state][state];
			}
			forward[i][state_sequence] = sum * o_prob;
		}
	}
}

void HMM::computeBackwardMatrix(vector<int> words, double ** backward, int len)
{

	for (int state = 0; state < _maxState; state++)
	{
		backward[len - 1][state] = 1.0;
	}

	for (int i = len - 2; i >= 0; i--)
	{
		int obs = words[i + 1];
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			double sum = 0;
			for (int next_state = 0; next_state < _numStates; next_state++)
			{
				int next_sequence = (state_sequence % int(pow(_numStates, _order - 1))) + next_state;
				sum += backward[i + 1][next_sequence] * _pTransition[state_sequence][next_state]
						* _pObservation[next_state][obs];
			}
			backward[i][state_sequence] = sum;
		}
	}
}
/*======================================================================+*/

/*==============SCALED FORWARD AND BACKWARD ALGORITHMS===================*/
void HMM::computeForwardMatrixScaled(vector<int> words, double** forward, double * scaleArray, int len)
{

	double scale = 0;
	forward[0][0] = 0;
	for (int state = 1; state < _numStates; state++)
	{
		double p = initial_probability[state] * _pObservation[state][words[0]];
		forward[0][state] = p;
		scale += p;
	}

	for (int state = 1; state < _numStates; state++)
	{
		forward[0][state] /= scale;
	}
	scaleArray[0] = scale;

	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		scale = 0;
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			int state = state_sequence % _numStates;
			double sum = 0;
			double o_prob = _pObservation[state][obs];
			for (int previous_state = 0; previous_state < _numStates; previous_state++)
			{
				int conditional_state = previous_state * int(pow(_numStates, _order - 1)) + state_sequence / _numStates;
				sum += forward[i - 1][conditional_state] * _pTransition[conditional_state][state];
			}
			double p = sum * _pObservation[state][words[i]];
			forward[i][state_sequence] = p;
			scale += p;
		}
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			forward[i][state_sequence] /= scale;
		}
		scaleArray[i] = scale;
	}
}

void HMM::computeSubForwardMatrixScaled(vector<int> words, double** forward, double * scaleArray, int len, int k)
{

	double scale = 0;
	forward[0][0] = 0;
	for (int state = 1; state < _numStates; state++)
	{
		double o_prob;
		if (k == 0)
			o_prob = 1;
		else
			o_prob = _pObservation[state][words[0]];

		double p = (1.0 / _numStates) * o_prob;
		forward[0][state] = p;
		scale += p;
	}

	for (int state = 1; state < _numStates; state++)
	{
		forward[0][state] /= scale;
	}
	scaleArray[0] = scale;

	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		scale = 0;
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			int state = state_sequence % _numStates;
			double sum = 0;
			double o_prob = _pObservation[state][obs];
			if (i == k)
			{
				o_prob = 1;

			}
			for (int previous_state = 0; previous_state < _numStates; previous_state++)
			{
				int conditional_state = previous_state * int(pow(_numStates, _order - 1)) + state_sequence / _numStates;
				sum += forward[i - 1][conditional_state] * _pTransition[conditional_state][state];
			}
			double p = sum * o_prob;
			forward[i][state_sequence] = p;
			scale += p;
		}
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			forward[i][state_sequence] /= scale;
		}
		scaleArray[i] = scale;
	}
}

void HMM::computeBackwardMatrixScaled(vector<int> words, double ** backward, double * scaleArray, int len)
{

	for (int state = 0; state < _maxState; state++)
	{
		backward[len - 1][state] = 1.0;
	}
	for (int i = len - 2; i >= 0; i--)
	{
		int obs = words[i + 1];
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			double sum = 0;
			for (int next_state = 0; next_state < _numStates; next_state++)
			{
				int next_sequence = _numStates * (state_sequence % int(pow(_numStates, _order - 1))) + next_state;
				sum += backward[i + 1][next_sequence] * _pTransition[state_sequence][next_state]
						* _pObservation[next_state][obs];
			}
			backward[i][state_sequence] = sum / scaleArray[i + 1];
		}
	}
}

double HMM::forwardAlgorithmScaled(vector<int> words)
{

	int len = words.size();
	double ** forward = createMatrix(len, _maxState);
	double * scaleArray = new double[len];
	computeForwardMatrixScaled(words, forward, scaleArray, len);
	double p = 0;
	for (int i = 0; i < len; i++)
	{
		p += log2(scaleArray[i]);
	}

	freeMatrix(forward, len, _maxState);
	delete[] scaleArray;
	return p;
}

double HMM::subForwardAlgorithmScaled(vector<int> words, int k)
{

	int len = words.size();
	double ** forward = createMatrix(len, _maxState);
	double * scaleArray = new double[len];
	computeSubForwardMatrixScaled(words, forward, scaleArray, len, k);
	double p = 0;
	for (int i = 0; i < len; i++)
	{
		p += log2(scaleArray[i]);
	}

	freeMatrix(forward, len, _maxState);
	delete[] scaleArray;
	return p;
}

/*=======================================================================*/

double HMM::forwardAlgorithm(vector<int> words)
{

	int len = words.size();
	double ** forward = createMatrix(len, _maxState);
	computeForwardMatrix(words, forward, len);
	double p = 0;
	for (int i = 0; i < _maxState; i++)
	{
		p += forward[len - 1][i];
	}

	freeMatrix(forward, len, _maxState);

	return p;
}

double HMM::backwardAlgorithm(vector<int> words)
{
	int len = words.size();
	double ** backward = createMatrix(len, _maxState);
	computeBackwardMatrix(words, backward, len);

	double p = 0;
	for (int i = 0; i < _maxState; i++)
	{
		p += backward[0][i] * _pObservation[i][words[0]] * initial_probability[i];
	}

	freeMatrix(backward, len, _maxState);

	return p;
}

//Forward Algorithm for computation of sequence probability
double HMM::forward(int* words)
{

	int len = sizeof(words) / sizeof(int);
	double *a;
	a = new double[_maxState];

	for (int state = 0; state < _maxState; state++)
	{
		a[state] = initial_probability[state] * _pObservation[state][words[0]];
	}

	double *current;
	current = new double[_maxState];
	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			int state = state_sequence % _numStates;
			double sum = 0;
			double o_prob = _pObservation[state][obs];
			for (int previous_state = 0; previous_state < _numStates; previous_state++)
			{
				int conditional_state = previous_state * int(pow(_numStates, _order - 1)) + state_sequence / _numStates;
				double trans = _pTransition[conditional_state][state];
				sum += a[conditional_state] * trans;
			}
			current[state_sequence] = sum * o_prob;
		}
		a = current;
	}
	double sum = 0;
	for (int i = 0; i < _maxState; i++)
	{
		sum += a[i];
	}
	delete[] a;
	delete[] current;
	return sum;
}

//Backward Algorithm
double HMM::backward(int* words)
{

	int len = sizeof(words) / sizeof(int);
	;

	double *b;
	b = new double[_maxState];

	for (int state = 0; state < _maxState; state++)
	{
		b[state] = 1.0;
	}
	double *current;
	current = new double[_maxState];
	for (int i = len - 2; i >= 0; i--)
	{
		int obs = words[i + 1];

		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			int state = state_sequence % _order;
			double sum = 0;
			for (int next_state = 0; next_state < _numStates; next_state++)
			{
				double o_prob = _pObservation[next_state][obs];
				int next_sequence = _numStates * (state_sequence % int(pow(_numStates, _order - 1))) + next_state;
				//int conditional_state = previous_state*msb + state_sequence/_order;
				double trans = _pTransition[state_sequence][next_state];

				sum += b[next_sequence] * trans * o_prob;
			}
			current[state_sequence] = sum;
		}
		b = current;
	}
	double sum = 0;
	for (int i = 0; i < _maxState; i++)
	{
		sum += b[i] * _pObservation[i][words[0]] * initial_probability[i];
	}
	delete[] b;
	delete[] current;
	return sum;

}

/*NOT IMPLEMENTED*/

/*int* HMM::viterbi(int* words) {
 
 int *opt_state;

 int len = sizeof(words)/sizeof(int);;

 opt_state = new int[len];


 double *a;
 a = new double[_maxState];

 for (int state = 0; state < _numStates; state++) {
 a[state] = initial_probability[state]*_pObservation[state][words[0]];
 }

 double *current;
 current = new double[_maxState];
 for (int i = 1; i < len; i++){
 int obs = words[i];
 for (int state_sequence = 0; state_sequence < _maxState; state_sequence++){
 int state = state_sequence % _numStates;
 double sum = 0;
 double o_prob = _pObservation[state][obs];
 for (int previous_state = 0; previous_state < _numStates; previous_state++) {
 int conditional_state = previous_state*int(pow(_numStates, _order-1)) + state_sequence/_order;
 double trans = _pTransition[conditional_state][state];
 sum += a[conditional_state] * trans;
 }
 current[state_sequence] = sum*o_prob;
 }
 a = current;
 }
 double max = 0;
 double sum = 0;
 for (int i =0; i<_maxState; i++){
 sum += a[i];
 }
 delete [] a;
 delete [] current;

 return opt_state;
 }

 */

/* HMM EM Training; 
 STRING inputFile
 inputFile is a training file
 
 */

void HMM::trainFromFile(char* inputFile)
{

	double **emit_count;
	double **trans_count;
	double *state_count;
	double *sequence_count;
	double *init_count;
	double *obs_count;
	emit_count = createMatrix(_numStates, _numObs);
	trans_count = createMatrix(_maxState, _numStates);
	state_count = new double[_numStates];
	zeroArray(state_count, _numStates);
	obs_count = new double[_numObs];
	zeroArray(obs_count, _numObs);
	sequence_count = new double[_maxState];
	zeroArray(sequence_count, _maxState);
	init_count = new double[_maxState];
	zeroArray(init_count, _maxState);

	ifstream trainFile(inputFile);
	string line;
	int count = 0;
	while (getline(trainFile, line))
	{ // for every sentence
		vector<int> words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(atoi(buf.c_str()));
		}
		int len = words.size();
		count++;
		//COMPUTE FORWARD PROBABILITY
		double **forward;
		double * scaleArray = new double[len];
		forward = createMatrix(len, _maxState);
		//double forward_prob = forwardAlgorithmScaled(words);
		//computeForwardMatrix(words, forward,  len);
		//printMatrix(forward, len, _maxState);
		computeForwardMatrixScaled(words, forward, scaleArray, len);
		//printMatrix(forward, len, _maxState);

		//COMPUTE_BACKWARD PROBABILITY
		double **backward;
		backward = createMatrix(len, _maxState);
		//computeBackwardMatrix(words, backward,  len);
		//printMatrix(backward, len, _maxState);
		computeBackwardMatrixScaled(words, backward, scaleArray, len);
		//printMatrix(backward, len, _maxState);

		//BAUM WELCH COUNTS
		for (int i = 0; i < len - 1; i++)
		{
			int obs = words[i];
			int next_obs = words[i + 1];
			for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
			{
				int state = state_sequence % _numStates;
				//cout << "for state" << state<<" "<<forward[i][state_sequence]<<"*"<<backward[i][state_sequence]<<endl;
				double gamma = (forward[i][state_sequence] * backward[i][state_sequence]);
				//				if (gamma != gamma)
				//				{
				//					cout << "gammma problem" << endl;
				//					printMatrix(forward, len, _maxState);
				//					printMatrix(backward, len, _maxState);
				//					printArray(scaleArray, len);
				//
				//					return;
				//				}
				if (i == 0)
				{
					_pTransition[0][state_sequence] += gamma;
				}
				emit_count[state][obs] += gamma;
				obs_count[obs] += gamma;
				state_count[state] += gamma;
				sequence_count[state_sequence] += gamma;
				for (int next_state = 0; next_state < _numStates; next_state++)
				{
					int next_sequence = _numStates * (state_sequence % int(pow(_numStates, _order - 1))) + next_state;
					double eta = (forward[i][state_sequence] * _pTransition[state_sequence][next_state]
							* _pObservation[next_state][next_obs] * backward[i + 1][next_sequence]) / scaleArray[i + 1];
					trans_count[state_sequence][next_state] += eta;
				}
			}
		}
		for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
		{
			int obs = words[len - 1];
			int state = state_sequence % _numStates;
			double gamma = (forward[len - 1][state_sequence] * backward[len - 1][state_sequence]);
			emit_count[state][obs] += gamma;
			obs_count[obs] += gamma;
			state_count[state] += gamma;
		}
		delete[] scaleArray;
		freeMatrix(forward, len, _maxState);
		freeMatrix(backward, len, _maxState);

	}//end for every sentence
	trainFile.close();
	updateHMM(emit_count, trans_count, state_count, sequence_count, init_count, obs_count);

	freeMatrix(trans_count, _maxState, _numStates);
	freeMatrix(emit_count, _numStates, _numObs);
	delete[] state_count;
	delete[] obs_count;
	delete[] sequence_count;

}

/* HMM EM Training; 
 VECTOR<STRING> files
 inputFile is a file containing a list of training files
 
 */

void HMM::trainParallel(vector<string> filesList_)
{

	int count = filesList_.size();
	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();
	const int root = 0;
	int dist = count / size;
	int start = rank * dist;
	int end = rank * dist + dist;

	double **emit_count;
	double **trans_count;
	double *state_count;
	double *sequence_count;
	double *init_count;
	double *obs_count;
	emit_count = createMatrix(_numStates, _numObs);
	trans_count = createMatrix(_maxState, _numStates);
	state_count = new double[_numStates];
	zeroArray(state_count, _numStates);
	obs_count = new double[_numObs];
	zeroArray(obs_count, _numObs);
	sequence_count = new double[_maxState];
	zeroArray(sequence_count, _maxState);
	init_count = new double[_maxState];
	zeroArray(init_count, _maxState);

	double **temit_count;
	double **ttrans_count;
	double *tstate_count;
	double *tsequence_count;
	double *tinit_count;
	double *tobs_count;

	temit_count = createMatrix(_numStates, _numObs);
	ttrans_count = createMatrix(_maxState, _numStates);
	tstate_count = new double[_numStates];
	tobs_count = new double[_numObs];
	zeroArray(tstate_count, _numStates);
	tsequence_count = new double[_maxState];
	zeroArray(tsequence_count, _maxState);
	tinit_count = new double[_maxState];
	zeroArray(tinit_count, _maxState);

	for (int i = start; i < end; i++)
	{
		const char* inputFile = filesList_[i].c_str();
		//		cout << "opening file "<<files_list[i].c_str()<< " On Process " << rank<<endl;
		ifstream trainFile(inputFile);
		string line;
		while (getline(trainFile, line))
		{ // for every sentence

			// Read in training sequence
			vector<int> words;
			stringstream ss(line);
			string buf;
			while (ss >> buf)
			{
				words.push_back(atoi(buf.c_str()));
			}
			int len = words.size();

			//COMPUTE FORWARD PROBABILITY
			double **forward;
			double * scaleArray = new double[len];
			forward = createMatrix(len, _maxState);
			computeForwardMatrixScaled(words, forward, scaleArray, len);
			//printMatrix(forward, len, _maxState);

			//COMPUTE_BACKWARD PROBABILITY
			double **backward;
			backward = createMatrix(len, _maxState);
			computeBackwardMatrixScaled(words, backward, scaleArray, len);
			//printMatrix(backward, len, _maxState);

			//BAUM WELCH COUNTS
			for (int i = 0; i < len - 1; i++)
			{
				int obs = words[i];
				int next_obs = words[i + 1];
				for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
				{
					int state = state_sequence % _numStates;
					double gamma = (forward[i][state_sequence] * backward[i][state_sequence]);
					if (i == 0)
					{
						init_count[state_sequence] += gamma;
					}
					emit_count[state][obs] += gamma;
					obs_count[obs] += gamma;
					state_count[state] += gamma;
					sequence_count[state_sequence] += gamma;
					for (int next_state = 0; next_state < _numStates; next_state++)
					{
						int next_sequence = _numStates * (state_sequence % int(pow(_numStates, _order - 1)))
								+ next_state;
						double eta = (forward[i][state_sequence] * _pTransition[state_sequence][next_state]
								* _pObservation[next_state][next_obs] * backward[i + 1][next_sequence]) / scaleArray[i
								+ 1];
						trans_count[state_sequence][next_state] += eta;
					}
				}
			}
			for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
			{
				int obs = words[len - 1];
				int state = state_sequence % _numStates;
				double gamma = (forward[len - 1][state_sequence] * backward[len - 1][state_sequence]);
				emit_count[state][obs] += gamma;
				obs_count[obs] += gamma;
				state_count[state] += gamma;
				//	sequence_count[state_sequence] += gamma;
			}
			delete[] scaleArray;
			freeMatrix(forward, len, _maxState);
			freeMatrix(backward, len, _maxState);
		}//end for every sentence

		trainFile.close();
		//	cout << " Training File Close, Updating Parameters " << inputFile << endl;
	} // for every file


	//Collect parameters on root
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		MPI_Reduce(trans_count[state_sequence], ttrans_count[state_sequence], _numStates, MPI_DOUBLE, MPI_SUM, root,
				MPI_COMM_WORLD);
	}
	MPI_Reduce(sequence_count, tsequence_count, _maxState, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	MPI_Reduce(init_count, tinit_count, _maxState, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	for (int state = 0; state < _numStates; state++)
	{
		MPI_Reduce(emit_count[state], temit_count[state], _numObs, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	}
	MPI_Reduce(state_count, tstate_count, _numStates, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	MPI_Reduce(obs_count, tobs_count, _numObs, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	//Send updated parameters too all children
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		MPI_Bcast(ttrans_count[state_sequence], _numStates, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}
	MPI_Bcast(tsequence_count, _maxState, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tinit_count, _maxState, MPI_DOUBLE, root, MPI_COMM_WORLD);
	for (int state = 0; state < _numStates; state++)
	{
		MPI_Bcast(temit_count[state], _numObs, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}
	MPI_Bcast(tstate_count, _numStates, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tobs_count, _numObs, MPI_DOUBLE, root, MPI_COMM_WORLD);

	//cout << "Update Step" << endl;
	updateHMM(temit_count, ttrans_count, tstate_count, tsequence_count, tinit_count, tobs_count);

	freeMatrix(trans_count, _maxState, _numStates);
	freeMatrix(emit_count, _numStates, _numObs);
	delete[] state_count;
	delete[] sequence_count;
	delete[] init_count;
	delete[] obs_count;

	freeMatrix(temit_count, _numStates, _numObs);
	freeMatrix(ttrans_count, _maxState, _numStates);
	delete[] tstate_count;
	delete[] tsequence_count;
	delete[] tinit_count;
	delete[] tobs_count;

}

void HMM::updateHMM(double** emit_count, double** trans_count, double* state_count, double* sequence_count,
		double* initial_count, double* obs_count)
{

	double state_total = 0;
	for (int state = 1; state < _numStates; state++)
	{
		double smooth = 1 / ((_numStates - 1) * _numObs);
		for (int obs = 0; obs < _numObs; obs++)
		{
			_pObservation[state][obs] = ((emit_count[state][obs]) + smooth) / (state_count[state] + smooth * _numObs);
			condstate_dist[obs][state] = (emit_count[state][obs] + smooth) / (obs_count[obs] + smooth
					* (_numStates - 1));
			state_total += state_count[state];

		}
	}
	for (int state = 0; state < _numStates; state++)
	{
		_pState[state] = state_count[state] / state_total;
	}
	double totalInit = 0;
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		if (sequence_count[state_sequence] != 0)
		{
			double smooth = 1 / (_maxState * _numStates - 1);
			_pTransition[state_sequence][0] = 0;
			for (int state = 1; state < _numStates; state++)
			{
				_pTransition[state_sequence][state] = (trans_count[state_sequence][state] + smooth)
						/ (sequence_count[state_sequence] + smooth * (_numStates - 1));
			}
		}
		else
		{
			for (int state = 0; state < _numStates; state++)
			{
				_pTransition[state_sequence][state] = 0;
			}
		}
		initial_probability[state_sequence] = initial_count[state_sequence];
		totalInit += initial_count[state_sequence];
	}

	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		initial_probability[state_sequence] /= totalInit;
	}

}

void HMM::trainGibbsFromFile(char* inputFile)
{

	double **emit_count;
	double **trans_count;
	double *state_count;
	double *sequence_count;
	double *init_count;
	double *obs_count;

	emit_count = createMatrix(_numStates, _numObs);
	trans_count = createMatrix(_maxState, _numStates);
	state_count = new double[_numStates];
	zeroArray(state_count, _numStates);
	obs_count = new double[_numObs];
	zeroArray(obs_count, _numObs);
	sequence_count = new double[_maxState];
	zeroArray(sequence_count, _maxState);
	init_count = new double[_maxState];
	zeroArray(init_count, _maxState);

	ifstream trainFile(inputFile);
	string line;
	int count = 0;
	while (getline(trainFile, line))
	{ // for every sentence
		vector<int> words;
		stringstream ss(line);
		string buf;

		while (ss >> buf)
		{
			words.push_back(atoi(buf.c_str()));
		}
		int len = words.size();
		count++;
		int *stateArray;
		stateArray = new int[len];
		for (int i = 0; i < len; i++)
		{
			int origState = (rand() % (_numStates - 1)) + 1;
			int obs = words[i];
			int prev_sequence = 0;
			int r = 0;
			stateArray[i] = origState;
			while ((r < _order) && (i - 1 - r) >= 0)
			{
				prev_sequence += stateArray[(i - 1) - r] * int(pow(_numStates, r));
				r++;
			}
			obs_count[obs]++;
			state_count[origState]++;
			if (i == 0)
				init_count[origState]++;
			else
			{
				trans_count[prev_sequence][origState]++;
				sequence_count[prev_sequence]++;
			}
			emit_count[origState][obs]++;

		}
		int sampleN = rand() % len;
		for (int i = 0; i < sampleN; i++)
		{
			int k = rand() % (len - 1);
			int obs = words[k];
			int prev_sequence = 0;
			int r = 0;
			//	cout << "Compute prev_seq" << endl;
			while ((r < _order) && (k - 1 - r) >= 0)
			{
				prev_sequence += stateArray[(k - 1) - r] * int(pow(_numStates, r));
				r++;
			}
			//	cout << "Done Compute prev_seq" << endl;
			int origState = stateArray[k];
			int nextState = stateArray[k + 1];
			int next_sequence = _numStates * (prev_sequence % int(pow(_numStates, _order - 1))) + origState;
			double *dist = new double[_numStates];
			double totalp = 0;
			for (int state = 0; state < _numStates; state++)
			{
				int state_sequence = _numStates * (prev_sequence % int(pow(_numStates, _order - 1))) + state;
				if (prev_sequence == 0)
					dist[state] = _pObservation[state][obs] * initial_probability[state]
							* _pTransition[state_sequence][nextState];
				else
					dist[state] = _pObservation[state][obs] * _pTransition[prev_sequence][state]
							* _pTransition[state_sequence][nextState];
				totalp += dist[state];
			}
			renormalize(dist, _numStates);
			Distribution d(dist, _numStates);
			int sample = d.generate_sample();
			delete[] dist;

			//	cout << "Update params" << endl;
			state_count[origState]--;
			if (k == 0)
			{
				init_count[origState]--;
				init_count[sample]++;
			}
			else
			{
				trans_count[prev_sequence][origState]--;
				trans_count[prev_sequence][sample]++;
			}
			trans_count[next_sequence][nextState]--;
			sequence_count[next_sequence]--;
			emit_count[origState][obs]--;
			stateArray[k] = sample;
			next_sequence = _numStates * (prev_sequence % int(pow(_numStates, _order - 1))) + sample;
			state_count[sample]++;
			trans_count[next_sequence][nextState]++;
			sequence_count[next_sequence]++;
			emit_count[sample][obs]++;

			//	cout << "Done Update params" << endl;

		}
	}//end for every sentence
	trainFile.close();
	updateHMM(emit_count, trans_count, state_count, sequence_count, init_count, obs_count);

	freeMatrix(trans_count, _maxState, _numStates);
	freeMatrix(emit_count, _numStates, _numObs);
	delete[] state_count;
	delete[] obs_count;
	delete[] sequence_count;

}

void HMM::trainGibbsParallel(vector<string> files_list)
{

	int count = files_list.size();
	int rank = MPI::COMM_WORLD.Get_rank();
	int size = MPI::COMM_WORLD.Get_size();
	const int root = 0;
	int dist = count / size;
	int start = rank * dist;
	int end = rank * dist + dist;

	double **emit_count;
	double **trans_count;
	double *state_count;
	double *sequence_count;
	double *init_count;
	double *obs_count;
	emit_count = createMatrix(_numStates, _numObs);
	trans_count = createMatrix(_maxState, _numStates);

	state_count = new double[_numStates];
	zeroArray(state_count, _numStates);

	obs_count = new double[_numObs];
	zeroArray(obs_count, _numObs);

	sequence_count = new double[_maxState];
	zeroArray(sequence_count, _maxState);

	init_count = new double[_maxState];
	zeroArray(init_count, _maxState);

	double **temit_count;
	double **ttrans_count;
	double *tstate_count;
	double *tsequence_count;
	double *tinit_count;
	double *tobs_count;

	temit_count = createMatrix(_numStates, _numObs);
	ttrans_count = createMatrix(_maxState, _numStates);

	tstate_count = new double[_numStates];
	zeroArray(tstate_count, _numStates);

	tobs_count = new double[_numObs];
	zeroArray(tobs_count, _numObs);

	tsequence_count = new double[_maxState];
	zeroArray(tsequence_count, _maxState);

	tinit_count = new double[_maxState];
	zeroArray(tinit_count, _maxState);

	for (int i = start; i < end; i++)
	{
		const char* inputFile = files_list[i].c_str();
		ifstream trainFile(inputFile);
		string line;
		while (getline(trainFile, line))
		{ // for every sentence

			// Read in training sequence
			vector<int> words;
			stringstream ss(line);
			string buf;

			while (ss >> buf)
			{
				words.push_back(atoi(buf.c_str()));
			}
			int len = words.size();
			count++;
			int *stateArray;
			stateArray = new int[len];
			for (int i = 0; i < len; i++)
			{
				int origState = (rand() % (_numStates - 1)) + 1;
				int obs = words[i];
				int prev_sequence = 0;
				int r = 0;
				stateArray[i] = origState;

				if (i == 0)
					init_count[origState]++;
				else
				{
					while ((r < _order) && (i - 1 - r) >= 0)
					{
						prev_sequence += stateArray[(i - 1) - r] * int(pow(_numStates, r));
						r++;
					}
					trans_count[prev_sequence][origState]++;
					sequence_count[prev_sequence]++;
				}
				obs_count[obs]++;
				state_count[origState]++;
				emit_count[origState][obs]++;

			}
			int sampleN = rand() % len;
			for (int i = 0; i < sampleN; i++)
			{
				int k = rand() % (len - 1);
				int obs = words[k];
				int prev_sequence = 0;
				int r = 0;
				//		cout << "Compute seq " <<endl;
				while ((r < _order) && (k - 1 - r) >= 0)
				{
					prev_sequence += stateArray[(k - 1) - r] * int(pow(_numStates, r));
					r++;
				}
				//		cout << "Done Compute seq " <<endl;
				int origState = stateArray[k];
				int nextState = stateArray[k + 1];
				int next_sequence = _numStates * (prev_sequence % int(pow(_numStates, _order - 1))) + nextState;
				double *dist = new double[_numStates];
				double totalp = 0;
				for (int state = 0; state < _numStates; state++)
				{
					int state_sequence = _numStates * (prev_sequence % int(pow(_numStates, _order - 1))) + state;
					if (prev_sequence == 0)
						dist[state] = _pObservation[state][obs] * initial_probability[state]
								* _pTransition[state_sequence][nextState];
					else
						dist[state] = _pObservation[state][obs] * _pTransition[prev_sequence][state]
								* _pTransition[state_sequence][nextState];
					totalp += dist[state];
				}
				renormalize(dist, _numStates);
				Distribution d(dist, _numStates);
				int sample = d.generate_sample();
				delete[] dist;

				if (k == 0)
				{
					init_count[origState]--;
					init_count[sample]++;
				}
				else
				{
					trans_count[prev_sequence][origState]--;
					trans_count[prev_sequence][sample]++;
				}
				state_count[origState]--;
				//			trans_count[next_sequence][nextState]--;
				//			sequence_count[next_sequence]--;
				emit_count[origState][obs]--;
				stateArray[k] = sample;
				//			next_sequence = _numStates*(prev_sequence  % int(pow(_numStates, _order-1))) + sample;
				state_count[sample]++;
				//			trans_count[next_sequence][nextState]++;
				//			sequence_count[next_sequence]++;
				emit_count[sample][obs]++;
				//		cout << "Done Update parameters" << endl;
			}
		}//end for every sentence

		trainFile.close();

	} // for every file

	//Collect parameters on root
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		MPI_Reduce(trans_count[state_sequence], ttrans_count[state_sequence], _numStates, MPI_DOUBLE, MPI_SUM, root,
				MPI_COMM_WORLD);
	}
	MPI_Reduce(sequence_count, tsequence_count, _maxState, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	MPI_Reduce(init_count, tinit_count, _maxState, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	for (int state = 0; state < _numStates; state++)
	{
		MPI_Reduce(emit_count[state], temit_count[state], _numObs, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	}
	MPI_Reduce(state_count, tstate_count, _numStates, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	MPI_Reduce(obs_count, tobs_count, _numObs, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

	//Send updated parameters too all children
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		MPI_Bcast(ttrans_count[state_sequence], _numStates, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}
	MPI_Bcast(tsequence_count, _maxState, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tinit_count, _maxState, MPI_DOUBLE, root, MPI_COMM_WORLD);
	for (int state = 0; state < _numStates; state++)
	{
		MPI_Bcast(temit_count[state], _numObs, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}
	MPI_Bcast(tstate_count, _numStates, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tobs_count, _numObs, MPI_DOUBLE, root, MPI_COMM_WORLD);

	//cout << "Update Step" << endl;
	updateHMM(temit_count, ttrans_count, tstate_count, tsequence_count, tinit_count, tobs_count);

	freeMatrix(trans_count, _maxState, _numStates);
	freeMatrix(emit_count, _numStates, _numObs);
	delete[] state_count;
	delete[] sequence_count;
	delete[] init_count;
	delete[] obs_count;

	freeMatrix(temit_count, _numStates, _numObs);
	freeMatrix(ttrans_count, _maxState, _numStates);
	delete[] tstate_count;
	delete[] tsequence_count;
	delete[] tinit_count;
	delete[] tobs_count;
}

bool HMM::checkDistribution(double * dist, int n)
{
	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += dist[i];
	}
	if (sum >= .9998 && sum <= 1.002)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}

/* saveHMM - saves HMM to text file; takes
 outputFile, file to print to */
void HMM::saveHMM(char * outputFile)
{
	ofstream HMMfile(outputFile);
	HMMfile << _order << " " << _numStates << " " << _numObs << endl;
	for (int i = 0; i < _numStates; i++)
	{
		for (int j = 0; j < _numObs; j++)
		{
			HMMfile << _pObservation[i][j] << " ";
		}
		HMMfile << endl;
	}
	for (int i = 0; i < _numObs; i++)
	{
		for (int j = 0; j < _numStates; j++)
		{
			HMMfile << condstate_dist[i][j] << " ";
		}
		HMMfile << endl;
	}
	HMMfile << endl;
	for (int i = 0; i < _maxState; i++)
	{
		for (int j = 0; j < _numStates; j++)
		{
			HMMfile << _pTransition[i][j] << " ";
		}
		HMMfile << endl;
	}
	for (int i = 0; i < _maxState; i++)
	{
		HMMfile << initial_probability[i] << endl;
	}
	HMMfile << endl;
	for (int j = 0; j < _numStates; j++)
	{
		HMMfile << _pState[j] << endl;
	}
	HMMfile.close();
}

void HMM::checkTransition()
{
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		double max = 0;
		double entropy = 0;
		for (int state = 0; state < _numStates; state++)
		{
			double p = _pTransition[state_sequence][state];
			if (p > max)
				max = p;
			entropy -= p * log2(p);
		}

		cout << "For state_seqeunce " << state_sequence << " entropy: " << entropy << "  and max: " << max << endl;
	}
}
void HMM::checkObservation()
{
	for (int state = 0; state < _numStates; state++)
	{
		double max = 0;
		double entropy = 0;
		for (int obs = 0; obs < _numObs; obs++)
		{
			double p = _pObservation[state][obs];
			if (p > max)
				max = p;
			entropy -= p * log2(p);
		}

		cout << "For state  " << state << " entropy: " << entropy << "  and max: " << max << endl;
	}

}

void HMM::loadWithStateDist(char * hmm)
{
	ifstream HMMfile(hmm);
	string line;
	getline(HMMfile, line);
	stringstream ss(line);
	string buf;
	int params[3];
	int i = 0;
	while (ss >> buf)
	{
		params[i] = atoi(buf.c_str());
		i++;
	}
	_order = params[0];
	_numStates = params[1];
	_numObs = params[2];
	_maxState = int(pow(_numStates, _order));

	_pTransition = createMatrix(_maxState, _numStates);
	_pObservation = createMatrix(_numStates, _numObs);
	initial_probability = new double[_maxState];

	condstate_dist = createMatrix(_numObs, _numStates);
	_pState = new double[_numStates];

	for (int state = 0; state < _numStates; state++)
	{
		for (int obs = 0; obs < _numObs; obs++)
		{
			HMMfile >> _pObservation[state][obs];
		}
	}
	for (int i = 0; i < _numObs; i++)
	{
		for (int j = 0; j < _numStates; j++)
		{
			HMMfile >> condstate_dist[i][j];
		}
	}
	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
	{
		for (int state = 0; state < _numStates; state++)
		{
			HMMfile >> _pTransition[state_sequence][state];
		}
	}

//	for (int state_sequence = 0; state_sequence < _maxState; state_sequence++)
//	{
//		HMMfile >> initial_probability[state_sequence];
//
//	}

	for (int j = 0; j < _numStates; j++)
	{
		HMMfile >> _pState[j];
	}

	HMMfile.close();
}
/*
 void HMM::recomputeEmitDist()
 {
 double * word_dist = new double[_numObs];

 for (int w = 0; w < _numObs; w++)
 {
 word_dist[w] = 0;
 for (int i = 0; i <_numStates; i++)
 {
 word_dist[w] += _pObservation[i][w]*_pState[i];
 }
 }

 renormalize(word_dist, _numObs);
 for (int state =1; state < _numStates; state++)
 {
 for (int obs = 0; obs < _numObs; obs++)
 {
 _pObservation[state][obs] = condstate_dist[obs][state]*word_dist[obs];
 }
 renormalize(_pObservation[state], _numObs);
 }
 }*/

