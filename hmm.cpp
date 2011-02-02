/*======================================================
 
 Unsupervised Hidden Markov Model
 
 Implemented here is a variable order unsupervised Hidden
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
 INT r, order
 INT n_states, number of states
 INT n_obs, vocabulary size */

HMM::HMM(int r, int n_states, int n_obs)
{
	order = r;
	num_states = n_states + 1;
	num_obs = n_obs;
	max_state = int(pow(num_states, order));

	/*Initial HMM parameter matrices */
	transition = createMatrix(max_state, num_states);
	observation = createMatrix(num_states, num_obs);
	condstate_dist = createMatrix(num_obs, num_states);
	initial_probability = new double[max_state];
	state_dist = new double[num_states];
	srand(time(0));

	int sum;
	for (int i = 0; i < max_state; i++)
	{
		sum = 0;
		transition[i][0] = 0;
		for (int j = 1; j < num_states; j++)
		{
			int r = (rand() % 10000) + 1;
			transition[i][j] = r;
			sum += r;
		}
		for (int j = 1; j < num_states; j++)
		{
			transition[i][j] /= sum;
		}

		if (!checkDistribution(transition[i], num_states))
		{
			cout << "Degenerate" << endl;
		}

	}

	for (int o = 0; o < num_obs; o++)
	{
		observation[0][o] = 0.0;
	}

	for (int i = 1; i < num_states; i++)
	{
		sum = 0;
		for (int o = 0; o < num_obs; o++)
		{
			int r = (rand() % 10000) + 1;
			observation[i][o] = r;
			sum += r;
		}
		for (int o = 0; o < num_obs; o++)
		{
			observation[i][o] /= sum;
		}
		if (!checkDistribution(observation[i], num_obs))
		{
			cout << "Degenerate" << endl;
		}
	}

	initial_probability[0] = 0.0;
	sum = 0;
	for (int i = 1; i < num_states; i++)
	{
		int r = (rand() % 10000) + 1;
		initial_probability[i] = r;
		sum += r;
	}
	for (int i = 1; i < num_states; i++)
	{
		initial_probability[i] /= sum;
	}

	for (int i = num_states; i < max_state; i++)
	{
		initial_probability[i] = 0.0;
	}
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
	order = params[0];
	num_states = params[1];
	num_obs = params[2];
	max_state = int(pow(num_states, order));

	transition = createMatrix(max_state, num_states);
	observation = createMatrix(num_states, num_obs);
	initial_probability = new double[max_state];

	condstate_dist = createMatrix(num_obs, num_states);
	state_dist = new double[num_states];

	for (int state = 0; state < num_states; state++)
	{
		for (int obs = 0; obs < num_obs; obs++)
		{
			HMMfile >> observation[state][obs];
		}
	}
	for (int i = 0; i < num_obs; i++)
	{
		for (int j = 0; j < num_states; j++)
		{
			HMMfile >> condstate_dist[i][j];
		}
	}
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		for (int state = 0; state < num_states; state++)
		{
			HMMfile >> transition[state_sequence][state];
		}
	}

	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		HMMfile >> initial_probability[state_sequence];

	}

	HMMfile.close();
}

int HMM::sampleStartState()
{
	Distribution d(initial_probability, max_state);
	return d.generate_sample();
}
int HMM::sampleState(int lastState)
{
	Distribution d(transition[lastState], num_states);
	return d.generate_sample();
}
int HMM::sampleWord(int state)
{
	Distribution d(observation[state], num_obs);
	return d.generate_sample();
}

double** HMM::getEmissionMatrix()
{
	return observation;
}

double** HMM::getTransitionMatrix()
{
	return transition;
}

int HMM::getNumObs()
{
	return num_obs;
}

int HMM::getNumStates()
{
	return num_states;
}

HMM::HMM(HMM &h)
{

	//copy constructor
}

/* HMM deconstructor */
HMM::~HMM()
{

	delete[] initial_probability;
	freeMatrix(transition, max_state, num_states);
	freeMatrix(observation, num_states, num_obs);
	delete[] state_dist;
	freeMatrix(condstate_dist, num_obs, num_states);

}
//void HMM::getSimilarWords(int word, vector<pair<int, double> > * similarWords) {
//
//	for (int i = 0; i < num_obs; i++) {
//		similarWords->push_back(pair<int, double> (i, klDivergence(
//				condstate_dist[word], condstate_dist[i], num_states)));
//	}
//	sort(similarWords->begin(), similarWords->end(), sort_scorepairLESSTHAN);
//
//}

/*==============STANDARD FORWARD AND BACKWARD ALGORITHMS===================*/
void HMM::computeForwardMatrix(vector<int> words, double** forward, int len)
{

	forward[0][0] = 0;
	for (int state = 1; state < num_states; state++)
	{
		forward[0][state] = initial_probability[state]
				* observation[state][words[0]];
	}

	for (int state = num_states; state < max_state; state++)
	{
		forward[0][state] = 0;
	}

	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			int state = state_sequence % num_states;
			double sum = 0;
			double o_prob = observation[state][obs];
			for (int previous_state = 0; previous_state < num_states; previous_state++)
			{
				int conditional_state = previous_state * int(pow(num_states,
						order - 1)) + state_sequence / num_states;
				double t = transition[conditional_state][state];
				sum += forward[i - 1][conditional_state]
						* transition[conditional_state][state];
			}
			forward[i][state_sequence] = sum * o_prob;
		}
	}
}

void HMM::computeBackwardMatrix(vector<int> words, double ** backward, int len)
{

	for (int state = 0; state < max_state; state++)
	{
		backward[len - 1][state] = 1.0;
	}

	for (int i = len - 2; i >= 0; i--)
	{
		int obs = words[i + 1];
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			double sum = 0;
			for (int next_state = 0; next_state < num_states; next_state++)
			{
				int next_sequence = (state_sequence % int(pow(num_states, order
						- 1))) + next_state;
				sum += backward[i + 1][next_sequence]
						* transition[state_sequence][next_state]
						* observation[next_state][obs];
			}
			backward[i][state_sequence] = sum;
		}
	}
}
/*======================================================================+*/

/*==============SCALED FORWARD AND BACKWARD ALGORITHMS===================*/
void HMM::computeForwardMatrixScaled(vector<int> words, double** forward,
		double * scaleArray, int len)
{

	double scale = 0;
	forward[0][0] = 0;
	for (int state = 1; state < num_states; state++)
	{
		double p = initial_probability[state] * observation[state][words[0]];
		forward[0][state] = p;
		scale += p;
	}

	for (int state = 1; state < num_states; state++)
	{
		forward[0][state] /= scale;
	}
	scaleArray[0] = scale;

	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		scale = 0;
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			int state = state_sequence % num_states;
			double sum = 0;
			double o_prob = observation[state][obs];
			for (int previous_state = 0; previous_state < num_states; previous_state++)
			{
				int conditional_state = previous_state * int(pow(num_states,
						order - 1)) + state_sequence / num_states;
				sum += forward[i - 1][conditional_state]
						* transition[conditional_state][state];
			}
			double p = sum * observation[state][words[i]];
			forward[i][state_sequence] = p;
			scale += p;
		}
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			forward[i][state_sequence] /= scale;
		}
		scaleArray[i] = scale;
	}
}

void HMM::computeSubForwardMatrixScaled(vector<int> words, double** forward,
		double * scaleArray, int len, int k)
{

	double scale = 0;
	forward[0][0] = 0;
	for (int state = 1; state < num_states; state++)
	{
		double o_prob;
		if (k == 0)
			o_prob = 1;
		else
			o_prob = observation[state][words[0]];

		double p = (1.0 / num_states) * o_prob;
		forward[0][state] = p;
		scale += p;
	}

	for (int state = 1; state < num_states; state++)
	{
		forward[0][state] /= scale;
	}
	scaleArray[0] = scale;

	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		scale = 0;
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			int state = state_sequence % num_states;
			double sum = 0;
			double o_prob = observation[state][obs];
			if (i == k)
			{
				o_prob = 1;

			}
			for (int previous_state = 0; previous_state < num_states; previous_state++)
			{
				int conditional_state = previous_state * int(pow(num_states,
						order - 1)) + state_sequence / num_states;
				sum += forward[i - 1][conditional_state]
						* transition[conditional_state][state];
			}
			double p = sum * o_prob;
			forward[i][state_sequence] = p;
			scale += p;
		}
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			forward[i][state_sequence] /= scale;
		}
		scaleArray[i] = scale;
	}
}

void HMM::computeBackwardMatrixScaled(vector<int> words, double ** backward,
		double * scaleArray, int len)
{

	for (int state = 0; state < max_state; state++)
	{
		backward[len - 1][state] = 1.0;
	}
	for (int i = len - 2; i >= 0; i--)
	{
		int obs = words[i + 1];
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			double sum = 0;
			for (int next_state = 0; next_state < num_states; next_state++)
			{
				int next_sequence = num_states * (state_sequence % int(pow(
						num_states, order - 1))) + next_state;
				sum += backward[i + 1][next_sequence]
						* transition[state_sequence][next_state]
						* observation[next_state][obs];
			}
			backward[i][state_sequence] = sum / scaleArray[i + 1];
		}
	}
}

double HMM::forwardAlgorithmScaled(vector<int> words)
{

	int len = words.size();
	double ** forward = createMatrix(len, max_state);
	double * scaleArray = new double[len];
	computeForwardMatrixScaled(words, forward, scaleArray, len);
	double p = 0;
	for (int i = 0; i < len; i++)
	{
		p += log2(scaleArray[i]);
	}

	freeMatrix(forward, len, max_state);
	delete[] scaleArray;
	return p;
}

double HMM::subForwardAlgorithmScaled(vector<int> words, int k)
{

	int len = words.size();
	double ** forward = createMatrix(len, max_state);
	double * scaleArray = new double[len];
	computeSubForwardMatrixScaled(words, forward, scaleArray, len, k);
	double p = 0;
	for (int i = 0; i < len; i++)
	{
		p += log2(scaleArray[i]);
	}

	freeMatrix(forward, len, max_state);
	delete[] scaleArray;
	return p;
}

/*=======================================================================*/

double HMM::forwardAlgorithm(vector<int> words)
{

	int len = words.size();
	double ** forward = createMatrix(len, max_state);
	computeForwardMatrix(words, forward, len);
	double p = 0;
	for (int i = 0; i < max_state; i++)
	{
		p += forward[len - 1][i];
	}

	freeMatrix(forward, len, max_state);

	return p;
}

double HMM::backwardAlgorithm(vector<int> words)
{
	int len = words.size();
	double ** backward = createMatrix(len, max_state);
	computeBackwardMatrix(words, backward, len);

	double p = 0;
	for (int i = 0; i < max_state; i++)
	{
		p += backward[0][i] * observation[i][words[0]] * initial_probability[i];
	}

	freeMatrix(backward, len, max_state);

	return p;
}

//Forward Algorithm for computation of sequence probability
double HMM::forward(int* words)
{

	int len = sizeof(words) / sizeof(int);
	double *a;
	a = new double[max_state];

	for (int state = 0; state < max_state; state++)
	{
		a[state] = initial_probability[state] * observation[state][words[0]];
	}

	double *current;
	current = new double[max_state];
	for (int i = 1; i < len; i++)
	{
		int obs = words[i];
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			int state = state_sequence % num_states;
			double sum = 0;
			double o_prob = observation[state][obs];
			for (int previous_state = 0; previous_state < num_states; previous_state++)
			{
				int conditional_state = previous_state * int(pow(num_states,
						order - 1)) + state_sequence / num_states;
				double trans = transition[conditional_state][state];
				sum += a[conditional_state] * trans;
			}
			current[state_sequence] = sum * o_prob;
		}
		a = current;
	}
	double sum = 0;
	for (int i = 0; i < max_state; i++)
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
	b = new double[max_state];

	for (int state = 0; state < max_state; state++)
	{
		b[state] = 1.0;
	}
	double *current;
	current = new double[max_state];
	for (int i = len - 2; i >= 0; i--)
	{
		int obs = words[i + 1];

		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			int state = state_sequence % order;
			double sum = 0;
			for (int next_state = 0; next_state < num_states; next_state++)
			{
				double o_prob = observation[next_state][obs];
				int next_sequence = num_states * (state_sequence % int(pow(
						num_states, order - 1))) + next_state;
				//int conditional_state = previous_state*msb + state_sequence/order;
				double trans = transition[state_sequence][next_state];

				sum += b[next_sequence] * trans * o_prob;
			}
			current[state_sequence] = sum;
		}
		b = current;
	}
	double sum = 0;
	for (int i = 0; i < max_state; i++)
	{
		sum += b[i] * observation[i][words[0]] * initial_probability[i];
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
 a = new double[max_state];

 for (int state = 0; state < num_states; state++) {
 a[state] = initial_probability[state]*observation[state][words[0]];
 }

 double *current;
 current = new double[max_state];
 for (int i = 1; i < len; i++){
 int obs = words[i];
 for (int state_sequence = 0; state_sequence < max_state; state_sequence++){
 int state = state_sequence % num_states;
 double sum = 0;
 double o_prob = observation[state][obs];
 for (int previous_state = 0; previous_state < num_states; previous_state++) {
 int conditional_state = previous_state*int(pow(num_states, order-1)) + state_sequence/order;
 double trans = transition[conditional_state][state];
 sum += a[conditional_state] * trans;
 }
 current[state_sequence] = sum*o_prob;
 }
 a = current;
 }
 double max = 0;
 double sum = 0;
 for (int i =0; i<max_state; i++){
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
	emit_count = createMatrix(num_states, num_obs);
	trans_count = createMatrix(max_state, num_states);
	state_count = new double[num_states];
	zeroArray(state_count, num_states);
	obs_count = new double[num_obs];
	zeroArray(obs_count, num_obs);
	sequence_count = new double[max_state];
	zeroArray(sequence_count, max_state);
	init_count = new double[max_state];
	zeroArray(init_count, max_state);

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
		forward = createMatrix(len, max_state);
		//double forward_prob = forwardAlgorithmScaled(words);
		//computeForwardMatrix(words, forward,  len);
		//printMatrix(forward, len, max_state);
		computeForwardMatrixScaled(words, forward, scaleArray, len);
		//printMatrix(forward, len, max_state);

		//COMPUTE_BACKWARD PROBABILITY
		double **backward;
		backward = createMatrix(len, max_state);
		//computeBackwardMatrix(words, backward,  len);
		//printMatrix(backward, len, max_state);		
		computeBackwardMatrixScaled(words, backward, scaleArray, len);
		//printMatrix(backward, len, max_state);

		//BAUM WELCH COUNTS
		for (int i = 0; i < len - 1; i++)
		{
			int obs = words[i];
			int next_obs = words[i + 1];
			for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
			{
				int state = state_sequence % num_states;
				//cout << "for state" << state<<" "<<forward[i][state_sequence]<<"*"<<backward[i][state_sequence]<<endl;
				double gamma = (forward[i][state_sequence]
						* backward[i][state_sequence]);
				if (gamma != gamma)
				{
					cout << "gammma problem" << endl;
					printMatrix(forward, len, max_state);
					printMatrix(backward, len, max_state);
					printArray(scaleArray, len);

					return;
				}
				if (i == 0)
				{
					init_count[state_sequence] += gamma;
				}
				emit_count[state][obs] += gamma;
				obs_count[obs] += gamma;
				state_count[state] += gamma;
				sequence_count[state_sequence] += gamma;
				for (int next_state = 0; next_state < num_states; next_state++)
				{
					int next_sequence = num_states * (state_sequence % int(pow(
							num_states, order - 1))) + next_state;
					double eta = (forward[i][state_sequence]
							* transition[state_sequence][next_state]
							* observation[next_state][next_obs] * backward[i
							+ 1][next_sequence]) / scaleArray[i + 1];
					trans_count[state_sequence][next_state] += eta;
				}
			}
		}
		for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
		{
			int obs = words[len - 1];
			int state = state_sequence % num_states;
			double gamma = (forward[len - 1][state_sequence]
					* backward[len - 1][state_sequence]);
			emit_count[state][obs] += gamma;
			obs_count[obs] += gamma;
			state_count[state] += gamma;
		}
		delete[] scaleArray;
		freeMatrix(forward, len, max_state);
		freeMatrix(backward, len, max_state);

	}//end for every sentence
	trainFile.close();
	updateHMM(emit_count, trans_count, state_count, sequence_count, init_count,
			obs_count);

	freeMatrix(trans_count, max_state, num_states);
	freeMatrix(emit_count, num_states, num_obs);
	delete[] state_count;
	delete[] obs_count;
	delete[] sequence_count;

}

/* HMM EM Training; 
 STRING inputFile
 inputFile is a file containing a list of training files
 
 */

void HMM::trainParallel(vector<string> files_list)
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
	emit_count = createMatrix(num_states, num_obs);
	trans_count = createMatrix(max_state, num_states);
	state_count = new double[num_states];
	zeroArray(state_count, num_states);
	obs_count = new double[num_obs];
	zeroArray(obs_count, num_obs);
	sequence_count = new double[max_state];
	zeroArray(sequence_count, max_state);
	init_count = new double[max_state];
	zeroArray(init_count, max_state);

	double **temit_count;
	double **ttrans_count;
	double *tstate_count;
	double *tsequence_count;
	double *tinit_count;
	double *tobs_count;

	temit_count = createMatrix(num_states, num_obs);
	ttrans_count = createMatrix(max_state, num_states);
	tstate_count = new double[num_states];
	tobs_count = new double[num_obs];
	zeroArray(tstate_count, num_states);
	tsequence_count = new double[max_state];
	zeroArray(tsequence_count, max_state);
	tinit_count = new double[max_state];
	zeroArray(tinit_count, max_state);

	for (int i = start; i < end; i++)
	{
		const char* inputFile = files_list[i].c_str();
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
			forward = createMatrix(len, max_state);
			computeForwardMatrixScaled(words, forward, scaleArray, len);
			//printMatrix(forward, len, max_state);

			//COMPUTE_BACKWARD PROBABILITY
			double **backward;
			backward = createMatrix(len, max_state);
			computeBackwardMatrixScaled(words, backward, scaleArray, len);
			//printMatrix(backward, len, max_state);

			//BAUM WELCH COUNTS
			for (int i = 0; i < len - 1; i++)
			{
				int obs = words[i];
				int next_obs = words[i + 1];
				for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
				{
					int state = state_sequence % num_states;
					double gamma = (forward[i][state_sequence]
							* backward[i][state_sequence]);
					if (i == 0)
					{
						init_count[state_sequence] += gamma;
					}
					emit_count[state][obs] += gamma;
					obs_count[obs] += gamma;
					state_count[state] += gamma;
					sequence_count[state_sequence] += gamma;
					for (int next_state = 0; next_state < num_states; next_state++)
					{
						int next_sequence = num_states * (state_sequence % int(
								pow(num_states, order - 1))) + next_state;
						double eta = (forward[i][state_sequence]
								* transition[state_sequence][next_state]
								* observation[next_state][next_obs]
								* backward[i + 1][next_sequence])
								/ scaleArray[i + 1];
						trans_count[state_sequence][next_state] += eta;
					}
				}
			}
			for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
			{
				int obs = words[len - 1];
				int state = state_sequence % num_states;
				double gamma = (forward[len - 1][state_sequence] * backward[len
						- 1][state_sequence]);
				emit_count[state][obs] += gamma;
				obs_count[obs] += gamma;
				state_count[state] += gamma;
				//	sequence_count[state_sequence] += gamma;
			}
			delete[] scaleArray;
			freeMatrix(forward, len, max_state);
			freeMatrix(backward, len, max_state);
		}//end for every sentence

		trainFile.close();
		//	cout << " Training File Close, Updating Parameters " << inputFile << endl;
	} // for every file


	//Collect parameters on root
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		MPI_Reduce(trans_count[state_sequence], ttrans_count[state_sequence],
				num_states, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	}
	MPI_Reduce(sequence_count, tsequence_count, max_state, MPI_DOUBLE, MPI_SUM,
			root, MPI_COMM_WORLD);
	MPI_Reduce(init_count, tinit_count, max_state, MPI_DOUBLE, MPI_SUM, root,
			MPI_COMM_WORLD);
	for (int state = 0; state < num_states; state++)
	{
		MPI_Reduce(emit_count[state], temit_count[state], num_obs, MPI_DOUBLE,
				MPI_SUM, root, MPI_COMM_WORLD);
	}
	MPI_Reduce(state_count, tstate_count, num_states, MPI_DOUBLE, MPI_SUM,
			root, MPI_COMM_WORLD);
	MPI_Reduce(obs_count, tobs_count, num_obs, MPI_DOUBLE, MPI_SUM, root,
			MPI_COMM_WORLD);

	//Send updated parameters too all children
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		MPI_Bcast(ttrans_count[state_sequence], num_states, MPI_DOUBLE, root,
				MPI_COMM_WORLD);
	}
	MPI_Bcast(tsequence_count, max_state, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tinit_count, max_state, MPI_DOUBLE, root, MPI_COMM_WORLD);
	for (int state = 0; state < num_states; state++)
	{
		MPI_Bcast(temit_count[state], num_obs, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}
	MPI_Bcast(tstate_count, num_states, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tobs_count, num_obs, MPI_DOUBLE, root, MPI_COMM_WORLD);

	//cout << "Update Step" << endl;
	updateHMM(temit_count, ttrans_count, tstate_count, tsequence_count,
			tinit_count, tobs_count);

	freeMatrix(trans_count, max_state, num_states);
	freeMatrix(emit_count, num_states, num_obs);
	delete[] state_count;
	delete[] sequence_count;
	delete[] init_count;
	delete[] obs_count;

	freeMatrix(temit_count, num_states, num_obs);
	freeMatrix(ttrans_count, max_state, num_states);
	delete[] tstate_count;
	delete[] tsequence_count;
	delete[] tinit_count;
	delete[] tobs_count;

}

void HMM::updateHMM(double** emit_count, double** trans_count,
		double* state_count, double* sequence_count, double* initial_count,
		double* obs_count)
{

	double state_total = 0;
	for (int state = 1; state < num_states; state++)
	{
		double smooth = 1 / ((num_states - 1) * num_obs);
		for (int obs = 0; obs < num_obs; obs++)
		{
			observation[state][obs] = ((emit_count[state][obs]) + smooth)
					/ (state_count[state] + smooth * num_obs);
			condstate_dist[obs][state] = (emit_count[state][obs] + smooth)
					/ (obs_count[obs] + smooth * (num_states - 1));
			state_total += state_count[state];

		}
	}
	for (int state = 0; state < num_states; state++)
	{
		state_dist[state] = state_count[state] / state_total;
	}
	double totalInit = 0;
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		if (sequence_count[state_sequence] != 0)
		{
			double smooth = 1 / (max_state * num_states - 1);
			transition[state_sequence][0] = 0;
			for (int state = 1; state < num_states; state++)
			{
				transition[state_sequence][state]
						= (trans_count[state_sequence][state] + smooth)
								/ (sequence_count[state_sequence] + smooth
										* (num_states - 1));
			}
		}
		else
		{
			for (int state = 0; state < num_states; state++)
			{
				transition[state_sequence][state] = 0;
			}
		}
		initial_probability[state_sequence] = initial_count[state_sequence];
		totalInit += initial_count[state_sequence];
	}

	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
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

	emit_count = createMatrix(num_states, num_obs);
	trans_count = createMatrix(max_state, num_states);
	state_count = new double[num_states];
	zeroArray(state_count, num_states);
	obs_count = new double[num_obs];
	zeroArray(obs_count, num_obs);
	sequence_count = new double[max_state];
	zeroArray(sequence_count, max_state);
	init_count = new double[max_state];
	zeroArray(init_count, max_state);

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
			int origState = (rand() % (num_states - 1)) + 1;
			int obs = words[i];
			int prev_sequence = 0;
			int r = 0;
			stateArray[i] = origState;
			while ((r < order) && (i - 1 - r) >= 0)
			{
				prev_sequence += stateArray[(i - 1) - r] * int(pow(num_states,
						r));
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
			while ((r < order) && (k - 1 - r) >= 0)
			{
				prev_sequence += stateArray[(k - 1) - r] * int(pow(num_states,
						r));
				r++;
			}
			//	cout << "Done Compute prev_seq" << endl;
			int origState = stateArray[k];
			int nextState = stateArray[k + 1];
			int next_sequence = num_states * (prev_sequence % int(pow(
					num_states, order - 1))) + origState;
			double *dist = new double[num_states];
			double totalp = 0;
			for (int state = 0; state < num_states; state++)
			{
				int state_sequence = num_states * (prev_sequence % int(pow(
						num_states, order - 1))) + state;
				if (prev_sequence == 0)
					dist[state] = observation[state][obs]
							* initial_probability[state]
							* transition[state_sequence][nextState];
				else
					dist[state] = observation[state][obs]
							* transition[prev_sequence][state]
							* transition[state_sequence][nextState];
				totalp += dist[state];
			}
			renormalize(dist, num_states);
			Distribution d(dist, num_states);
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
			next_sequence = num_states * (prev_sequence % int(pow(num_states,
					order - 1))) + sample;
			state_count[sample]++;
			trans_count[next_sequence][nextState]++;
			sequence_count[next_sequence]++;
			emit_count[sample][obs]++;

			//	cout << "Done Update params" << endl;

		}
	}//end for every sentence
	trainFile.close();
	updateHMM(emit_count, trans_count, state_count, sequence_count, init_count,
			obs_count);

	freeMatrix(trans_count, max_state, num_states);
	freeMatrix(emit_count, num_states, num_obs);
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
	emit_count = createMatrix(num_states, num_obs);
	trans_count = createMatrix(max_state, num_states);

	state_count = new double[num_states];
	zeroArray(state_count, num_states);

	obs_count = new double[num_obs];
	zeroArray(obs_count, num_obs);

	sequence_count = new double[max_state];
	zeroArray(sequence_count, max_state);

	init_count = new double[max_state];
	zeroArray(init_count, max_state);

	double **temit_count;
	double **ttrans_count;
	double *tstate_count;
	double *tsequence_count;
	double *tinit_count;
	double *tobs_count;

	temit_count = createMatrix(num_states, num_obs);
	ttrans_count = createMatrix(max_state, num_states);

	tstate_count = new double[num_states];
	zeroArray(tstate_count, num_states);

	tobs_count = new double[num_obs];
	zeroArray(tobs_count, num_obs);

	tsequence_count = new double[max_state];
	zeroArray(tsequence_count, max_state);

	tinit_count = new double[max_state];
	zeroArray(tinit_count, max_state);

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
				int origState = (rand() % (num_states - 1)) + 1;
				int obs = words[i];
				int prev_sequence = 0;
				int r = 0;
				stateArray[i] = origState;

				if (i == 0)
					init_count[origState]++;
				else
				{
					while ((r < order) && (i - 1 - r) >= 0)
					{
						prev_sequence += stateArray[(i - 1) - r] * int(pow(
								num_states, r));
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
				while ((r < order) && (k - 1 - r) >= 0)
				{
					prev_sequence += stateArray[(k - 1) - r] * int(pow(
							num_states, r));
					r++;
				}
				//		cout << "Done Compute seq " <<endl;
				int origState = stateArray[k];
				int nextState = stateArray[k + 1];
				int next_sequence = num_states * (prev_sequence % int(pow(
						num_states, order - 1))) + nextState;
				double *dist = new double[num_states];
				double totalp = 0;
				for (int state = 0; state < num_states; state++)
				{
					int state_sequence = num_states * (prev_sequence % int(pow(
							num_states, order - 1))) + state;
					if (prev_sequence == 0)
						dist[state] = observation[state][obs]
								* initial_probability[state]
								* transition[state_sequence][nextState];
					else
						dist[state] = observation[state][obs]
								* transition[prev_sequence][state]
								* transition[state_sequence][nextState];
					totalp += dist[state];
				}
				renormalize(dist, num_states);
				Distribution d(dist, num_states);
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
				//			next_sequence = num_states*(prev_sequence  % int(pow(num_states, order-1))) + sample;
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
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		MPI_Reduce(trans_count[state_sequence], ttrans_count[state_sequence],
				num_states, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
	}
	MPI_Reduce(sequence_count, tsequence_count, max_state, MPI_DOUBLE, MPI_SUM,
			root, MPI_COMM_WORLD);
	MPI_Reduce(init_count, tinit_count, max_state, MPI_DOUBLE, MPI_SUM, root,
			MPI_COMM_WORLD);
	for (int state = 0; state < num_states; state++)
	{
		MPI_Reduce(emit_count[state], temit_count[state], num_obs, MPI_DOUBLE,
				MPI_SUM, root, MPI_COMM_WORLD);
	}
	MPI_Reduce(state_count, tstate_count, num_states, MPI_DOUBLE, MPI_SUM,
			root, MPI_COMM_WORLD);
	MPI_Reduce(obs_count, tobs_count, num_obs, MPI_DOUBLE, MPI_SUM, root,
			MPI_COMM_WORLD);

	//Send updated parameters too all children
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		MPI_Bcast(ttrans_count[state_sequence], num_states, MPI_DOUBLE, root,
				MPI_COMM_WORLD);
	}
	MPI_Bcast(tsequence_count, max_state, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tinit_count, max_state, MPI_DOUBLE, root, MPI_COMM_WORLD);
	for (int state = 0; state < num_states; state++)
	{
		MPI_Bcast(temit_count[state], num_obs, MPI_DOUBLE, root, MPI_COMM_WORLD);
	}
	MPI_Bcast(tstate_count, num_states, MPI_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(tobs_count, num_obs, MPI_DOUBLE, root, MPI_COMM_WORLD);

	//cout << "Update Step" << endl;
	updateHMM(temit_count, ttrans_count, tstate_count, tsequence_count,
			tinit_count, tobs_count);

	freeMatrix(trans_count, max_state, num_states);
	freeMatrix(emit_count, num_states, num_obs);
	delete[] state_count;
	delete[] sequence_count;
	delete[] init_count;
	delete[] obs_count;

	freeMatrix(temit_count, num_states, num_obs);
	freeMatrix(ttrans_count, max_state, num_states);
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
	HMMfile << order << " " << num_states << " " << num_obs << endl;
	for (int i = 0; i < num_states; i++)
	{
		for (int j = 0; j < num_obs; j++)
		{
			HMMfile << observation[i][j] << " ";
		}
		HMMfile << endl;
	}
	for (int i = 0; i < num_obs; i++)
	{
		for (int j = 0; j < num_states; j++)
		{
			HMMfile << condstate_dist[i][j] << " ";
		}
		HMMfile << endl;
	}
	HMMfile << endl;
	for (int i = 0; i < max_state; i++)
	{
		for (int j = 0; j < num_states; j++)
		{
			HMMfile << transition[i][j] << " ";
		}
		HMMfile << endl;
	}
	for (int i = 0; i < max_state; i++)
	{
		HMMfile << initial_probability[i] << endl;
	}
	HMMfile << endl;
	for (int j = 0; j < num_states; j++)
	{
		HMMfile << state_dist[j] << endl;
	}
	HMMfile.close();
}

void HMM::checkTransition()
{
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		double max = 0;
		double entropy = 0;
		for (int state = 0; state < num_states; state++)
		{
			double p = transition[state_sequence][state];
			if (p > max)
				max = p;
			entropy -= p * log2(p);
		}

		cout << "For state_seqeunce " << state_sequence << " entropy: "
				<< entropy << "  and max: " << max << endl;
	}
}
void HMM::checkObservation()
{
	for (int state = 0; state < num_states; state++)
	{
		double max = 0;
		double entropy = 0;
		for (int obs = 0; obs < num_obs; obs++)
		{
			double p = observation[state][obs];
			if (p > max)
				max = p;
			entropy -= p * log2(p);
		}

		cout << "For state  " << state << " entropy: " << entropy
				<< "  and max: " << max << endl;
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
	order = params[0];
	num_states = params[1];
	num_obs = params[2];
	max_state = int(pow(num_states, order));

	transition = createMatrix(max_state, num_states);
	observation = createMatrix(num_states, num_obs);
	initial_probability = new double[max_state];

	condstate_dist = createMatrix(num_obs, num_states);
	state_dist = new double[num_states];

	for (int state = 0; state < num_states; state++)
	{
		for (int obs = 0; obs < num_obs; obs++)
		{
			HMMfile >> observation[state][obs];
		}
	}
	for (int i = 0; i < num_obs; i++)
	{
		for (int j = 0; j < num_states; j++)
		{
			HMMfile >> condstate_dist[i][j];
		}
	}
	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		for (int state = 0; state < num_states; state++)
		{
			HMMfile >> transition[state_sequence][state];
		}
	}

	for (int state_sequence = 0; state_sequence < max_state; state_sequence++)
	{
		HMMfile >> initial_probability[state_sequence];

	}

	for (int j = 0; j < num_states; j++)
	{
		HMMfile >> state_dist[j];
	}

	HMMfile.close();
}
/*
 void HMM::recomputeEmitDist()
 {
 double * word_dist = new double[num_obs];

 for (int w = 0; w < num_obs; w++)
 {
 word_dist[w] = 0;
 for (int i = 0; i <num_states; i++)
 {
 word_dist[w] += observation[i][w]*state_dist[i];
 }
 }

 renormalize(word_dist, num_obs);
 for (int state =1; state < num_states; state++)
 {
 for (int obs = 0; obs < num_obs; obs++)
 {
 observation[state][obs] = condstate_dist[obs][state]*word_dist[obs];
 }
 renormalize(observation[state], num_obs);
 }
 }*/

