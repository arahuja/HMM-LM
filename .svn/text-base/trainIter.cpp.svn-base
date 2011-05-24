#include "hmm.h"
#include <iostream>
#include<stdlib.h>
#include<cstring>
#include<sstream>
#include "mpi.h"
#include <vector>
#include<fstream>
using namespace std;

int main(int argc, char** argv){
	if (argc < 7) {
		cout << "Too few argument" << endl;
	} else {

		/* Read in HMM parameters */	
		int order = atoi(argv[1]);
		int num_states = atoi(argv[2]);
		int num_obs = atoi(argv[3]);

		HMM h(order, num_states, num_obs);
		
		/*Read in Training Files */
		ifstream filesList(argv[4]);
		string line;
		int print_mark = atoi(argv[7]);
		vector<string> files;
		while(getline(filesList, line))
		{
			files.push_back(line);	
		}
		string iter  = "-iter";
		
		int iterations = atoi(argv[6]);
		int count = 0;
		/*Start MPI sessions */
		MPI::Init();
		int rank = MPI::COMM_WORLD.Get_rank();
		while (count < iterations)
		{
			count++;
			h.trainParallel(files);
			/*Save iteration from root*/
			if (rank == 0 && (count % print_mark) == 0){
				stringstream s;
				s << count;
				string str = s.str();
				string name = argv[5] + iter + str;
				char* file = new char[name.size() +1];
				strcpy(file, name.c_str());
				h.saveHMM(file);
		//		cout << "Finished Iteration: "<<count<<endl;
			}
		}
		MPI::Finalize();

	}

	return 0;
}
