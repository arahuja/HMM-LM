#include "hmm.h"
#include <iostream>
#include<stdlib.h>

int main(int argc, char** argv){
	if (argc < 7) {
		cout << "Too few argument" << endl;
	} else {
		int order = atoi(argv[1]);
		int num_states = atoi(argv[2]);
		int num_obs = atoi(argv[3]);
		HMM h(order, num_states, num_obs);
		int iterations = atoi(argv[6]);
		int count = 0;
		while (count < iterations){
			count++;
			h.trainFromFile(argv[4]);
		}
		h.saveHMM(argv[5]);

	}

	


	return 0;
}
