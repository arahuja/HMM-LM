#include "utilities.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{

	if (argc > 1)
	{
		char* vocab = argv[1];
		createVocab(vocab, argv[2]);
	}
	else
	{
		cout << "Not enough input files" << endl;
	}

	return 0;

}
