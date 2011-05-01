CC = mpic++
DEBUG = -g -showme
FLAGS = -g -O2
TAR = tar czvf
COMPRESS = gzip
PROJ = pHMM
VERSION = `date +%Y%m%d`

DELIVERY = Makefile *.h *.cpp *.py
PROGS = createVocab trainHMM testHMM evalIter
SRCS = hmm.cpp basicProbability.cpp utilities.cpp distribution.cpp
OBJS = hmm.o basicProbability.o distribution.o utilities.o

all: $(PROGS)
tar: 
	$(TAR) ${PROJ}-${VERSION}.tar ${DELIVERY}

.C.o:
	 $(CC) $< -c $(FLAGS) $(INC)

trainHMM: trainHMM.o $(OBJS)
	$(CC) -o $@ trainHMM.o $(OBJS) $(FLAGS) $(INC) $(LIBS) $(XLIBS) -lm

testHMM: testHMM.o $(OBJS)
	$(CC) -o $@ testHMM.o $(OBJS) $(FLAGS) $(INC) $(LIBS) $(XLIBS) -lm

evalIter: evalIter.o $(OBJS)
	$(CC) -o $@ evalIter.o $(OBJS) $(FLAGS) $(INC) $(LIBS) $(XLIBS) -lm

createVocab: createVocab.o $(OBJS)
	$(CC) -o $@ createVocab.o $(OBJS) $(FLAGS) $(INC) $(LIBS) $(XLIBS) -lm

	
clean:
	rm *.o