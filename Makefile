CC = mpic++
DEBUG = -g -showme
CFLAGS = -c -g -O2
LFLAGS = -O2
TAR = tar czvf
COMPRESS = gzip
PROJ = pHMM
VERSION = `date +%Y%m%d`

DELIVERY = Makefile *.h *.cpp *.py
PROGS = createVocab
SRCS = hmm.cpp basicProbability.cpp utilities.cpp distribution.cpp
OBJS = $(SRCS:.cpp=.o}

all: $(PROGS)
tar: 
	$(TAR) ${PROJ}-${VERSION}.tar ${DELIVERY}

createVocab: 
	$(CC) $(LFLAGS) createVocab.o utilities.o distribution.o -o $@

.o:
	$(CC) *.cpp

clean:
	rm *.o
