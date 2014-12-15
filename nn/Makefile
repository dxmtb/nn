CC = g++
CCFLAGS = -fPIC -O3 -Wall -ffast-math -msse -msse2
LINKFLAGS = -shared
INPUT = pooling.cpp
TARGET = pooling.so

all: $(INPUT)
	$(CC) -c $(CCFLAGS) $(INPUT)
	$(CC) $(LINKFLAGS) -o $(TARGET) *.o
clean:
	rm *.so
	rm *.o

