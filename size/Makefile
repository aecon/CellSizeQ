.POSIX:
.SUFFIXES:
CC = gcc-12
#CFLAGS = -O3
#CFLAGS_OPENMP = -fopenmp
CFLAGS_SHARED = -shared -fPIC

all: objects.so
M = \
objects.so\

objects.so: objects.c
	$(CC) $(CFLAGS) $(CFLAGS_OPENMP) $(CFLAGS_SHARED) -o $@ $< $(LDFLAGS)

clean:; rm -f objects.so