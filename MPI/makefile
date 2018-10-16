# ============================================================
# INTEL COMPILERS
# ============================================================

CC = mpicc
CFLAGS = -Wall -g -openmp -I.
CFLAGS = -Wall -openmp -I.
CFLAGS = -Wall -openmp -I. $(MKL_COPTIONS)

CLINKER = mpicc
FLINKER = mpif77
LDFLAGS = -openmp
LIBLIST = -L. -lhbio -lclock -lsparsenew -lvector -lm -lc
LIBLIST = -L. -lhbio -lclock -lvector -lm -lc
LIBLIST = -L. -lhbio -lclock -lm -lc
LIBLIST = -L. -lsparse -lvector -lclock -lm -lc

LIBMKL = -L$(DIR_MKL) $(MKL_FMULTIS_INTEL)

# ============================================================
# GNU COMPILERS
# ============================================================

CC = mpicc
CFLAGS = -Wall -g -fopenmp -I.
CLFLAGS = -Wall -fopenmp -I. 
CLFLAGS = -Wall -fopenmp -I. $(MKL_COPTIONS)

CLINKER = mpicc
FLINKER = mpif77
LDFLAGS = -fopenmp
LIBLIST = -L. -lhbio -lclock -lsparsenew -lvector -lm -lc
LIBLIST = -L. -lhbio -lclock -lvector -lm -lc
LIBLIST = -L. -lhbio -lclock -lm -lc
LIBLIST = -L. -lsparse -lvector -lclock -lm -lc

LIBMKL = -L$(DIR_MKL) $(MKL_FMULTIS_INTEL)

# ============================================================

AR = ar
ARFLAGS = ru

RL = ranlib

# ============================================================

OBJS_CLOCK  = reloj.o 
OBJS_VECTOR = ScalarVectors.o
OBJS_SPARSE = hb_io.o SparseProduct.o

OBJS = $(OBJS_CLOCK) $(OBJS_VECTOR) $(OBJS_SPARSE) 

# ============================================================

default: libclock.a libvector.a libsparse.a CG_MPI

libshared.a : $(OBJS)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

libclock.a : $(OBJS_CLOCK)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

libvector.a : $(OBJS_VECTOR)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

libsparse.a : $(OBJS_SPARSE)
	$(AR) $(ARFLAGS) $@ $?
	$(RL) $(RLFLAGS) $@

CG_MPI: CG_MPI.o ToolsMPI.o 
	$(CLINKER) $(LDFLAGS) -o CG_MPI CG_MPI.o ToolsMPI.o $(LIBMKL) $(LIBLIST)

# ============================================================

.c.o:
	echo compilando
	$(CC) $(CFLAGS) -c $*.c

clean:
	rm *.o

clear:
	rm *.o *.a

# ============================================================
