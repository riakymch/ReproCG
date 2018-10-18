#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mkl_blas.h>
#include <mpi.h>
#include <hb_io.h>

#include "reloj.h"
#include "ScalarVectors.h"
#include "SparseProduct.h"
#include "ToolsMPI.h"

// ================================================================================

#define PRECOND 0

void ConjugateGradient (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, double *rbuf, int myId) {
    int size = mat.dim2, sizeR = mat.dim1; 
    int IONE = 1; 
    double DONE = 1.0, DMONE = -1.0, DZERO = 0.0;
    int n, n_dist, iter, maxiter, nProcs;
    double beta, tol, rho, alpha, umbral;
    double *res = NULL, *z = NULL, *d = NULL, *y = NULL;
    double *aux = NULL;
    double t1, t2, t3, t4;
    double p1, p2, p3, p4, pp1 = 0.0, pp2 = 0.0;
#if PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    n = size; n_dist = sizeR; maxiter = size; umbral = 1.0e-8;
    CreateDoubles (&res, n_dist); CreateDoubles (&z, n_dist); 
    CreateDoubles (&d, n_dist);  
#if PRECOND
    CreateDoubles (&y, n_dist);
    CreateInts (&posd, n_dist);
    CreateDoubles (&diags, n_dist);
    GetDiagonalSparseMatrix2 (mat, dspls[myId], diags, posd);
//#pragma omp parallel for
    for (i=0; i<n_dist; i++) diags[i] = DONE / diags[i];
#endif
    CreateDoubles (&aux, n); 

    if (myId == 0) reloj (&t1, &t2);
    iter = 0;
    reloj (&p1, &p2);
    MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    InitDoubles (z, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, z);            			// z = A * x
    reloj (&p3, &p4); pp1 = (p3 - p1); pp2 = (p4 - p2);
    dcopy (&n_dist, b, &IONE, res, &IONE);                          		// res = b
    daxpy (&n_dist, &DMONE, z, &IONE, res, &IONE);                      // res -= z
#if PRECOND
    VvecDoubles (DONE, diags, res, DZERO, y, n_dist);                    // y = D^-1 * res
#else
    y = res;																														// y = res
#endif
    dcopy (&n_dist, y, &IONE, d, &IONE);                                // d = y
    beta = ddot (&n_dist, res, &IONE, y, &IONE);                        // beta = res' * y
    //MPI_Allreduce (MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // ReproAllReduce -- Begin
    //MPI_Allgather(&beta, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Gather(&beta, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (myId == 0) {
        beta = 0.0;
        for (int ired = 0; ired < nProcs; ired++) {
            beta += rbuf[ired];
        }
    }
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ReproAllReduce -- End
#ifdef PRECOND
//              tol = dnrm2 (&n_dist, res, &IONE);                                      // tol = norm (res)
    tol = ddot (&n_dist, res, &IONE, res, &IONE);                      // beta = res' * res                     
    MPI_Allreduce (MPI_IN_PLACE, &tol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    tol = sqrt (tol);                                                                                                   // tol = norm (res)
#else
    tol = sqrt (beta);                                                                                                  // tol = norm (res)
#endif
    while ((iter < maxiter) && (tol > umbral)) {
        if (myId == 0) reloj (&p1, &p2);
        MPI_Allgatherv (d, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (z, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, z);            		// z = A * d
        if (myId == 0) printf ("(%d,%20.10e)\n", iter, tol);
        if (myId == 0) {
            reloj (&p3, &p4); pp1 += (p3 - p1); pp2 += (p4 - p2);
        }
        rho = ddot (&n_dist, d, &IONE, z, &IONE);                   
        //MPI_Allreduce (MPI_IN_PLACE, &rho, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // ReproAllReduce -- Begin
        //MPI_Allgather(&rho, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Gather(&rho, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (myId == 0) {
            rho = 0.0;
            for (int ired = 0; ired < nProcs; ired++) {
                rho += rbuf[ired];
            }
        }
        MPI_Bcast(&rho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ReproAllReduce -- End
        rho = beta / rho;
        daxpy (&n_dist, &rho, d, &IONE, x, &IONE);                      	// x += rho * d;
        rho = -rho;
        daxpy (&n_dist, &rho, z, &IONE, res, &IONE);                      // res -= rho * z
#if PRECOND
        VvecDoubles (DONE, diags, res, DZERO, y, n_dist);                 // y = D^-1 * res
#else
        y = res;																													// y = res
#endif
        alpha = beta;                                                 		// alpha = beta
        beta = ddot (&n_dist, res, &IONE, y, &IONE);                      // beta = res' * y                     
        //MPI_Allreduce (MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // ReproAllReduce -- Begin
        //MPI_Allgather(&beta, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Gather(&beta, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (myId == 0) {
            beta = 0.0;
            for (int ired = 0; ired < nProcs; ired++) {
                beta += rbuf[ired];
            }
        }
        MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ReproAllReduce -- End
        alpha = beta / alpha;                                         		// alpha = beta / alpha
        dscal (&n_dist, &alpha, d, &IONE);                                // d = alpha * d
        daxpy (&n_dist, &DONE, y, &IONE, d, &IONE);                       // d += y
#ifdef PRECOND
//              tol = dnrm2 (&n_dist, res, &IONE);                                      // tol = norm (res)
       tol = ddot (&n_dist, res, &IONE, res, &IONE);                      // beta = res' * res                     
       MPI_Allreduce (MPI_IN_PLACE, &tol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       tol = sqrt (tol);                                                                                                   // tol = norm (res)
#else
       tol = sqrt (beta);                                                                                                  // tol = norm (res)
#endif
       iter++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) reloj (&t3, &t4);

    if (myId == 0)
        printf ("Fin(%d) --> (%d,%20.10e) tiempo (%20.10e,%20.10e) prod (%20.10e,%20.10e)\n", 
                n, iter, tol, t3-t1, t4-t2, pp1, pp2);

    RemoveDoubles (&aux); RemoveDoubles (&res); RemoveDoubles (&z); RemoveDoubles (&d);
#if PRECOND
    RemoveDoubles (&diags); RemoveInts (&posd); RemoveDoubles(&y);
#endif
}

/*********************************************************************************/

int main (int argc, char **argv) {
    int dim; 
    double *vec = NULL, *sol1 = NULL, *sol2 = NULL;
    int index = 0, indexL = 0;
    SparseMatrix mat  = {0, 0, NULL, NULL, NULL}, sym = {0, 0, NULL, NULL, NULL};

    int root = 0, myId, nProcs;
    int dimL, *vdimL = NULL, *vdspL = NULL;
    SparseMatrix matL = {0, 0, NULL, NULL, NULL};
    double *vecL = NULL, *sol1L = NULL, *sol2L = NULL, *rbuf = NULL;

    /***************************************/

    MPI_Init (&argc, &argv);

    // Definition of the variables nProcs and myId
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    root = nProcs-1;
    root = 0;

    /***************************************/

    // Creating the matrix
    if (myId == root) {
        ReadMatrixHB (argv[1], &sym);
        DesymmetrizeSparseMatrices (sym, 0, &mat, 0);
        dim = mat.dim1;
    }

    // Distributing the matrix
    CreateInts (&vdimL, nProcs); CreateInts (&vdspL, nProcs); 
    dim = DistributeMatrix (mat, index, &matL, indexL, vdimL, vdspL, root, MPI_COMM_WORLD);
    dimL = vdimL[myId];

    // Creating the vectors
    if (myId == root) {
        CreateDoubles (&vec , dim);
        CreateDoubles (&sol1, dim);
        CreateDoubles (&sol2, dim);
        CreateDoubles (&rbuf, nProcs);
        InitRandDoubles (vec, dim, -1.0, 1.0);
        InitDoubles (sol1, dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
        InitDoubles (rbuf , nProcs, 0.0, 0.0);
    } else {
        CreateDoubles (&vec , dim);
        CreateDoubles (&sol2, dim);
        InitDoubles (vec , dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
    }
    CreateDoubles (&vecL , dimL);
    CreateDoubles (&sol1L, dimL);
    CreateDoubles (&sol2L, dimL);
    InitDoubles (vecL , dimL, 0.0, 0.0);
    InitDoubles (sol1L, dimL, 0.0, 0.0);
    InitDoubles (sol2L, dimL, 0.0, 0.0);

    /***************************************/

    int i, IONE = 1;
    double beta;
    if (myId == root) {
        InitDoubles (vec, dim, 1.0, 0.0);
        InitDoubles (sol1, dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
        ProdSparseMatrixVectorByRows (mat, 0, vec, sol1);
    }
    MPI_Scatterv (sol1, vdimL, vdspL, MPI_DOUBLE, sol1L, dimL, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Scatterv (sol2, vdimL, vdspL, MPI_DOUBLE, sol2L, dimL, MPI_DOUBLE, root, MPI_COMM_WORLD);

    ConjugateGradient (matL, sol2L, sol1L, vdimL, vdspL, rbuf, myId);

    // Error computation
    for (i=0; i<dimL; i++) sol2L[i] -= 1.0;
    beta = ddot (&dimL, sol2L, &IONE, sol2L, &IONE);                       
    //MPI_Allreduce (MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // ReproAllReduce -- Begin
    //MPI_Allgather(&beta, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Gather(&beta, 1, MPI_DOUBLE, rbuf, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (myId == 0) {
        beta = 0.0;
        for (int ired = 0; ired < nProcs; ired++) {
            beta += rbuf[ired];
        }
    }
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ReproAllReduce -- End
    beta = sqrt(beta);
    if (myId == 0) printf ("error = %10.5e\n", beta);

    /***************************************/
    // Freeing memory
    RemoveDoubles (&sol2L); RemoveDoubles (&sol1L); RemoveDoubles (&vecL);
    RemoveInts (&vdspL); RemoveInts (&vdimL); 
    if (myId == root) {
        RemoveDoubles (&sol2); RemoveDoubles (&sol1); RemoveDoubles (&vec);
        RemoveSparseMatrix (&mat); RemoveSparseMatrix (&sym);
    } else {
        RemoveDoubles (&sol2); RemoveDoubles (&vec);
    }

    MPI_Finalize ();

    return 0;
}
