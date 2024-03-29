#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mkl_blas.h>
#include <mpi.h>
#include <hb_io.h>
#include <vector>

#include "reloj.h"
#include "ScalarVectors.h"
#include "SparseProduct.h"
#include "ToolsMPI.h"
#include "matrix.h"
#include "common.h"

#include "exblas/exdot.h"

// ================================================================================

#define DIRECT_ERROR 0
#define PRECOND 1
#define VECTOR_OUTPUT 0

void ConjugateGradient (SparseMatrix mat, double *x, double *b, int *sizes, int *dspls, double *rbuf, int myId) {
    int size = mat.dim2, sizeR = mat.dim1; 
    int IONE = 1; 
    double DONE = 1.0, DMONE = -1.0, DZERO = 0.0;
    int n, n_dist, iter, maxiter, nProcs;
    double beta, tol, rho, alpha, umbral;
    double *res = NULL, *z = NULL, *d = NULL, *y = NULL;
    double *aux = NULL;
    double t1, t2, t3, t4;
#if PRECOND
    int i, *posd = NULL;
    double *diags = NULL;
#endif

    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    n = size; n_dist = sizeR; maxiter = 16 * size; umbral = 1.0e-8;
    CreateDoubles (&res, n_dist); CreateDoubles (&z, n_dist); 
    CreateDoubles (&d, n_dist);  
#if DIRECT_ERROR
    // init exact solution
    double *res_err = NULL, *x_exact = NULL;
	CreateDoubles (&x_exact, n_dist);
	CreateDoubles (&res_err, n_dist);
    InitDoubles(x_exact, n_dist, DONE, DZERO);
#endif // DIRECT_ERROR 

#if PRECOND
    CreateDoubles (&y, n_dist);
    CreateInts (&posd, n_dist);
    CreateDoubles (&diags, n_dist);
    GetDiagonalSparseMatrix2 (mat, dspls[myId], diags, posd);
#pragma omp parallel for
    for (i=0; i<n_dist; i++)
        diags[i] = DONE / diags[i];
#endif
    CreateDoubles (&aux, n); 

#if VECTOR_OUTPUT
    // write to file for testing purpose
    FILE *fp;
    if (myId == 0) {
        char name[50];
        sprintf(name, "%d.txt", nProcs);
        fp = fopen(name,"w");
    }
#endif

    iter = 0;

	MPI_Allgatherv (x, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
	InitDoubles (z, sizeR, DZERO, DZERO);
    ProdSparseMatrixVectorByRows (mat, 0, aux, z);            			// z = A * x
    dcopy (&n_dist, b, &IONE, res, &IONE);                          		// res = b
    daxpy (&n_dist, &DMONE, z, &IONE, res, &IONE);                      // res -= z
#if PRECOND
    VvecDoubles (DONE, diags, res, DZERO, y, n_dist);                    // y = D^-1 * res
#else
    y = res;
#endif
    dcopy (&n_dist, y, &IONE, d, &IONE);                                // d = y

    double vAux[2];
    std::vector<int64_t> h_superacc(2 * exblas::BIN_COUNT);
    std::vector<int64_t> h_superacc_tol(exblas::BIN_COUNT);

#if PRECOND
    // beta = res' * y 
    exblas::cpu::exdot (n_dist, res, y, &h_superacc[0]);
    // ReproAllReduce -- Begin
    int imin=exblas::IMIN, imax=exblas::IMAX;
    exblas::cpu::Normalize(&h_superacc[0], imin, imax);

    // compute tolerance
    //     tol = res' * res
    exblas::cpu::exdot (n_dist, res, res, &h_superacc_tol[0]);
    imin=exblas::IMIN, imax=exblas::IMAX;
    exblas::cpu::Normalize(&h_superacc_tol[0], imin, imax);

    // merge two superaccs into one for reduction
    for (int i = 0; i < exblas::BIN_COUNT; i++) {
        h_superacc[exblas::BIN_COUNT + i] = h_superacc_tol[i]; 
    } 

    if (myId == 0) {
        MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce (&h_superacc[0], NULL, 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (myId == 0) {
        // split them back
        for (int i = 0; i < exblas::BIN_COUNT; i++) {
            h_superacc_tol[i] = h_superacc[exblas::BIN_COUNT + i]; 
        } 
        vAux[0] = exblas::cpu::Round( &h_superacc[0] );
        vAux[1] = exblas::cpu::Round( &h_superacc_tol[0] );
    }
    MPI_Bcast(vAux, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    beta = vAux[0]; 
    tol  = vAux[1]; 
    // ReproAllReduce -- End
    
    tol = sqrt (tol);
#else
    // beta = res' * y
    exblas::cpu::exdot (n_dist, res, y, &h_superacc[0]);
    // ReproAllReduce -- Begin
    int imin=exblas::IMIN, imax=exblas::IMAX;
    exblas::cpu::Normalize(&h_superacc[0], imin, imax);
    if (myId == 0) {
        MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce (&h_superacc[0], NULL, exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (myId == 0) {
        beta = exblas::cpu::Round( &h_superacc[0] );
    }
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ReproAllReduce -- End

    tol = sqrt (beta);
#endif

#if DIRECT_ERROR
    // compute direct error
    double direct_err;
	dcopy (&n_dist, x_exact, &IONE, res_err, &IONE);                        // res_err = x_exact
	daxpy (&n_dist, &DMONE, x, &IONE, res_err, &IONE);                      // res_err -= x

    // compute inf norm
    direct_err = norm_inf(n_dist, res_err);
    MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//    // compute euclidean norm
//    // direct_err = res_err' * res_err
//    exblas::cpu::exdot (n_dist, res_err, res_err, &h_superacc[0]);
//    // ReproAllReduce -- Begin
//    imin=exblas::IMIN, imax=exblas::IMAX;
//    exblas::cpu::Normalize(&h_superacc[0], imin, imax);
//    if (myId == 0) {
//        MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
//    } else {
//        MPI_Reduce (&h_superacc[0], NULL, exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
//    }
//    if (myId == 0) {
//        direct_err = exblas::cpu::Round( &h_superacc[0] );
//    }
//    MPI_Bcast(&direct_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    // ReproAllReduce -- End
//
//    direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) 
        reloj (&t1, &t2);
    while ((iter < maxiter) && (tol > umbral)) {

        MPI_Allgatherv (d, sizeR, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
        InitDoubles (z, sizeR, DZERO, DZERO);
        ProdSparseMatrixVectorByRows (mat, 0, aux, z);            		// z = A * d

        if (myId == 0) 
#if DIRECT_ERROR
            printf ("%d \t %a \t %a \n", iter, tol, direct_err);
            //printf ("%d \t %20.10e \t %20.10e \n", iter, tol, direct_err);
#else        
            printf ("%d \t %20.10e \n", iter, tol);
#endif // DIRECT_ERROR

        // ReproAllReduce -- Begin
        exblas::cpu::exdot (n_dist, d, z, &h_superacc[0]);
        imin=exblas::IMIN, imax=exblas::IMAX;
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
        if (myId == 0) {
            MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce (&h_superacc[0], NULL, exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (myId == 0) {
            rho = exblas::cpu::Round( &h_superacc[0] );
        }
        MPI_Bcast(&rho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ReproAllReduce -- End

        rho = beta / rho;
        // TODO: replace with fma
        daxpy (&n_dist, &rho, d, &IONE, x, &IONE);                      	// x += rho * d;

        rho = -rho;
        // TODO: replace with fma
        daxpy (&n_dist, &rho, z, &IONE, res, &IONE);                      // res -= rho * z

#if PRECOND
        VvecDoubles (DONE, diags, res, DZERO, y, n_dist);                 // y = D^-1 * res
#else
        y = res;
#endif
        alpha = beta;                                                 		// alpha = beta

#if PRECOND
        // ReproAllReduce -- Begin
        // beta = res' * y 
        exblas::cpu::exdot (n_dist, res, y, &h_superacc[0]);
        imin=exblas::IMIN, imax=exblas::IMAX;
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);

        // compute tolerance
        //     tol = res' * res
        exblas::cpu::exdot (n_dist, res, res, &h_superacc_tol[0]);
        imin=exblas::IMIN, imax=exblas::IMAX;
        exblas::cpu::Normalize(&h_superacc_tol[0], imin, imax);

        // merge two superaccs into one for reduction
        for (int i = 0; i < exblas::BIN_COUNT; i++) {
            h_superacc[exblas::BIN_COUNT + i] = h_superacc_tol[i]; 
        } 

        if (myId == 0) {
            MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce (&h_superacc[0], NULL, 2*exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (myId == 0) {
            // split them back
            for (int i = 0; i < exblas::BIN_COUNT; i++) {
                h_superacc_tol[i] = h_superacc[exblas::BIN_COUNT + i]; 
            } 
            vAux[0] = exblas::cpu::Round( &h_superacc[0] );
            vAux[1] = exblas::cpu::Round( &h_superacc_tol[0] );
        }
        MPI_Bcast(vAux, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	    beta = vAux[0]; 
	    tol  = vAux[1]; 
        // ReproAllReduce -- End
    	
        tol = sqrt (tol);
#else
        // beta = res' * y 
        // ReproAllReduce -- Begin
        exblas::cpu::exdot (n_dist, res, y, &h_superacc[0]);
        imin=exblas::IMIN, imax=exblas::IMAX;
        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
        if (myId == 0) {
            MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        } else {
            MPI_Reduce (&h_superacc[0], NULL, exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        }
        if (myId == 0) {
            beta = exblas::cpu::Round( &h_superacc[0] );
        }
        MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // ReproAllReduce -- End

        // compute tolerance
        tol = sqrt (beta);
#endif

#if DIRECT_ERROR
        // compute direct error
        dcopy (&n_dist, x_exact, &IONE, res_err, &IONE);                        // res_err = x_exact
        daxpy (&n_dist, &DMONE, x, &IONE, res_err, &IONE);                      // res_err -= x

        // compute inf norm
        direct_err = norm_inf(n_dist, res_err);
        MPI_Allreduce(MPI_IN_PLACE, &direct_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//        // compute euclidean norm
//        // direct_err = res_err' * res_err
//        exblas::cpu::exdot (n_dist, res_err, res_err, &h_superacc[0]);
//        // ReproAllReduce -- Begin
//        imin=exblas::IMIN, imax=exblas::IMAX;
//        exblas::cpu::Normalize(&h_superacc[0], imin, imax);
//        if (myId == 0) {
//            MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
//        } else {
//            MPI_Reduce (&h_superacc[0], NULL, exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
//        }
//        if (myId == 0) {
//            direct_err = exblas::cpu::Round( &h_superacc[0] );
//        }
//        MPI_Bcast(&direct_err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//        // ReproAllReduce -- End
//
//        direct_err = sqrt(direct_err);
#endif // DIRECT_ERROR

        alpha = beta / alpha;                                         		// alpha = beta / alpha
        dscal (&n_dist, &alpha, d, &IONE);                                // d = alpha * d
        daxpy (&n_dist, &DONE, y, &IONE, d, &IONE);                       // d += y

        iter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (myId == 0) 
        reloj (&t3, &t4);

#if VECTOR_OUTPUT
    // print aux
    MPI_Allgatherv (x, n_dist, MPI_DOUBLE, aux, sizes, dspls, MPI_DOUBLE, MPI_COMM_WORLD);
    if (myId == 0) {
        fprintf(fp, "%d ", iter);
        for (int ip = 0; ip < n; ip++)
            fprintf(fp, "%20.10e ", aux[ip]);
        fprintf(fp, "\n");

        fclose(fp);
    }
#endif

    if (myId == 0) {
        printf ("Size: %d \n", n);
        printf ("Iter: %d \n", iter);
        printf ("Tol: %a \n", tol);
        //printf ("Tol: %20.10e \n", tol);
        printf ("Time_loop: %20.10e\n", (t3-t1));
        printf ("Time_iter: %20.10e\n", (t3-t1)/iter);
    }

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
    int dimL, dspL, *vdimL = NULL, *vdspL = NULL;
    SparseMatrix matL = {0, 0, NULL, NULL, NULL};
    double *vecL = NULL, *sol1L = NULL, *sol2L = NULL, *rbuf = NULL;
    int mat_from_file, nodes, size_param, stencil_points;
    if (argc == 3) {
        mat_from_file = atoi(argv[2]);
    } else {
        mat_from_file = atoi(argv[2]);
        nodes = atoi(argv[3]);
        size_param = atoi(argv[4]);
        stencil_points = atoi(argv[5]);
    }

    /***************************************/

    MPI_Init (&argc, &argv);

    // Definition of the variables nProcs and myId
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs); MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    root = nProcs-1;
    root = 0;

    /***************************************/

    CreateInts (&vdimL, nProcs); CreateInts (&vdspL, nProcs); 
    if(mat_from_file) {
        if (myId == root) {
            // Creating the matrix
            ReadMatrixHB (argv[1], &sym);
            DesymmetrizeSparseMatrices (sym, 0, &mat, 0);
            dim = mat.dim1;
        }

        // Distributing the matrix
        dim = DistributeMatrix (mat, index, &matL, indexL, vdimL, vdspL, root, MPI_COMM_WORLD);
        dimL = vdimL[myId]; dspL = vdspL[myId];
    }
    else {
        dim = size_param * size_param * size_param;
        int divL, rstL, i;
        divL = (dim / nProcs); rstL = (dim % nProcs);
        for (i=0; i<nProcs; i++) vdimL[i] = divL + (i < rstL);
        vdspL[0] = 0; for (i=1; i<nProcs; i++) vdspL[i] = vdspL[i-1] + vdimL[i-1];
        dimL = vdimL[myId]; dspL = vdspL[myId];
        int band_width = size_param * (size_param + 1) + 1;
        band_width = 100 * nodes;
        long nnz_here = ((long) (stencil_points + 2 * band_width)) * dimL;
        printf ("dimL: %d, nodes: %d, size_param: %d, band_width: %d, stencil_points: %d, nnz_here: %ld\n",
                dimL, nodes, size_param, band_width, stencil_points, nnz_here);
        allocate_matrix(dimL, dim, nnz_here, &matL);
        generate_Poisson3D_filled(&matL, size_param, stencil_points, band_width, dspL, dimL, dim);

        // To generate ill-conditioned matrices
        double factor = 1.0e6;
        ScaleFirstRowCol(matL, dspL, dimL, myId, root, factor);
    }
    MPI_Barrier(MPI_COMM_WORLD);

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

    int i;
    double beta;
    if (myId == root) {
        InitDoubles (vec, dim, 1.0, 0.0);
        InitDoubles (sol1, dim, 0.0, 0.0);
        InitDoubles (sol2, dim, 0.0, 0.0);
        //  ProdSparseMatrixVectorByRows (mat, 0, vec, sol1);
    }
    int k=0;
    int *vptrM = matL.vptr;
    for (int i=0; i < matL.dim1; i++) {
        for(int j=vptrM[i]; j<vptrM[i+1]; j++) {
            sol1L[k] += matL.vval[j];
        }
        k++;
    }

    MPI_Scatterv (sol2, vdimL, vdspL, MPI_DOUBLE, sol2L, dimL, MPI_DOUBLE, root, MPI_COMM_WORLD);

    ConjugateGradient (matL, sol2L, sol1L, vdimL, vdspL, rbuf, myId);

    // Error computation
    for (i=0; i<dimL; i++) sol2L[i] -= 1.0;

    // ReproAllReduce -- Begin
    std::vector<int64_t> h_superacc(exblas::BIN_COUNT);
    exblas::cpu::exdot (dimL, sol2L, sol2L, &h_superacc[0]);
    int imin=exblas::IMIN, imax=exblas::IMAX;
    exblas::cpu::Normalize(&h_superacc[0], imin, imax);
    if (myId == 0) {
        MPI_Reduce (MPI_IN_PLACE, &h_superacc[0], exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce (&h_superacc[0], NULL, exblas::BIN_COUNT, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (myId == 0) {
        beta = exblas::cpu::Round( &h_superacc[0] );
    }
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // ReproAllReduce -- End

    beta = sqrt(beta);
    if (myId == 0) 
        printf ("Error: %a\n", beta);
        //printf ("Error: %10.5e\n", beta);

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

