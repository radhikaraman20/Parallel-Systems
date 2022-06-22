// Group info:
// psomash Prakruthi Somashekarappa
// rbraman Radhika B Raman
// srames22 Srivatsan Ramesh
        #include <stdio.h>
        #include <stdlib.h>
        #include <math.h>
        #include <mpi.h>

        #define   XI              1.0
        #define   XF              100.0
        #define   tag               123        
        #define   tag_new           456    

        double     fn(double);
        void        print_function_data(int, double*, double*, double*);
        int         main(int, char**);

        int main (int argc, char *argv[])
        {       
                int size, rank;
                MPI_Status status;
                MPI_Status stats[4];
                MPI_Request reqs[4];
                MPI_Request request[4];
                MPI_Init(&argc, &argv);
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &size);

                double start_time, end_time;
                int NGRID, p2p, gather_type;
                if(argc > 1)
                {   
                    // Collecting command line arguments     
                    NGRID = atoi(argv[1]);
                    p2p = atoi(argv[2]);
                    gather_type = atoi(argv[3]);

                    // Handling cases when communication type and gather type values are not 0 or 1
                    if((p2p >= 2) || (gather_type >= 2))
                    {
                        printf("Please specify point to point communication/gather type value as 0 or 1.\n");
                        exit(0);  
                    }
                }
                else 
                {
                        printf("Please specify the number of grid points.\n");
                        exit(0);
                }
                
                int i, j;
                double *xc = (double *)malloc(sizeof(double)* (NGRID + 2));
                double *dyc = (double*)malloc((NGRID) * sizeof(double));
                double  dx;

                // Arrays local to processes in order to store calculated yc and dyc values
                double  *sub_yc, *sub_dyc;
                int    imin, imax;  
                imin = 1;
                imax = NGRID;

                // Calculating the xc values
                for (i = 1; i <= NGRID ; i++)
                {
                        xc[i] = XI + (XF - XI) * (double)(i - 1)/(double)(NGRID - 1);
                }
                
                // Calculating the step size
                dx = xc[2] - xc[1];
                xc[0] = xc[1] - dx;
                xc[NGRID + 1] = xc[NGRID] + dx;

                
                int count = NGRID/size; // Calculating number of data points to be split between the processes
                int src;
                int sum = 0;
                int diff = NGRID % size;        // To find the number of data points remaining that have not been 
                int temp_diff = NGRID % size;   // equally split to processes if NGRID is not evenly divisible by number of processes
                int count_array[size];  
                int disp_array[size];   
                
                // To store the number of data pts. for each process for even and uneven split of data pts. b/w processes
                for(i = 0; i<size; i++)
                {
                        count_array[i] = count;
                        if(temp_diff > 0)
                        {
                                count_array[i] = count_array[i] + 1;
                                temp_diff--;
                        }
                        disp_array[i] = sum;        // To store the location in terms of number of elements of the from 
                        sum = sum + count_array[i]; // where data must be taken from xc and put the final array
                }
                int cnt = count_array[rank];
                int cnt1 = count_array[rank];

                // Array to store received data chunks for each process
                double array_split[count + diff];
                double *new_xc =(double*) malloc((NGRID) * sizeof(double));

                for(i = 0; i < NGRID; i++)
                {
                        new_xc[i] = xc[i+1];
                }

                start_time = MPI_Wtime();

                // Scattering data chunks amongst all processors irrespective of whether NGRID is divisible by num. of processes
                MPI_Scatterv(new_xc, count_array, disp_array, MPI_DOUBLE, array_split, count + diff, MPI_DOUBLE, 0, MPI_COMM_WORLD);

                sub_yc  =   (double*) malloc((count + diff) * sizeof(double));
                sub_dyc =   (double*) malloc((count + diff) * sizeof(double));
                double *yc = (double *)malloc(sizeof(double)* (NGRID));
                
                // Variables to store right and left boundary values
                double temp_r, temp_l; 

                // Calculation of yc array i.e applying function for each data pt. at each processor for its respective data points
                for(i = 0; i < count_array[rank]; i++)
                {
                        sub_yc[i] = fn(array_split[i]);
                }

                // Checking if communication type is BLOCKING
                if(p2p == 0)
                {

                // All processors except the first processor i.e Process ID: 0
                if(rank != 0)
                {       
                        // Perform BLOCKING Send and Recv to Communicate Boundary values
                        MPI_Send(&sub_yc[0], 1, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD );
                        MPI_Recv(&temp_l, 1, MPI_DOUBLE, rank - 1, tag + 1, MPI_COMM_WORLD, &status);
                        
                }
                
                // All processors except the last processor
                if(rank != size -1)
                {       
                        // Perform BLOCKING recv and send to communicate boundary values
                        MPI_Recv(&temp_r, 1, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD, &status);
                        MPI_Send(&sub_yc[count_array[rank] -1], 1, MPI_DOUBLE, rank + 1, tag + 1, MPI_COMM_WORLD );           
                }
                
                // Calculation of dyc i.e derivatives at all processors except first and last
                if(rank != 0 && rank != size -1)
                {
                        for (i = 0; i < count_array[rank]; i++)
                        {       if(i == 0)
                                {
                                        sub_dyc[i] = (sub_yc[i + 1] - temp_l)/(2.0 * dx);
                                }
                                else if(i == count_array[rank] -1)
                                {
                                       sub_dyc[i] = (temp_r - sub_yc[i-1])/(2.0 * dx);   
                                }
                                else
                                {
                                        sub_dyc[i] = (sub_yc[i + 1] - sub_yc[i - 1])/(2.0 * dx);
                                }
                        }       

                }
                else
                {       // Calculation of derivatives at first processor
                        if(rank == 0)
                        {
                                for (i = 0; i < count_array[rank]; i++)
                                {
                                        if(i == 0)
                                        {
                                                sub_dyc[i] = (sub_yc[i + 1] - fn(xc[0]))/(2.0 * dx);  
                                        }
                                        else if(i == count_array[rank] -1)
                                        {
                                                sub_dyc[i] = (temp_r - sub_yc[i-1])/(2.0 * dx);   
                                        }
                                        else
                                        {
                                                sub_dyc[i] = (sub_yc[i + 1] - sub_yc[i - 1])/(2.0 * dx);

                                        }

                                }
                              
                        }
                        else
                        {
                                for (i = 0; i < count_array[rank]; i++)
                                {
                                        if(i == count_array[rank] - 1)
                                        {
                                                sub_dyc[i] = (fn(xc[NGRID + 1]) - sub_yc[i - 1])/(2.0 * dx);  
                                        }
                                        else if(i == 0)
                                        {
                                                sub_dyc[i] = (sub_yc[i+1] - temp_l )/(2.0 * dx);   
                                        }
                                        else
                                        {
                                                sub_dyc[i] = (sub_yc[i + 1] - sub_yc[i - 1])/(2.0 * dx);

                                        }

                                }

                        }

                }

                }
                else    // If communication type is NON_BLOCKING 
                {       // All processors expect the first one
                        if(rank != 0)
                        {
                                // Perform non-blocking recv and send to communicate boundary data pts.
                                MPI_Irecv(&temp_l, 1, MPI_DOUBLE, rank - 1, tag + 1, MPI_COMM_WORLD, &reqs[3]);
                                MPI_Isend(&sub_yc[0], 1, MPI_DOUBLE, rank - 1, tag, MPI_COMM_WORLD, &reqs[0] );
                        }

                        // All processors except the last one
                        if(rank != size -1)
                        {       
                                // Perform non-blocking recv and send to communicate boundary data pts.
                                MPI_Irecv(&temp_r, 1, MPI_DOUBLE, rank + 1, tag, MPI_COMM_WORLD, &reqs[1]);
                                MPI_Isend(&sub_yc[count_array[rank]-1], 1, MPI_DOUBLE, rank + 1, tag + 1, MPI_COMM_WORLD, &reqs[2] );
                        
                        }

                        // To make sure all processors rach this part of the code 
                        MPI_Barrier(MPI_COMM_WORLD);

                        // Calculating derivatives values at all processors except first and last
                        if(rank != 0 && rank != size -1)
                        {
                                for (i = 0; i < count_array[rank]; i++)
                                {       
                                        if(i == 0)
                                        {
                                                sub_dyc[i] = (sub_yc[i + 1] - temp_l)/(2.0 * dx);
                                        }
                                        else if(i == count_array[rank] -1)
                                        {
                                                sub_dyc[i] = (temp_r - sub_yc[i-1])/(2.0 * dx);   
                                        }
                                        else
                                        {
                                                sub_dyc[i] = (sub_yc[i + 1] - sub_yc[i - 1])/(2.0 * dx);
                                        }
                                }       

                        }
                        else
                        {
                                if(rank == 0)
                                {
                                        for (i = 0; i < count_array[rank]; i++)
                                        {
                                                if(i == 0)
                                                {
                                                        sub_dyc[i] = (sub_yc[i + 1] - fn(xc[0]))/(2.0 * dx);  
                                                }
                                                else if(i == count_array[rank] -1)
                                                {
                                                        sub_dyc[i] = (temp_r - sub_yc[i-1])/(2.0 * dx);   
                                                }
                                                else
                                                {
                                                        sub_dyc[i] = (sub_yc[i + 1] - sub_yc[i - 1])/(2.0 * dx);

                                                }

                                        }       
                              
                                }
                                else
                                {
                                        for (i = 0; i < count_array[rank]; i++)
                                        {       
                                                if(i == count_array[rank] - 1)
                                                {
                                                        sub_dyc[i] = (fn(xc[NGRID + 1]) - sub_yc[i - 1])/(2.0 * dx);  
                                                }       
                                                else if(i == 0)
                                                {
                                                        sub_dyc[i] = (sub_yc[i+1] - temp_l )/(2.0 * dx);   
                                                }
                                                else
                                                {
                                                        sub_dyc[i] = (sub_yc[i + 1] - sub_yc[i - 1])/(2.0 * dx);

                                                }

                                        }

                                }

                        }

                }

                // To check if Gather type is MPI_Gather
                if(gather_type == 0)
                {
                        MPI_Gatherv(sub_yc, count_array[rank], MPI_DOUBLE, yc, count_array, disp_array, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                        MPI_Gatherv(sub_dyc, count_array[rank], MPI_DOUBLE, dyc, count_array, disp_array, MPI_DOUBLE, 0, MPI_COMM_WORLD);


                        
                }
                // To check if Gather type is Manual Gather
                else if(gather_type == 1)
                {       // Gather Type: Manual | Message Passing type: BLOCKING
                        if(p2p == 0)
                        { // Making all processors send yc values to first processor

                       for(src = 1; src < size ; src++)
                       {
                                if(src == rank)
                                {
                                        MPI_Send(sub_yc, count_array[rank], MPI_DOUBLE, 0, tag_new + 1 , MPI_COMM_WORLD);                                        
                                }
                                if(rank == 0)
                                {
                                        MPI_Recv(&yc[cnt], count_array[src], MPI_DOUBLE, src, tag_new + 1, MPI_COMM_WORLD, &status);   
                                        cnt = cnt + count_array[src];
                                }        
                                
                       }
                       if(rank == 0)
                       {
                                for(i = 0; i < count_array[rank] ; i++)
                                {
                                        yc[i] = sub_yc[i];
                                }
                       } 

                       // Making all processors send yc values to first processor
                       for(src = 1; src < size ; src++)
                       {
                                if(src == rank)
                                {
                                        MPI_Send(sub_dyc, count_array[rank], MPI_DOUBLE, 0, tag + 1 , MPI_COMM_WORLD);        
                                }
                                if(rank == 0)
                                {
                                        MPI_Recv(&dyc[cnt1], count_array[src], MPI_DOUBLE, src, tag + 1, MPI_COMM_WORLD, &status);   
                                        cnt1 = cnt1 + count_array[src];
                                }      
                                
                       }
                       if(rank == 0)
                       {
                                for(i = 0; i < count_array[rank] ; i++)
                                {
                                        dyc[i] = sub_dyc[i];
                                }
                       }
                       }
                       // Gather Type: Manual | Message Passing type: NON-BLOCKING
                       else if(p2p == 1)
                       {          
                                for(src = 1; src < size ; src++)
                       {
                                if(src == rank)
                                {
                                        MPI_Isend(sub_yc, count_array[rank], MPI_DOUBLE, 0, tag_new + 1 , MPI_COMM_WORLD, &request[0]);
                                }
                                if(rank == 0)
                                {               
                                        MPI_Irecv(&yc[cnt], count_array[src], MPI_DOUBLE, src, tag_new + 1, MPI_COMM_WORLD, &request[1]);   
                                        cnt = cnt + count_array[src];
                                }
                                
                                MPI_Barrier(MPI_COMM_WORLD);        
                       }
                       if(rank == 0)
                       {
                                for(i = 0; i < count_array[rank] ; i++)
                                {
                                        yc[i] = sub_yc[i];
                                }
                       } 

                       for(src = 1; src < size ; src++)
                       {
                                if(src == rank)
                                {
                                        MPI_Isend(sub_dyc, count_array[rank], MPI_DOUBLE, 0, tag + 1 , MPI_COMM_WORLD, &request[2]);                                        

                                }
                                if(rank == 0)
                                {
                                        MPI_Irecv(&dyc[cnt1], count_array[src], MPI_DOUBLE, src, tag + 1, MPI_COMM_WORLD, &request[3]);   
                                        cnt1 = cnt1 + count_array[src];
                                }
                                MPI_Barrier(MPI_COMM_WORLD);    
                                
                       }
                       if(rank == 0)
                       {
                                for(i = 0; i < count_array[rank] ; i++)
                                {
                                        dyc[i] = sub_dyc[i];
                                }
                       }
                       
                       }
                }

                end_time = MPI_Wtime();        

                // Making First process write values to .dat files after computations and gathering
                if(rank == 0)
                {
                        print_function_data(NGRID, &xc[1], &yc[0], &dyc[0]);
                }

                free(yc);
                free(dyc);

                MPI_Finalize();
                return 0;
        }


        void print_function_data(int np, double *x, double *y, double *dydx)
        {
                int   i;

                char filename[1024];
                sprintf(filename, "fn-%d.dat", np);

                FILE *fp = fopen(filename, "w");

                for(i = 0; i < np; i++)
                {
                        fprintf(fp, "%f %f %f\n", x[i], y[i], dydx[i]);
                }

                fclose(fp);
        }
