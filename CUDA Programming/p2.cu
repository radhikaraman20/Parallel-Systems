//Single Author info:
//rbraman Radhika B Raman

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

/* first grid point */
#define   XI              0.0
/* last grid point */
#define   XF              M_PI

#define NUMBER_OF_THREADS  512


/* function declarations */
//double     fn(double);
void        print_function_data(int, double*, double*, double*);
int         main(int, char**);

// Device function that generates the cos values for each grid point 
__device__ double cosine_comp( double  val )         
{
        return cos(val);
}
// Kernel function 
__global__ void func(int ngrid, double *dev_x, double *dev_y, double *dev_inf_arr, double *dev_area, double h)
{       
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // To globally id the thread
        if (idx < ngrid)    // Restricting out of bound values
        {
                dev_y[idx] = cosine_comp(dev_x[idx]);         
        }
        __syncthreads(); // barrier to all threads                     
        
        if(idx < ngrid && idx!=0) // Except 0, applying formula               
        {
                dev_area[idx] = (dev_y[idx] + dev_y[idx-1])/2*h;   
        }
        
        else if(idx == 0 ) 
        {       
                dev_area[0] = (dev_y[idx] + cosine_comp(0))/2*h;
        }
        __syncthreads(); 

        if(idx < ngrid && idx!=0) 
        {       
                dev_inf_arr[idx] = dev_area[idx];         
        }
        else if(idx == 0)
        {
                dev_inf_arr[0] = dev_area[0];
        }
        //__syncthreads();
        if(idx < ngrid && idx!=0) 
        {       int k;
                for(k=idx;k>0;k--) // Cumulative sum
                {
                        dev_inf_arr[idx] = dev_area[k-1] + dev_inf_arr[idx]; 
                } 
        }
        __syncthreads();
}

int main (int argc, char *argv[])
{
        int NGRID;
        if(argc > 1)
            NGRID = atoi(argv[1]);
        else 
        {
                printf("Please specify the number of grid points.\n");
                exit(0);
        }
        //loop index
        int     i;
        int number_of_blocks, temp_var;
        double  h, *host_xc;

        double *device_xc, *device_yc, *device_inf, *device_area;

        double *inf = (double *)malloc(sizeof(double) * (NGRID) );
        double  *xc = (double *)malloc(sizeof(double)* (NGRID + 1));
        double  *yc = (double*)malloc(sizeof(double) * (NGRID));

        // Eliminating 0th index and storing in host xc
        host_xc = (double*) malloc((NGRID) * sizeof(double));

        // device variable's memory locations are being allocated
        cudaMalloc(&device_xc, sizeof(double)*NGRID);
        cudaMalloc(&device_yc, sizeof(double)*NGRID);
        cudaMalloc(&device_inf, sizeof(double)*NGRID);
        cudaMalloc(&device_area, sizeof(double)*NGRID);

        //construct grid of x axis
        for (i = 1; i <= NGRID ; i++)
        {
                xc[i] = XI + (XF - XI) * (double)(i - 1)/(double)(NGRID - 1);
        }

        // copying all elements from xc to new array except the first element
        for(i = 0; i < NGRID; i++)
        {
                host_xc[i] = xc[i+1];
        }

        // Copying memory of host array to device array
        cudaMemcpy(device_xc, host_xc, sizeof(double)*NGRID, cudaMemcpyHostToDevice); // To copy memory of alloted space into device global memory 

        h = (XF - XI) / (NGRID - 1);

        // kernel config
        if(NGRID % NUMBER_OF_THREADS == 0)
        {
                temp_var = 0;
        }
        else
        {
                temp_var = 1;       
        }

        // To ensure enough number of blocks with enough threads for given NGRID
        number_of_blocks = NGRID/NUMBER_OF_THREADS + temp_var;

        // Calling the kernel i.e GPU of the processor
        func<<<number_of_blocks, NUMBER_OF_THREADS>>>(NGRID, device_xc, device_yc, device_inf, device_area, h);      
        
        // Retrieving data back to host 
        cudaMemcpy(yc, device_yc, sizeof(double)*NGRID, cudaMemcpyDeviceToHost); 
        cudaMemcpy(inf, device_inf, sizeof(double)*NGRID, cudaMemcpyDeviceToHost);
        
        print_function_data(NGRID, &xc[1], &yc[0], &inf[0]);

        //free allocated memory 
        cudaFree(device_xc);
        cudaFree(device_yc);
        cudaFree(device_inf);
        free(xc);
        free(yc);
        free(inf);

        return 0;
}

//prints out the function and its derivative to a file
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
