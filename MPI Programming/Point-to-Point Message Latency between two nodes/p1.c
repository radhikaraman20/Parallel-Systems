//Single Author info:
//rbraman Radhika B Raman
#include "mpi.h"
#include <stdio.h>
#include <sys/time.h> //to use gettimeofday
#include <math.h> //to make use of pow and sqrt functions for the calculation of stddev

int main(int argc, char *argv[])
{
        MPI_Init(&argc, &argv);

        //declare variables for process ID, number of processes and time difference in communication between a pair of nodes
        int p_id, num_of_proc, comm_time_diff=0;

        //variables to record start time before MPI_Send and end time after MPI_Recv for a pair of nodes
        struct timeval comm_start_time;
        struct timeval comm_end_time;

        MPI_Status status;

        MPI_Comm_rank(MPI_COMM_WORLD, &p_id);
        MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);

        //storing 7 message sizes in terms of number of integer elements for each size e.g (32KB * 1024)/4 = 8192
        long int msg_size[7] = {8192, 16384, 32768, 65536, 131072, 262144, 524288};


        int i,j,k;

        //calculating point-to-point latency for each message size
        for(j=0; j<7; j++)
        {
                //sending and receiving buffers
                int send_msg_array[msg_size[j]];
                int recv_msg_array[msg_size[j]];

                float time_diff_array[10]={0,0,0,0,0,0,0,0,0,0};

                float stddev_sum=0, mean_val=0, sum_of_times=0, stddev=0;
                comm_time_diff=0;        

                //iterating through each message size 10 times 
                for (i=0; i<10; i++)
                {

                        //perform MPI_Send and MPI_Recv between pairs of processes 
                        //Processes 0 and 1 have been chosen as the pair for which latency will be recorded 
                        //check if it is the first process and first iteration/message exchange, if yes skip calculation of RTT
                        if(p_id == 0 && i == 0){
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 1, 101, MPI_COMM_WORLD);
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 1, 102, MPI_COMM_WORLD, &status);
                        }

                        //check if it is the first process but not first iteration/message exchange, if yes perform arithmetic for RTT
                        else if(p_id == 0){
                                gettimeofday(&comm_start_time, NULL);
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 1, 101, MPI_COMM_WORLD);
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 1, 102, MPI_COMM_WORLD, &status);
                                gettimeofday(&comm_end_time, NULL);
                                time_diff_array[i] = comm_end_time.tv_usec - comm_start_time.tv_usec;
                                comm_time_diff += comm_end_time.tv_usec - comm_start_time.tv_usec;

                        }

                        //process 1 receiving from 0 and sending back to 0
                        else if(p_id == 1){
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 0, 101, MPI_COMM_WORLD, &status);
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 0, 102, MPI_COMM_WORLD);

                        }

                        //process 2 sending to 3 and recieving from 3
                        else if(p_id == 2){
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 3, 103, MPI_COMM_WORLD);
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 3, 104, MPI_COMM_WORLD, &status);

                        }

                        //process 3 receiving from 2 and sending back to 2
                        else if(p_id == 3){
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 2, 103, MPI_COMM_WORLD, &status);
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 2, 104, MPI_COMM_WORLD);

                        }

                        //process 4 sending to 5 and recieving from 5
                        else if(p_id == 4){
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 5, 105, MPI_COMM_WORLD);
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 5, 106, MPI_COMM_WORLD, &status);
                        }

                        //process 5 receiving from 4 and sending back to 4
                        else if(p_id == 5){
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 4, 105, MPI_COMM_WORLD, &status);
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 4, 106, MPI_COMM_WORLD);

                        }

                        //process 6 sending to 7 and recieving from 7
                        else if(p_id == 6){
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 7, 106, MPI_COMM_WORLD);
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 7, 107, MPI_COMM_WORLD, &status);

                        }

                        //process 7 receiving from 6 and sending back to 6
                        else if(p_id == 7){
                                MPI_Recv(recv_msg_array, msg_size[j], MPI_INT, 6, 106, MPI_COMM_WORLD, &status);
                                MPI_Send(send_msg_array, msg_size[j], MPI_INT, 6, 107, MPI_COMM_WORLD);

                        }       


                }

                //calculate mean as total time difference by 9
                mean_val = comm_time_diff/9;

                //summation for standard deviation
                for(k=1; k<10; k++)
                {
                        stddev_sum += pow(time_diff_array[k] - mean_val,2);
                }

                //final standard deviation
                stddev = sqrt(stddev_sum/9);

                //final required output given by process 0, in order to avoid multiple outputs from all processes
                if(p_id == 0){
			//Printing message size in KB, avg RTT in micro seconds, and standard deviation
                        printf("%d %d %f\n", (msg_size[j]*4)/1024,comm_time_diff/9, stddev);
                }       

        }



        MPI_Finalize();
        return 0;
}
