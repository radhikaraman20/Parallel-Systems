//Single Author info:
//rbraman Radhika B Raman

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<dirent.h>
#include<math.h>
#include "mpi.h"
#include<stddef.h> //needed for offset

#define MAX_WORDS_IN_CORPUS 32
#define MAX_FILEPATH_LENGTH 16
#define MAX_WORD_LENGTH 16
#define MAX_DOCUMENT_NAME_LENGTH 8
#define MAX_STRING_LENGTH 64

typedef char word_document_str[MAX_STRING_LENGTH];

typedef struct o {
	char word[32];
	char document[8];
	int wordCount;
	int docSize;
	int numDocs;
	int numDocsWithWord;
	double tficf_val;
} obj;

typedef struct w {
	char word[32];
	int numDocsWithWord;
	int currDoc;
} u_w;

static int myCompare (const void * a, const void * b)
{
    return strcmp (a, b);
}

int main(int argc , char *argv[]){
	
	int proc_rank, num_of_proc;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	DIR* files;
	struct dirent* file;
	int i,j;
	int numDocs = 0, docSize, contains;
	char filename[MAX_FILEPATH_LENGTH], word[MAX_WORD_LENGTH], document[MAX_DOCUMENT_NAME_LENGTH];
	
	// Will hold all TFICF objects for all documents
	obj TFICF[MAX_WORDS_IN_CORPUS];
	int TF_idx = 0;
	
	// Will hold all unique words in the corpus and the number of documents with that word
	u_w unique_words[MAX_WORDS_IN_CORPUS];
	int uw_idx = 0;
	
	// Will hold the final strings that will be printed out
	word_document_str strings[MAX_WORDS_IN_CORPUS];
	
	//count number of documents at root node
	if(proc_rank == 0)
	{
		//Count numDocs
		if((files = opendir("input")) == NULL){
			printf("Directory failed to open\n");
			exit(1);
		}
		while((file = readdir(files))!= NULL){
			// On linux/Unix we don't want current and parent directories
			if(!strcmp(file->d_name, "."))	 continue;
			if(!strcmp(file->d_name, "..")) continue;
			numDocs++;
		}
		// Sending numDocs from root node to worker nodes
		for(int worker_rank=1; worker_rank<num_of_proc; worker_rank++)
		{
			MPI_Send(&numDocs, 1, MPI_INT, worker_rank, worker_rank, MPI_COMM_WORLD);
		}
		
	}
	//worker nodes get numDocs value from root node
	else
	{
		MPI_Recv(&numDocs, 1, MPI_INT, 0, proc_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	//wait to ensure that all processes have numDocs value from root node
	MPI_Barrier(MPI_COMM_WORLD);

	if(proc_rank!=0)
	{
	// Loop through each document and gather TFICF variables for each word
		for(i=proc_rank; i<=numDocs; i = i + (num_of_proc - 1))
		{
			sprintf(document, "doc%d", i);
			sprintf(filename,"input/%s",document);
			FILE* fp = fopen(filename, "r");
			if(fp == NULL){
				printf("Error Opening File: %s\n", filename);
				exit(0);
			}	
		
			// Get the document size
			docSize = 0;
			while((fscanf(fp,"%s",word))!= EOF)
				docSize++;
			
			// For each word in the document
			fseek(fp, 0, SEEK_SET);
			while((fscanf(fp,"%s",word))!= EOF){
				contains = 0;
				
				// If TFICF array already contains the word@document, just increment wordCount and break
				for(j=0; j<TF_idx; j++) {
					if(!strcmp(TFICF[j].word, word) && !strcmp(TFICF[j].document, document)){
						contains = 1;
						TFICF[j].wordCount++;
						break;
					}
				}
				
				//If TFICF array does not contain it, make a new one with wordCount=1
				if(!contains) {
					strcpy(TFICF[TF_idx].word, word);
					strcpy(TFICF[TF_idx].document, document);
					TFICF[TF_idx].wordCount = 1;
					TFICF[TF_idx].docSize = docSize;
					TFICF[TF_idx].numDocs = numDocs;
					TF_idx++;
				}
				
			contains = 0;
			// If unique_words array already contains the word, just increment numDocsWithWord
			for(j=0; j<uw_idx; j++) {
				if(!strcmp(unique_words[j].word, word)){
					contains = 1;
					if(unique_words[j].currDoc != i) {
						unique_words[j].numDocsWithWord++;
						unique_words[j].currDoc = i;
					}
					break;
				}
			}
				
			// If unique_words array does not contain it, make a new one with numDocsWithWord=1 
			if(!contains) {
				strcpy(unique_words[uw_idx].word, word);
				unique_words[uw_idx].numDocsWithWord = 1;
				unique_words[uw_idx].currDoc = i;
				uw_idx++;
			}
		}
		fclose(fp);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Print TF job similar to HW4/HW5 (For debugging purposes)
	printf("-------------TF Job-------------\n");
	for(j=0; j<TF_idx; j++)
		printf("%s@%s\t%d/%d\n", TFICF[j].word, TFICF[j].document, TFICF[j].wordCount, TFICF[j].docSize);
	
	//defining an array to store unique word counts that will be received from all other worker nodes
	int *workers_uw_count_recv;
	workers_uw_count_recv = (int *) malloc((num_of_proc) * sizeof(int));

	//for MPI_Isend and MPI_Irecv
    MPI_Request recv_req[num_of_proc-2], send_req[num_of_proc-2];
	//send_req = (MPI_Request *)malloc(sizeof(MPI_Request)*(num_of_proc-2));
	//recv_req = (MPI_Request *)malloc(sizeof(MPI_Request)*(num_of_proc-2));
	int req_count = 0;
	int req_tag = 123;
	
	if(proc_rank !=0)
	{
	for(int n = 1; n < num_of_proc; n++)
	{
		if(proc_rank != n)
		{
			//send your unique word count to remaining worker nodes, also receive unique word counts from them
			MPI_Irecv(&workers_uw_count_recv[n], 1, MPI_INT, n, req_tag, MPI_COMM_WORLD, &recv_req[req_count]);
			MPI_Isend(&uw_idx, 1, MPI_INT, n, req_tag, MPI_COMM_WORLD, &send_req[req_count]);
			req_count+=1;
		}
	}

	//wait for all non-blocking calls to complete before proceeding, acts like a barrier
	MPI_Waitall(num_of_proc-2, recv_req, MPI_STATUS_IGNORE);
	}


	//now, calculate total number of unique words in all other worker nodes, by combining above received values 
	int sum_of_uwcount_from_workers = 0;
	if(proc_rank != 0)
	{
	for(int n = 1; n < num_of_proc; n++)
	{
		if(proc_rank != n)
		{
			sum_of_uwcount_from_workers += workers_uw_count_recv[n];
		}
	}
	}

	//setting up new mpi_datatype/struct to save unique words from other worker nodes
	int num_var = 3;
	int blocklengths[num_var];
	blocklengths[0]=32;
	blocklengths[1]=1;
	blocklengths[2]=1;
    MPI_Datatype types[num_var];
	types[0]=MPI_CHAR;
	types[1]=MPI_INT;
	types[2]=MPI_INT;
    MPI_Datatype my_uw_type;
    MPI_Aint offsets[num_var];

    offsets[0] = offsetof(u_w, word);
    offsets[1] = offsetof(u_w, numDocsWithWord);
	offsets[2] = offsetof(u_w, currDoc);

    MPI_Type_create_struct(num_var, blocklengths, offsets, types, &my_uw_type);
    MPI_Type_commit(&my_uw_type);

	u_w *uw_details_from_workers;
	uw_details_from_workers = (u_w *) malloc(sum_of_uwcount_from_workers * sizeof(u_w));
	int traversed_word_count = 0;

	MPI_Request recv_req_new[num_of_proc-2], send_req_new[num_of_proc-2];
	//send_req_new = (MPI_Request *)malloc(sizeof(MPI_Request)*(num_of_proc-2));
	//recv_req_new = (MPI_Request *)malloc(sizeof(MPI_Request)*(num_of_proc-2));
	int send_count = 0, recv_count_new = 0;
	int req_tag_new = 234;

	//receive unique word details from other worker nodes
	if(proc_rank !=0)
	{
	for(int n = 1; n < num_of_proc; n++)
    {
		if(proc_rank!=n) 
		{
			for(int m = 0; m < uw_idx; m++)
        	{
				MPI_Isend(&unique_words[m], 1, my_uw_type, n, req_tag_new, MPI_COMM_WORLD, &send_req_new[send_count]);
          		send_count += 1;
			}

			for(int k = 0; k < workers_uw_count_recv[n]; k++) 
			{
				MPI_Irecv(&uw_details_from_workers[traversed_word_count], 1, my_uw_type, n, req_tag_new, MPI_COMM_WORLD, &recv_req_new[recv_count_new]);
				recv_count_new += 1;
				traversed_word_count += 1;
			}
			
		}
	}

	//wait for all non-blocking calls to complete before proceeding, acts like a barrier
	MPI_Waitall(sum_of_uwcount_from_workers, recv_req_new, MPI_STATUS_IGNORE);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if(proc_rank != 0)
	{

	//Updating unique word details of current worker node with total numDocsWithWord from other worker nodes for each word
	//loop through unique words of other worker nodes, compare each with current node's word, and update current node's numDocsWithWord accordingly
	// printf("rank %d sum %d\n", proc_rank,sum_of_uwcount_from_workers );
	for(int i = 0; i < uw_idx; i++)
	{
		for( int j = 0; j < sum_of_uwcount_from_workers; j++)
		{
			if(!strcmp(uw_details_from_workers[j].word, unique_words[i].word))
			{
				unique_words[i].numDocsWithWord += uw_details_from_workers[j].numDocsWithWord;
				//break;
			}
		}
	}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	if(proc_rank != 0)
	{

	// //debug
	// for(int j = 0; j < uw_idx; j++)
	// {
	// 	printf("Rank %d word %s numdocswithword %d\n", proc_rank, unique_words[j].word, unique_words[j].numDocsWithWord);
	// }


	// Use unique_words array to populate TFICF objects with: numDocsWithWord
	for(i=0; i<TF_idx; i++) {
		for(j=0; j<uw_idx; j++) {
			if(!strcmp(TFICF[i].word, unique_words[j].word)) {
				TFICF[i].numDocsWithWord = unique_words[j].numDocsWithWord;	
				break;
			}
		}
	}
	
	// Print ICF job similar to HW4/HW5 (For debugging purposes)
	printf("------------ICF Job-------------\n");
	for(j=0; j<TF_idx; j++)
		printf("%s@%s\t%d/%d\n", TFICF[j].word, TFICF[j].document, TFICF[j].numDocs, TFICF[j].numDocsWithWord);
		
	// Calculates TFICF value 
	for(j=0; j<TF_idx; j++) {
		double TF = log10( 1.0 * TFICF[j].wordCount / TFICF[j].docSize + 1 );
		double ICF = log10(1.0 * (TFICF[j].numDocs + 1) / (TFICF[j].numDocsWithWord + 1) );
		double TFICF_value = TF * ICF;
		TFICF[j].tficf_val = TFICF_value;
		//sprintf(strings[j], "%s@%s\t%.16f", TFICF[j].document, TFICF[j].word, TFICF_value);
	}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//Setting up the struct object that needs to be gathered by the root from every worker
	int count_tficf = 7;

	MPI_Datatype types_tficf[count_tficf];
	types_tficf[0] = MPI_CHAR;
	types_tficf[1] = MPI_CHAR;
	types_tficf[2] = MPI_INT;
	types_tficf[3] = MPI_INT;
	types_tficf[4] = MPI_INT;
	types_tficf[5] = MPI_INT;
	types_tficf[6] = MPI_DOUBLE;

	int blocklength_tficf[count_tficf];
	blocklength_tficf[0]=32;
	blocklength_tficf[1]=8;
	blocklength_tficf[2]=1;
	blocklength_tficf[3]=1;
	blocklength_tficf[4]=1;
	blocklength_tficf[5]=1;
	blocklength_tficf[6]=1;

	MPI_Aint offset_tficf[count_tficf];
	offset_tficf[0] = offsetof(obj, word);
	offset_tficf[1] = offsetof(obj, document);
	offset_tficf[2] = offsetof(obj, wordCount);
	offset_tficf[3] = offsetof(obj, docSize);
	offset_tficf[4] = offsetof(obj, numDocs);
	offset_tficf[5] = offsetof(obj, numDocsWithWord);
	offset_tficf[6] = offsetof(obj, tficf_val);

	MPI_Datatype my_tficf_type;

	// Creating the struct
	MPI_Type_create_struct(count_tficf, blocklength_tficf, offset_tficf, types_tficf, &my_tficf_type);
	MPI_Type_commit(&my_tficf_type);

	int *num_of_tficf_values;
	num_of_tficf_values = (int *) malloc((num_of_proc) * sizeof(int));
	MPI_Request *gather_req;
	gather_req = (MPI_Request *)malloc(sizeof(MPI_Request));

	//gather all tf_idx from worker nodes
	MPI_Igather(&TF_idx, 1, MPI_INT, num_of_tficf_values, 1, MPI_INT, 0, MPI_COMM_WORLD, &gather_req[0]);
	MPI_Waitall(1, gather_req, MPI_STATUS_IGNORE);

	
	// for(int i=0; i<num_of_proc;i++)
	// {
	// 	printf("rank %d num of tficf %d\n", proc_rank, num_of_tficf_values[i]);
	// }

	int TF_idx1=0;
	int current_tficf_recv_num = 0;
	int displacement_values[num_of_proc];
	if(proc_rank == 0)
	{
		int temp_sum = 0;
		//find sum of tf_idx values	
		for(int i = 1; i< num_of_proc; i++)
		{
			temp_sum += num_of_tficf_values[i];
		}
		//printf("temp sum %d\n", temp_sum);

		TF_idx1 = temp_sum;
		// printf("tf_idx1 %d\n", TF_idx1);

		//calculate displacement array for Igatherv, which is the final gather
		
		for(int i=0; i<num_of_proc;i++)
		{
			displacement_values[i]=0;
		}
		int sum_of_displacement=0;
		for(int j=1; j<num_of_proc;j++)
		{
			current_tficf_recv_num += num_of_tficf_values[j];
			displacement_values[j] = sum_of_displacement;
			sum_of_displacement += num_of_tficf_values[j];
		}
	}	


	obj *TFICF_master;
	TFICF_master = (obj*)malloc(sizeof(obj) * MAX_WORDS_IN_CORPUS);
	MPI_Request final_gather[1];

	//Gather all TFICF values at master node from worker nodes
    MPI_Igatherv(TFICF, TF_idx, my_tficf_type, TFICF_master, num_of_tficf_values,displacement_values,my_tficf_type, 0, MPI_COMM_WORLD, &final_gather[0]);
	MPI_Waitall(1, final_gather, MPI_STATUS_IGNORE);

	// if(proc_rank==0)
	// {
	// 	for(int i=0; i<TF_idx1;i++)
	// 	{
	// 		printf("word %s numdocswithword %d\n", TFICF_master[i].word, TFICF_master[i].numDocsWithWord);
	// 	}
		
	// }
	if(proc_rank==0)
	{
		for(j=0; j<TF_idx1; j++)
		{
			sprintf(strings[j], "%s@%s\t%.16f", TFICF_master[j].document, TFICF_master[j].word, TFICF_master[j].tficf_val);
		}

		// Sort strings and print to file
		qsort(strings, TF_idx1, sizeof(char)*MAX_STRING_LENGTH, myCompare);
		FILE* fp = fopen("output.txt", "w");
		if(fp == NULL){
			printf("Error Opening File: output.txt\n");
			exit(0);
		}
		for(i=0; i<TF_idx1; i++)
			fprintf(fp, "%s\n", strings[i]);
		fclose(fp);
		
		
	}	
	
		MPI_Finalize();
		return 0;
}
