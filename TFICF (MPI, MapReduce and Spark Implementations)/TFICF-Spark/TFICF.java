//Single Author info:
//rbraman Radhika B Raman

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

import java.util.*;

/*
 * Main class of the TFICF Spark implementation.
 * Author: Tyler Stocksdale
 * Date:   10/31/2017
 */
public class TFICF {

	static boolean DEBUG = true;

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 1) {
            System.err.println("Usage: TFICF <input dir>");
            System.exit(1);
        }
		
		// Create a Java Spark Context
		SparkConf conf = new SparkConf().setAppName("TFICF");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load our input data
		// Output is: ( filePath , fileContents ) for each file in inputPath
		String inputPath = args[0];
		JavaPairRDD<String,String> filesRDD = sc.wholeTextFiles(inputPath);
		
		// Get/set the number of documents (to be used in the ICF job)
		long numDocs = filesRDD.count();
		
		//Print filesRDD contents
		if (DEBUG) {
			List<Tuple2<String, String>> list = filesRDD.collect();
			System.out.println("------Contents of filesRDD------");
			for (Tuple2<String, String> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2.trim() + ")");
			}
			System.out.println("--------------------------------");
		}
		
		/* 
		 * Initial Job
		 * Creates initial JavaPairRDD from filesRDD
		 * Contains each word@document from the corpus and also attaches the document size for 
		 * later use
		 * 
		 * Input:  ( filePath , fileContents )
		 * Map:    ( (word@document) , docSize )
		 */
		JavaPairRDD<String,Integer> wordsRDD = filesRDD.flatMapToPair(
			new PairFlatMapFunction<Tuple2<String,String>,String,Integer>() {
				public Iterable<Tuple2<String,Integer>> call(Tuple2<String,String> x) {
					// Collect data attributes
					String[] filePath = x._1.split("/");
					String document = filePath[filePath.length-1];
					String fileContents = x._2;
					String[] words = fileContents.split("\\s+");
					int docSize = words.length;
					
					// Output to Arraylist
					ArrayList ret = new ArrayList();
					for(String word : words) {
						ret.add(new Tuple2(word.trim() + "@" + document, docSize));
					}
					return ret;
				}
			}
		);
		
		//Print wordsRDD contents
		if (DEBUG) {
			List<Tuple2<String, Integer>> list = wordsRDD.collect();
			System.out.println("------Contents of wordsRDD------");
			for (Tuple2<String, Integer> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
			}
			System.out.println("--------------------------------");
		}		
		
		/* 
		 * TF Job (Word Count Job + Document Size Job)
		 * Gathers all data needed for TF calculation from wordsRDD
		 *
		 * Input:  ( (word@document) , docSize )
		 * Map:    ( (word@document) , (1/docSize) )
		 * Reduce: ( (word@document) , (wordCount/docSize) )
		 */
		JavaPairRDD<String,String> tfRDD = wordsRDD.mapToPair(

				/************ YOUR CODE HERE ************/
				new PairFunction<Tuple2<String, Integer>, String, String>()
				{
					public Tuple2<String, String> call(Tuple2<String, Integer> func_input)
					{
						//obtain docSize value from the input ( (word@document) , docSize )
						int doc_size_value = func_input._2;

						//formulate required output of format 1/docSize
						String map_output_value = "1/" + doc_size_value;

						//create the final output tuple
						String word_at_doc_value = func_input._1;
						Tuple2<String, String> map_output_final = new Tuple2<>(word_at_doc_value, map_output_value);

						//return the above Tuple2
						return map_output_final;
						
					}
				}

		).reduceByKey(
			
			/************ YOUR CODE HERE ************/
			new Function2<String, String, String>()
			{
				public String call(String str1, String str2)
				{
					//obtain individual values from (wordCount/docSize) that is part of the input, store in a String array
					String[] split_of_str1 = str1.split("/");

					//repeat the above for str2 as well
					String[] split_of_str2 = str2.split("/");

					//obtain docSize value from split of either str1 or str2, which would be the second element in the array
					String doc_siz_val = split_of_str1[1];

					//now perform reduction by calculating word count, by adding numerators of str1 and str2
					int str1_numer = Integer.parseInt(split_of_str1[0]);
					int str2_numer = Integer.parseInt(split_of_str2[0]);

					int req_word_count_val = str1_numer + str2_numer;

					//formulate required output of format (wordCount/docSize)
					String reduce_output = req_word_count_val + "/" + doc_siz_val;
					return reduce_output;

					
				}
			}
			
		);
		
		//Print tfRDD contents
		if (DEBUG) {
			List<Tuple2<String, String>> list = tfRDD.collect();
			System.out.println("-------Contents of tfRDD--------");
			for (Tuple2<String, String> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
			}
			System.out.println("--------------------------------");
		}
		
		/*
		 * ICF Job
		 * Gathers all data needed for ICF calculation from tfRDD
		 *
		 * Input:  ( (word@document) , (wordCount/docSize) )
		 * Map:    ( word , (1/document) )
		 * Reduce: ( word , (numDocsWithWord/document1,document2...) )
		 * Map:    ( (word@document) , (numDocs/numDocsWithWord) )
		 */
		JavaPairRDD<String,String> icfRDD = tfRDD.mapToPair(
			
			/************ YOUR CODE HERE ************/
			new PairFunction<Tuple2<String, String>, String, String>()
			{
				public Tuple2<String, String> call(Tuple2<String, String> func_input)
				{
					//obtain word and document values from the input in order to be used in the output
					String word_at_doc_value = func_input._1;
					String[] word_doc_split = word_at_doc_value.split("@");
					String word_value = word_doc_split[0];
					String doc_value = word_doc_split[1];

					//formulate the output of format (1/document)
					String output_value = "1/" + doc_value;

					//create the final output tuple
					Tuple2<String, String> icfmap_output_final = new Tuple2<>(word_value, output_value);

					//return the above Tuple2
					return icfmap_output_final;

				}
			}
			
		).reduceByKey(
			
			/************ YOUR CODE HERE ************/
			new Function2<String, String, String>()
			{
				public String call(String str1, String str2)
				{
					//obtain individual values from (1/document) for every word, store in a String array
					String[] split_of_str1 = str1.split("/");

					//repeat the above for str2 as well
					String[] split_of_str2 = str2.split("/");

					//perform reduction by calculating num of docs that have a particular word, by adding numerators of str1 and str2
					int str1_numer = Integer.parseInt(split_of_str1[0]);
					int str2_numer = Integer.parseInt(split_of_str2[0]);
					int num_of_docs_with_word = str1_numer + str2_numer;

					//formulate output of required format (numDocsWithWord/document1,document2...)
					String str1_docvalue = split_of_str1[1];
					String str2_docvalue = split_of_str2[1];
					String req_output_value = num_of_docs_with_word + "/" + str1_docvalue + "," + str2_docvalue;

					return req_output_value;

				}
			}
			
		).flatMapToPair(
			
			/************ YOUR CODE HERE ************/
			new PairFlatMapFunction<Tuple2<String,String>, String, String>()
			{
				public Iterable<Tuple2<String, String>> call(Tuple2<String, String> func_input)
				{
					//obtain word from the input tuple, along with (numDocsWithWord/document1,document2...)
					String word_value = func_input._1;
					String input_numdocs_listdocs = func_input._2;

					//split (numDocsWithWord/document1,document2...) to seperate numerator and denominator
					String[] split_numdocs_listdocs = input_numdocs_listdocs.split("/");

					String num_of_docs_with_word = split_numdocs_listdocs[0];
					String[] doclist = split_numdocs_listdocs[1].split(",");

					ArrayList second_map_output = new ArrayList();
					for(String doc : doclist)
					{
						String tuple_key = word_value + "@" + doc;
						String tuple_value = numDocs + "/" + num_of_docs_with_word;

						Tuple2<String, String> req_output = new Tuple2<>(tuple_key, tuple_value);
						second_map_output.add(req_output);
					}
					return second_map_output;
					
				}
			}
			
		);
		
		//Print icfRDD contents
		if (DEBUG) {
			List<Tuple2<String, String>> list = icfRDD.collect();
			System.out.println("-------Contents of icfRDD-------");
			for (Tuple2<String, String> tuple : list) {
				System.out.println("(" + tuple._1 + ") , (" + tuple._2 + ")");
			}
			System.out.println("--------------------------------");
		}
	
		/*
		 * TF * ICF Job
		 * Calculates final TFICF value from tfRDD and icfRDD
		 *
		 * Input:  ( (word@document) , (wordCount/docSize) )          [from tfRDD]
		 * Map:    ( (word@document) , TF )
		 * 
		 * Input:  ( (word@document) , (numDocs/numDocsWithWord) )    [from icfRDD]
		 * Map:    ( (word@document) , ICF )
		 * 
		 * Union:  ( (word@document) , TF )  U  ( (word@document) , ICF )
		 * Reduce: ( (word@document) , TFICF )
		 * Map:    ( (document@word) , TFICF )
		 *
		 * where TF    = log( wordCount/docSize + 1 )
		 * where ICF   = log( (Total numDocs in the corpus + 1) / (numDocsWithWord in the corpus + 1) )
		 * where TFICF = TF * ICF
		 */
		JavaPairRDD<String,Double> tfFinalRDD = tfRDD.mapToPair(
			new PairFunction<Tuple2<String,String>,String,Double>() {
				public Tuple2<String,Double> call(Tuple2<String,String> x) {
					double wordCount = Double.parseDouble(x._2.split("/")[0]);
					double docSize = Double.parseDouble(x._2.split("/")[1]);
					double TF = Math.log10(wordCount/docSize + 1);
					return new Tuple2(x._1, TF);
				}
			}
		);
		
		JavaPairRDD<String,Double> idfFinalRDD = icfRDD.mapToPair(

				/************ YOUR CODE HERE ************/
				new PairFunction<Tuple2<String, String>, String, Double>()
				{
					public Tuple2<String, Double> call(Tuple2<String, String> func_input)
					{
						//numDocs is globally defined, so I will extract only numDocsWithWord from the input
						double num_docs_with_word = Double.parseDouble(func_input._2.split("/")[1]);
						double ICF = Math.log10((numDocs + 1)/(num_docs_with_word + 1));
						return new Tuple2(func_input._1, ICF);
					}
				}

		);
		
		JavaPairRDD<String,Double> tficfRDD = tfFinalRDD.union(idfFinalRDD).reduceByKey(

			/************ YOUR CODE HERE ************/
				new Function2<Double, Double, Double>()
				{
					public Double call(Double TF, Double ICF)
					{
						return TF * ICF;
					}
				}
			
		).mapToPair(

			/************ YOUR CODE HERE ************/
				new PairFunction<Tuple2<String, Double>, String, Double>()
				{
					public Tuple2<String, Double> call(Tuple2<String, Double> func_input)
					{
						//obtain (word@document) value from the input
						String word_at_document = func_input._1;

						//split the above at "@" to get the word and document values seperately
						String word_value = word_at_document.split("@")[0];
						String document_value = word_at_document.split("@")[1];

						double tficf_value = func_input._2;

						//return final output of format ( (document@word) , TFICF )
						return new Tuple2(document_value + "@" + word_value, tficf_value);
						
					}
				}
			
		);
		
		//Print tficfRDD contents in sorted order
		Map<String, Double> sortedMap = new TreeMap<>();
		List<Tuple2<String, Double>> list = tficfRDD.collect();
		for (Tuple2<String, Double> tuple : list) {
			sortedMap.put(tuple._1, tuple._2);
		}
		if(DEBUG) System.out.println("-------Contents of tficfRDD-------");
		for (String key : sortedMap.keySet()) {
			System.out.println(key + "\t" + sortedMap.get(key));
		}
		if(DEBUG) System.out.println("--------------------------------");	 
	}	
}
