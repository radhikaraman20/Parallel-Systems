//Single Author info:
//rbraman Radhika B Raman

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.util.*;

import java.util.Map.Entry;

/*
 * Main class of the TFICF MapReduce implementation.
 * Author: Tyler Stocksdale
 * Date:   10/18/2017
 */
public class TFICF {

    public static void main(String[] args) throws Exception {
        // Check for correct usage
        if (args.length != 2) {
            System.err.println("Usage: TFICF <input corpus0 dir> <input corpus1 dir>");
            System.exit(1);
        }
		
		// return value of run func
		int ret = 0;
		
		// Create configuration
		Configuration conf0 = new Configuration();
		Configuration conf1 = new Configuration();
		
		// Input and output paths for each job
		Path inputPath0 = new Path(args[0]);
		Path inputPath1 = new Path(args[1]);
        try{
            ret = run(conf0, inputPath0, 0);
        }catch(Exception e){
            e.printStackTrace();
        }
        if(ret == 0){
        	try{
            	run(conf1, inputPath1, 1);
        	}catch(Exception e){
            	e.printStackTrace();
        	}        	
        }
     
     	System.exit(ret);
    }
		
	public static int run(Configuration conf, Path path, int index) throws Exception{
		// Input and output paths for each job

		Path wcInputPath = path;
		Path wcOutputPath = new Path("output" +index + "/WordCount");
		Path dsInputPath = wcOutputPath;
		Path dsOutputPath = new Path("output" + index + "/DocSize");
		Path tficfInputPath = dsOutputPath;
		Path tficfOutputPath = new Path("output" + index + "/TFICF");
		
		// Get/set the number of documents (to be used in the TFICF MapReduce job)
        FileSystem fs = path.getFileSystem(conf);
        FileStatus[] stat = fs.listStatus(path);
		String numDocs = String.valueOf(stat.length);
		conf.set("numDocs", numDocs);
		
		// Delete output paths if they exist
		FileSystem hdfs = FileSystem.get(conf);
		if (hdfs.exists(wcOutputPath))
			hdfs.delete(wcOutputPath, true);
		if (hdfs.exists(dsOutputPath))
			hdfs.delete(dsOutputPath, true);
		if (hdfs.exists(tficfOutputPath))
			hdfs.delete(tficfOutputPath, true);
		
		// Create and execute Word Count job
		
			/************ YOUR CODE HERE ************/

		Job wordcount_job = new Job(conf, "wordCount");
		wordcount_job.setJarByClass(TFICF.class);
		wordcount_job.setMapperClass(WCMapper.class);
		wordcount_job.setCombinerClass(WCReducer.class);
        wordcount_job.setReducerClass(WCReducer.class);

		wordcount_job.setOutputKeyClass(Text.class);
        wordcount_job.setOutputValueClass(IntWritable.class);

		//wordcount_job.setInputFormatClass(TextInputFormat.class);
        //wordcount_job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(wordcount_job, wcInputPath);
		FileOutputFormat.setOutputPath(wordcount_job, wcOutputPath);

		wordcount_job.waitForCompletion(true);

	
		// Create and execute Document Size job
		
			/************ YOUR CODE HERE ************/
		Job docsize_job = new Job(conf, "docSize");
		//docsize_job.setJarByClass(TFICF.class);
		docsize_job.setMapperClass(DSMapper.class);
		//docsize_job.setCombinerClass(DSReducer.class);
        docsize_job.setReducerClass(DSReducer.class);

		docsize_job.setOutputKeyClass(Text.class);
        docsize_job.setOutputValueClass(Text.class);

		//wordcount_job.setInputFormatClass(TextInputFormat.class);
        //wordcount_job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(docsize_job, dsInputPath);
		FileOutputFormat.setOutputPath(docsize_job, dsOutputPath);

		docsize_job.waitForCompletion(true);

		
		//Create and execute TFICF job
		
			/************ YOUR CODE HERE ************/
		Job tficf_job = new Job(conf, "tficf");
		//tficf_job.setJarByClass(TFICF.class);
		tficf_job.setMapperClass(TFICFMapper.class);
		//docsize_job.setCombinerClass(DSReducer.class);
        tficf_job.setReducerClass(TFICFReducer.class);

		tficf_job.setOutputKeyClass(Text.class);
        tficf_job.setOutputValueClass(Text.class);

		//wordcount_job.setInputFormatClass(TextInputFormat.class);
        //wordcount_job.setOutputFormatClass(TextOutputFormat.class);

		FileInputFormat.addInputPath(tficf_job, tficfInputPath);
		FileOutputFormat.setOutputPath(tficf_job, tficfOutputPath);


		//Return final job code , e.g. retrun tficfJob.waitForCompletion(true) ? 0 : 1
			/************ YOUR CODE HERE ************/

		//System.out.println("End of run function");
		return tficf_job.waitForCompletion(true) ? 0 : 1;
		
    }
	
	/*
	 * Creates a (key,value) pair for every word in the document 
	 *
	 * Input:  ( byte offset , contents of one line )
	 * Output: ( (word@document) , 1 )
	 *
	 * word = an individual word in the document
	 * document = the filename of the document
	 */
	public static class WCMapper extends Mapper<Object, Text, Text, IntWritable> {
		
		/************ YOUR CODE HERE ************/
		private String obtained_word = new String();

		private static final IntWritable one = new IntWritable(1);

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String individual_line = value.toString();

			StringTokenizer my_tokenizer = new StringTokenizer(individual_line);

			String name_of_file = new String();
			
			FileSplit fileSplit = (FileSplit)context.getInputSplit();
			name_of_file = (fileSplit.getPath().getName());
			
			String output_file = new String();
			
			Text output_file_name = new Text();
		

			while (my_tokenizer.hasMoreTokens()) {
				obtained_word = my_tokenizer.nextToken();

				if(Character.isDigit(obtained_word.charAt(0)))
				{
					continue;
				}

				//remove all occurences of double quotes from the string
				obtained_word = obtained_word.replaceAll("\"", "");
				obtained_word = obtained_word.replaceAll("'", "");

				
				obtained_word = obtained_word.replaceAll("\\(", "");
				obtained_word = obtained_word.replaceAll("\\)", "");

				obtained_word = obtained_word.replaceAll("\\[", "");
				obtained_word = obtained_word.replaceAll("\\]", "");
				
				obtained_word = obtained_word.replaceAll("\\,", "");
				obtained_word = obtained_word.replaceAll("\\!", "");
				obtained_word = obtained_word.replaceAll("\\.", "");
				obtained_word = obtained_word.replaceAll("\\`", "");
				obtained_word = obtained_word.replaceAll("\\:", "");
				obtained_word = obtained_word.replaceAll("\\?", "");
				obtained_word = obtained_word.replaceAll("\\&", "");
				obtained_word = obtained_word.replaceAll("\\*", "");
				obtained_word = obtained_word.replaceAll("\\=", "");
				obtained_word = obtained_word.replaceAll("\\¡", "");
				obtained_word = obtained_word.replaceAll("\\ª", "");
				obtained_word = obtained_word.replaceAll("\\;", "");
				
				obtained_word = obtained_word.replaceAll("\\{", "");
				obtained_word = obtained_word.replaceAll("\\}", "");


				if(obtained_word.length() == 0)
				{
					continue;
				}
				else if(Character.isDigit(obtained_word.charAt(0)))
				{
					continue;
				}
				else if(obtained_word.startsWith("-"))
				{
					continue;
				}
				else if(obtained_word.startsWith("¿"))
				{
					continue;
				}

				obtained_word = obtained_word.toLowerCase();

				output_file = obtained_word + "@" + name_of_file;
				output_file_name.set(output_file);
				context.write(output_file_name, one);
				
			}

			//System.out.println("end of mapper func");
		
    	}
	}

    /*
	 * For each identical key (word@document), reduces the values (1) into a sum (wordCount)
	 *
	 * Input:  ( (word@document) , 1 )
	 * Output: ( (word@document) , wordCount )
	 *
	 * wordCount = number of times word appears in document
	 */
	public static class WCReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
		
		/************ YOUR CODE HERE ************/
		public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int total_count = 0;

			for(IntWritable val: values){
				total_count += val.get();
			}

			context.write(key, new IntWritable(total_count));
			//System.out.println("end of reduce func");

		}
		
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the document as the key
	 *
	 * Input:  ( (word@document) , wordCount )
	 * Output: ( document , (word=wordCount) )
	 */
	public static class DSMapper extends Mapper<Object, Text, Text, Text> {
		
		/************ YOUR CODE HERE ************/
		private static Text obtained_document = new Text();
		private static Text count_of_word = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
			String[] word_doc_counter = value.toString().split("\t");
			String[] word_with_doc = word_doc_counter[0].split("@");

			obtained_document.set(word_with_doc[1]);
			count_of_word.set(word_with_doc[0] + "=" + word_doc_counter[1]);
			//System.out.println(obtained_document.toString() + " " + count_of_word.toString());
			context.write(obtained_document, count_of_word);
		}
		
    }

    /*
	 * For each identical key (document), reduces the values (word=wordCount) into a sum (docSize) 
	 *
	 * Input:  ( document , (word=wordCount) )
	 * Output: ( (word@document) , (wordCount/docSize) )
	 *
	 * docSize = total number of words in the document
	 */
	public static class DSReducer extends Reducer<Text, Text, Text, Text> {
		
		/************ YOUR CODE HERE ************/
		private static Text final_key = new Text(); 
 		private static Text final_value = new Text();

		protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
			Map<String, Integer> temp_count = new HashMap<String, Integer>(); 
			int sum_words_in_doc = 0;

			//System.out.println("INSIDE REDUUCER");		

			for(Text v: values)
			{	//System.out.println(key.toString() + " " + v.toString());
				String[] word_count = v.toString().split("="); 
   				temp_count.put(word_count[0], Integer.valueOf(word_count[word_count.length - 1])); 
   				sum_words_in_doc += Integer.parseInt(word_count[1]); 
			}

			for(Entry<String, Integer> entry : temp_count.entrySet())
			{
				final_key.set(entry.getKey() + "@" + key.toString()); 
   				final_value.set(entry.getValue() + "/" + sum_words_in_doc); 
   				
				context.write(final_key, final_value); 
			}
		}
    }
	
	/*
	 * Rearranges the (key,value) pairs to have only the word as the key
	 * 
	 * Input:  ( (word@document) , (wordCount/docSize) )
	 * Output: ( word , (document=wordCount/docSize) )
	 */
	public static class TFICFMapper extends Mapper<Object, Text, Text, Text> {

		/************ YOUR CODE HERE ************/
		private static Text final_key_word = new Text(); 
 		private static Text final_value_document = new Text();

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
			String[] key_value_split = value.toString().split("\t");
			String[] key_split = key_value_split[0].split("@");

			final_key_word.set(key_split[0]);
			final_value_document.set(key_split[1] + "=" + key_value_split[1]);

			context.write(final_key_word, final_value_document);
		}
	
		
    }

    /*
	 * For each identical key (word), reduces the values (document=wordCount/docSize) into a 
	 * the final TFICF value (TFICF). Along the way, calculates the total number of documents and 
	 * the number of documents that contain the word.
	 * 
	 * Input:  ( word , (document=wordCount/docSize) )
	 * Output: ( (document@word) , TFICF )
	 *
	 * numDocs = total number of documents
	 * numDocsWithWord = number of documents containing word
	 * TFICF = ln(wordCount/docSize + 1) * ln(numDocs/numDocsWithWord +1)
	 *
	 * Note: The output (key,value) pairs are sorted using TreeMap ONLY for grading purposes. For
	 *       extremely large datasets, having a for loop iterate through all the (key,value) pairs 
	 *       is highly inefficient!
	 */
	public static class TFICFReducer extends Reducer<Text, Text, Text, Text> {
		
		private static int numDocs;
		private Map<Text, Text> tficfMap = new HashMap<>();
		
		// gets the numDocs value and stores it
		protected void setup(Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			numDocs = Integer.parseInt(conf.get("numDocs"));
		}
		
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			
			/************ YOUR CODE HERE ************/
			int num_docs_with_key = 0;

			Map<String, String> temp_word_freq = new HashMap<String, String>();

			for (Text v : values)
			{
				String[] doc_and_freq = v.toString().split("=");
				num_docs_with_key++;
				temp_word_freq.put(doc_and_freq[0], doc_and_freq[1]);
			}

			//Map<String, Double> tficfMap = new HashMap<String, Double>();

			for (Entry<String, String> entry : temp_word_freq.entrySet())
			{
				Text final_doc_at_word = new Text();
				Text final_TFICF = new Text();
				//System.out.println(entry);
				String[] freq_and_total = entry.getValue().split("/"); 
				//System.out.println(Arrays.toString(freq_and_total));

				//calculate term frequency and inverse document frequency
				double term_frequency = Double.parseDouble(freq_and_total[0]) / Double.parseDouble(freq_and_total[1]); 
				double inv_doc_frequency = ((double) numDocs + 1 ) / ((double) num_docs_with_key + 1);

				double tficf = Math.log10(term_frequency + 1) * Math.log10(inv_doc_frequency); 

				final_doc_at_word.set(entry.getKey() + "@" + key);
				//System.out.println(final_doc_at_word);
				final_TFICF.set(Double.toString(tficf));
				//System.out.println(final_TFICF);


				//Put the output (key,value) pair into the tficfMap instead of doing a context.write
				tficfMap.put(final_doc_at_word, final_TFICF);

			}

			// for (Map.Entry<Text, Text> entry : tficfMap.entrySet()) {
			// 	System.out.println(entry.getKey()+" : "+entry.getValue());
			// 	}			
		}
		
		// sorts the output (key,value) pairs that are contained in the tficfMap
		protected void cleanup(Context context) throws IOException, InterruptedException {
            Map<Text, Text> sortedMap = new TreeMap<Text, Text>(tficfMap);
			for (Text key : sortedMap.keySet()) {
				//System.out.println(key + " " + sortedMap.keySet());
                context.write(key, sortedMap.get(key));
            }
        }
		
    }
}
