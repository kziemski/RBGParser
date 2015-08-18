package parser;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import parser.io.DependencyReader;
import parser.io.DependencyWriter;

public class DependencyParser implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	protected Options options;
	protected DependencyPipe pipe;
	protected Parameters parameters;
	
	public static void main(String[] args) 
		throws IOException, ClassNotFoundException, CloneNotSupportedException
	{
		
		Options options = new Options();
		options.processArguments(args);
		
		if (options.train) {
			DependencyParser parser = new DependencyParser();
			parser.options = options;
			options.printOptions();
			
			DependencyPipe pipe = new DependencyPipe(options);
			parser.pipe = pipe;
			
			pipe.createAlphabets(options.trainFile);
			
			DependencyInstance[] lstTrain = pipe.createInstances(options.trainFile);
			pipe.pruneLabel(lstTrain);
			
			Parameters parameters = new Parameters(pipe, options);
			parser.parameters = parameters;
			
			parser.train(lstTrain);
			parser.saveModel();
		}
		
		if (options.test) {
			DependencyParser parser = new DependencyParser();
			parser.options = options;
			parser.loadModel();
			parser.options.processArguments(args);
			if (!options.train) parser.options.printOptions();
			
			System.out.printf(" Evaluating: %s%n", options.testFile);
			parser.evaluateSet(true, true);
		}
		
	}
	
    public void saveModel() throws IOException 
    {
    	ObjectOutputStream out = new ObjectOutputStream(
    			new GZIPOutputStream(new FileOutputStream(options.modelFile)));
    	out.writeObject(pipe);
    	out.writeObject(parameters);
    	out.writeObject(options);
    	out.close();
    }
	
    public void loadModel() throws IOException, ClassNotFoundException 
    {
        ObjectInputStream in = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(options.modelFile)));    
        pipe = (DependencyPipe) in.readObject();
        parameters = (Parameters) in.readObject();
        options = (Options) in.readObject();
        pipe.options = options;      
        in.close();
        pipe.closeAlphabets();
    }
	
    public void train(DependencyInstance[] lstTrain) 
    	throws IOException, CloneNotSupportedException 
    {
    	long start = 0, end = 0;
    	
        if ((options.R > 0 || options.R2 > 0) && options.gammaLabel < 1 && options.initTensorWithPretrain) {

        	Options optionsBak = (Options) options.clone();
        	options.R = 0;
        	options.R2 = 0;
        	options.gammaLabel = 1.0;
        	options.maxNumIters = options.numPretrainIters;
        	parameters.rank = 0;
        	parameters.rank2 = 0;
        	parameters.gammaL = 1.0;
        	
    		System.out.println("=============================================");
    		System.out.printf(" Pre-training:%n");
    		System.out.println("=============================================");
    		
    		start = System.currentTimeMillis();

    		System.out.println("Running MIRA ... ");
    		trainIter(lstTrain, false);
    		System.out.println();
    		
    		options = optionsBak;
    		parameters.rank = options.R;
        	parameters.rank2 = options.R2;
        	parameters.gammaL = options.gammaLabel;
    		
    		System.out.println("Init tensor ... ");
    		int n = parameters.N;
    		int d = parameters.DL;
        	LowRankTensor tensor = new LowRankTensor(new int[] {n, n, d}, options.R);
        	LowRankTensor tensor2 = new LowRankTensor(new int[] {n, n, n, d, d}, options.R2);
        	pipe.synFactory.fillParameters(tensor, tensor2, parameters);
        	
        	ArrayList<double[][]> param = new ArrayList<double[][]>();
        	param.add(parameters.U);
        	param.add(parameters.V);
        	param.add(parameters.WL);
        	tensor.decompose(param);
        	if (options.useGP) {
        		ArrayList<double[][]> param2 = new ArrayList<double[][]>();
        		param2.add(parameters.U2);
        		param2.add(parameters.V2);
        		param2.add(parameters.W2);
            	param2.add(parameters.X2L);
            	param2.add(parameters.Y2L);
            	tensor2.decompose(param2);
        	}
        	parameters.assignTotal();
        	parameters.printStat();
        	
            System.out.println();
    		end = System.currentTimeMillis();
            System.out.println();
            System.out.printf("Pre-training took %d ms.%n", end-start);    		
    		System.out.println("=============================================");
    		System.out.println();

        } else {
        	parameters.randomlyInit();
        }
        
		System.out.println("=============================================");
		System.out.printf(" Training:%n");
		System.out.println("=============================================");
		
		start = System.currentTimeMillis();

		System.out.println("Running MIRA ... ");
		trainIter(lstTrain, true);
		System.out.println();
		
		end = System.currentTimeMillis();
		
		System.out.printf("Training took %d ms.%n", end-start);    		
		System.out.println("=============================================");
		System.out.println();		    	
    }
    
    public void trainIter(DependencyInstance[] lstTrain, boolean evalAndSave) throws IOException
    {	
    	int N = lstTrain.length;
    	int printPeriod = 10000 < N ? N/10 : 1000;
    	
    	for (int iIter = 0; iIter < options.maxNumIters; ++iIter) {

    		long start = 0;
    		double loss = 0;
    		int las = 0, tot = 0;
    		start = System.currentTimeMillis();	
    		
    		for (int i = 0; i < N; ++i) {
    			
    			if ((i + 1) % printPeriod == 0) {
				System.out.printf("  %d (time=%ds)", (i+1),
					(System.currentTimeMillis()-start)/1000);
    			}

    			DependencyInstance inst = lstTrain[i];
    			LocalFeatureData lfd = new LocalFeatureData(inst, this);
    		    int n = inst.length;
    		    int[] predDeps = inst.heads;
    		    int[] predLabs = new int [n];
    		        		
        		lfd.predictLabels(predDeps, predLabs, true);
        		int la = evaluateLabelCorrect(inst.heads, inst.deplbids, predDeps, predLabs);
    			if (la != n-1) {
    				loss += parameters.updateLabel(inst, predDeps, predLabs, lfd,
    						iIter * N + i + 1);
    			}
        		las += la;
        		tot += n-1;
    		}
    		
    		System.out.printf("%n  Iter %d\tloss=%.4f\tlas=%.4f\t[%ds]%n", iIter+1,
    				loss, las/(tot+0.0),
    				(System.currentTimeMillis() - start)/1000);
    		System.out.println();
    		
    		parameters.printStat();
    		
    		// evaluate on a development set
    		if (evalAndSave && options.test && ((iIter+1) % 1 == 0 || iIter+1 == options.maxNumIters)) {		
    			System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
	  			System.out.printf(" Evaluation: %s%n", options.testFile);
	  			System.out.println(); 
                if (options.average) 
                	parameters.averageParameters((iIter+1)*N, 1);
	  			evaluateSet(false, true);
                System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
                if (options.average) 
                	parameters.averageParameters((iIter+1)*N, -1);
    		} 
    	}
    	
    	if (evalAndSave && options.average) {
            parameters.averageParameters(options.maxNumIters * N, 1);
    	}
    }
    
    public int evaluateLabelCorrect(int[] actDeps, int[] actLabs, int[] predDeps, int[] predLabs)
    {
    	int nCorrect = 0;
    	for (int i = 1, N = actDeps.length; i < N; ++i) {
    		if (actDeps[i] == predDeps[i] && actLabs[i] == predLabs[i])
    			++nCorrect;
    	}    		  		
    	return nCorrect;
    }
    
    public void evaluateSet(boolean output, boolean evalWithPunc)
    		throws IOException {
    	
    	DependencyReader reader = DependencyReader.createDependencyReader(options);
    	reader.startReading(options.testFile);

    	DependencyWriter writer = null;
    	if (output && options.outFile != null) {
    		writer = DependencyWriter.createDependencyWriter(options, pipe);
    		writer.startWriting(options.outFile);
    	}  	
    	
    	Evaluator eval = new Evaluator(options, pipe);
    	
		long start = System.currentTimeMillis();
    	
    	DependencyInstance inst = pipe.createInstance(reader);    	
    	while (inst != null) {
    		LocalFeatureData lfd = new LocalFeatureData(inst, this);
    		int n = inst.length;
    		int[] predDeps = inst.heads;
		    int[] predLabs = new int [n];
            lfd.predictLabels(predDeps, predLabs, false);
            
            eval.add(inst, predDeps, predLabs, evalWithPunc);
    		
    		if (writer != null) {
    			writer.writeInstance(inst, predDeps, predLabs);
    		}
    		
    		inst = pipe.createInstance(reader);
    	}
    	
    	reader.close();
    	if (writer != null) writer.close();
    	
    	System.out.printf("  Tokens: %d%n", eval.tot);
    	System.out.printf("  Sentences: %d%n", eval.nsents);
    	System.out.printf("  UAS=%.6f\tLAS=%.6f\tCAS=%.6f\t[%.2fs]%n",
    			eval.UAS(), eval.LAS(), eval.CAS(),
    			(System.currentTimeMillis() - start)/1000.0);
    }
}
