package parser;

import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import parser.Options.LearningMode;
import parser.decoding.DependencyDecoder;
import parser.io.DependencyReader;
import parser.io.DependencyWriter;
import parser.pruning.BasicArcPruner;
import parser.sampling.RandomWalkSampler;
import utils.Utils;

public class DependencyParser implements Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	
	protected Options options;
	protected DependencyPipe pipe;
	protected Parameters parameters;
	
	DependencyParser pruner;
	
	double pruningGoldHits = 0;
	double pruningTotGold = 1e-30;
	double pruningTotUparcs = 0;
	double pruningTotArcs = 1e-30;
	
	public static void main(String[] args) 
		throws IOException, ClassNotFoundException, CloneNotSupportedException
	{
		
		Options options = new Options();
		options.processArguments(args);		
		
		DependencyParser pruner = null;
		if (options.train && options.pruning && options.learningMode != LearningMode.Basic) {
			Options prunerOptions = new Options();
			prunerOptions.processArguments(args);
			prunerOptions.maxNumIters = 10;
			
			prunerOptions.learningMode = LearningMode.Basic;
			prunerOptions.pruning = false;
			prunerOptions.test = false;
			prunerOptions.learnLabel = false;
			prunerOptions.gamma = 1.0;
			prunerOptions.gammaLabel = 1.0;
			prunerOptions.R = 0;
			prunerOptions.bits = options.bits - 2;
			
			//pruner = new DependencyParser();
			pruner = new BasicArcPruner();
			pruner.options = prunerOptions;
			
			DependencyPipe pipe = new DependencyPipe(prunerOptions);
			pruner.pipe = pipe;
			
			pipe.createAlphabets(prunerOptions.trainFile);
			pipe.synFactory.clearFeatureHashSet();
			
			DependencyInstance[] lstTrain = pipe.createInstances(prunerOptions.trainFile);
			
			Parameters parameters = new Parameters(pipe, prunerOptions);
			pruner.parameters = parameters;
			
			pruner.train(lstTrain);
		}
		
		if (options.train) {
			DependencyParser parser = new DependencyParser();
			parser.options = options;
			options.printOptions();
			
			DependencyPipe pipe = new DependencyPipe(options);
			parser.pipe = pipe;
			
			if (options.pruning) parser.pruner = pruner;
			
			pipe.createAlphabets(options.trainFile);
			
			DependencyInstance[] lstTrain = pipe.createInstances(options.trainFile);
			
			Parameters parameters = new Parameters(pipe, options);
			parser.parameters = parameters;
			
			parser.train(lstTrain);
			if (options.dev && options.learningMode != LearningMode.Basic) 
				parser.tuneSpeed();
			parser.saveModel();
		}
		
		if (options.test) {
			DependencyParser parser = new DependencyParser();
			parser.options = options;			
			
			parser.loadModel();
			parser.options.processArguments(args);
			if (!options.train) parser.options.printOptions(); 
			if (options.dev && options.learningMode != LearningMode.Basic) {
				parser.tuneSpeed();
				parser.saveModel();
			}
			
			System.out.printf(" Evaluating: %s%n", options.testFile);
			parser.evaluateSet(true, false);
		}
		
	}
	
	public void tuneSpeed() throws IOException, CloneNotSupportedException
	{
		if (options.numTestConverge < 10) return;
		System.out.println(" Tuning hill-climbing converge number on eval set...");
		double maxUAS = evaluateWithConvergeNum(options.numTrainConverge);
		System.out.printf("\tconverge=%d\tUAS=%f%n", options.numTrainConverge, maxUAS);
		int max = options.numTrainConverge / 5;
		int min = 2;
		while (min < max) {
			int mid = (min+max)/2;
			double uas = evaluateWithConvergeNum(mid*5);
			System.out.printf("\tconverge=%d\tUAS=%f%n", mid*5, uas);
			if (uas + 0.001 < maxUAS)
				min = mid+1;
			else
				max = mid;
		}
		options.numTestConverge = min * 5;
		options.dev = false;	// set dev=false because already tuned
		System.out.printf(" final converge=%d%n%n", options.numTestConverge);
	}
	
    public void saveModel() throws IOException 
    {
    	ObjectOutputStream out = new ObjectOutputStream(
    			new GZIPOutputStream(new FileOutputStream(options.modelFile)));
    	out.writeObject(pipe);
    	out.writeObject(parameters);
    	out.writeObject(options);
    	if (options.pruning && options.learningMode != LearningMode.Basic) 
    		out.writeObject(pruner);
    	out.close();
    }
	
    public void loadModel() throws IOException, ClassNotFoundException 
    {
        ObjectInputStream in = new ObjectInputStream(
                new GZIPInputStream(new FileInputStream(options.modelFile)));    
        pipe = (DependencyPipe) in.readObject();
        parameters = (Parameters) in.readObject();
        options = (Options) in.readObject();
        if (options.pruning && options.learningMode != LearningMode.Basic)
        	//pruner = (DependencyParser) in.readObject();
        	pruner = (BasicArcPruner) in.readObject();
        pipe.options = options;      
        in.close();
        pipe.closeAlphabets();
    }
    
	public void printPruningStats()
	{
		System.out.printf("  Pruning Recall: %.4f\tEffcy: %.4f%n",
				pruningGoldHits / pruningTotGold,
				pruningTotUparcs / pruningTotArcs);
	}
	
	public double pruningRecall()
	{
		return pruningGoldHits / pruningTotGold;
	}
	
	public void resetPruningStats()
	{
		pruningGoldHits = 0;
		pruningTotGold = 1e-30;
		pruningTotUparcs = 0;
		pruningTotArcs = 1e-30;
	}
	
    public void train(DependencyInstance[] lstTrain) 
    	throws IOException, CloneNotSupportedException 
    {
    	long start = 0, end = 0;
    	
        if (options.R > 0 && options.gamma < 1 && options.initTensorWithPretrain) {

        	Options optionsBak = (Options) options.clone();
        	if (options.useGP)
        		options.learningMode = LearningMode.Second;
        	else options.learningMode = LearningMode.Basic;
        	options.useCS = false;
        	options.useHB = false;
        	options.useGS = false;
        	options.useTS = false;
        	options.useGGP = false;
        	options.usePSC = false;
        	options.useHO = false;
        	options.R = 0;
        	options.R2 = 0;
        	options.gamma = 1.0;
        	options.gammaLabel = 1.0;
        	options.maxNumIters = options.numPretrainIters;
        	parameters.rank = 0;
        	parameters.rank2 = 0;
        	parameters.gamma = 1.0;
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
        	parameters.gamma = options.gamma;
        	parameters.gammaL = options.gammaLabel;
    		
    		System.out.println("Init tensor ... ");
    		int n = parameters.N;
    		int d = parameters.D + (options.learnLabel ? parameters.DL : 0);
    		int d2 = parameters.D2 + (options.learnLabel ? parameters.DL : 0);
        	LowRankTensor tensor = new LowRankTensor(new int[] {n, n, d}, options.R);
        	LowRankTensor tensor2 = new LowRankTensor(new int[] {n, n, n, d2, d2}, options.R2);
        	pipe.synFactory.fillParameters(tensor, tensor2, parameters);
        	
        	ArrayList<double[][]> param = new ArrayList<double[][]>();
        	param.add(parameters.U);
        	param.add(parameters.V);
        	double[][] W = new double[options.R][d];
        	param.add(W);
        	tensor.decompose(param);
        	for (int i = 0; i < options.R; ++i) {
    			for (int j = 0; j < parameters.D; ++j)
    				parameters.W[i][j] = W[i][j];
    			for (int j = parameters.D; j < d; ++j)
    				parameters.WL[i][j-parameters.D] = W[i][j];
    		}
        	
        	if (options.useGP) {
        		ArrayList<double[][]> param2 = new ArrayList<double[][]>();
        		param2.add(parameters.U2);
        		param2.add(parameters.V2);
        		param2.add(parameters.W2);
        		double[][] X2 = new double[options.R2][d2];
            	param2.add(X2);
            	double[][] Y2 = new double[options.R2][d2];
            	param2.add(Y2);
            	tensor2.decompose(param2);
            	for (int i = 0; i < options.R2; ++i) {
	    			for (int j = 0; j < parameters.D2; ++j) {
	    				parameters.X2[i][j] = X2[i][j];
	    				parameters.Y2[i][j] = Y2[i][j];
	    			}
	    			for (int j = parameters.D2; j < d2; ++j) {
	    				parameters.X2L[i][j-parameters.D2] = X2[i][j];
	    				parameters.Y2L[i][j-parameters.D2] = Y2[i][j];
	    			}
	    		}
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

    	DependencyDecoder decoder = DependencyDecoder
    			.createDependencyDecoder(options);
    	
    	int N = lstTrain.length;
    	int printPeriod = 10000 < N ? N/10 : 1000;
    	
    	for (int iIter = 0; iIter < options.maxNumIters; ++iIter) {
    	    
    		if (pruner != null) pruner.resetPruningStats();
    		
            // use this offset to change the udpate ordering of U, V and W
            // when N is a multiple of 3, such that U, V and W get updated
            // on each sentence.
            int offset = (N % 3 == 0) ? iIter : 0;

    		long start = 0;
    		double loss = 0;
    		int uas = 0, las = 0, tot = 0;
    		start = System.currentTimeMillis();	
    		
    		
    		for (int i = 0; i < N; ++i) {
    			
    			if ((i + 1) % printPeriod == 0) {
				System.out.printf("  %d (time=%ds)", (i+1),
					(System.currentTimeMillis()-start)/1000);
    			}

    			//DependencyInstance inst = new DependencyInstance(lstTrain[i]);
    			DependencyInstance inst = lstTrain[i];
    			LocalFeatureData lfd = new LocalFeatureData(inst, this, true, true);
    		    GlobalFeatureData gfd = new GlobalFeatureData(lfd);
    		    
    		    int n = inst.length;
    		    
    		    DependencyInstance predInst = decoder.decode(inst, lfd, gfd, true);

        		int ua = evaluateUnlabelCorrect(inst, predInst), la = 0;
    		
        		uas += ua;
        		tot += n-1;        		
        		if (ua != n-1) {
        			loss += parameters.update(inst, predInst, lfd, gfd,
        					iIter * N + i + 1, offset);
        			lfd.updateProjection();
                }
        		
        		if (options.learnLabel) {
        			predInst.heads = inst.heads;
        			lfd.predictLabels(predInst.heads, predInst.deplbids, true);
        			la = evaluateLabelCorrect(inst, predInst);
        			if (la != n-1) {
        				loss += parameters.updateLabel(inst, predInst, lfd,
        						iIter * N + i + 1);
        			}
        			las += la;
        		}
    		}
    		
    		System.out.printf("%n  Iter %d\tloss=%.4f\tuas=%.4f\tlas=%.4f\t[%ds]%n", iIter+1,
    				loss, uas/(tot+0.0), las/(tot+0.0),
    				(System.currentTimeMillis() - start)/1000);
    		System.out.println();
    		
    		parameters.printStat();
    		
    		if (options.learningMode != LearningMode.Basic && options.pruning && pruner != null)
    			pruner.printPruningStats();
    		
    		// evaluate on a development set
    		if (evalAndSave && options.test && ((iIter+1) % 1 == 0 || iIter+1 == options.maxNumIters)) {		
    			System.out.println();
	  			System.out.println("_____________________________________________");
	  			System.out.println();
	  			System.out.printf(" Evaluation: %s%n", options.testFile);
	  			System.out.println(); 
                if (options.average) 
                	parameters.averageParameters((iIter+1)*N, 1);
                int cnvg = options.numTestConverge;
                options.numTestConverge = options.numTrainConverge;
	  			double res = evaluateSet(false, false);
                options.numTestConverge = cnvg;
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

        decoder.shutdown();
    }
    
    public int evaluateUnlabelCorrect(DependencyInstance act, DependencyInstance pred) 
    {
    	int nCorrect = 0;
    	for (int i = 1, N = act.length; i < N; ++i) {
    		if (act.heads[i] == pred.heads[i])
    			++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public int evaluateLabelCorrect(DependencyInstance act, DependencyInstance pred) 
    {
    	int nCorrect = 0;
    	for (int i = 1, N = act.length; i < N; ++i) {
    		if (act.heads[i] == pred.heads[i] && act.deplbids[i] == pred.deplbids[i])
    			++nCorrect;
    	}    		  		
    	return nCorrect;
    }
    
    public static int evaluateUnlabelCorrect(DependencyInstance inst, 
    		DependencyInstance pred, boolean evalWithPunc) 
    {
    	int nCorrect = 0;    	
    	for (int i = 1, N = inst.length; i < N; ++i) {

            if (!evalWithPunc)
            	if (inst.forms[i].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;

    		if (inst.heads[i] == pred.heads[i]) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public static int evaluateLabelCorrect(DependencyInstance inst, 
    		DependencyInstance pred, boolean evalWithPunc) 
    {
    	int nCorrect = 0;    	
    	for (int i = 1, N = inst.length; i < N; ++i) {

            if (!evalWithPunc)
            	if (inst.forms[i].matches("[-!\"#%&'()*,./:;?@\\[\\]_{}、]+")) continue;

    		if (inst.heads[i] == pred.heads[i] && inst.deplbids[i] == pred.deplbids[i]) ++nCorrect;
    	}    		
    	return nCorrect;
    }
    
    public double evaluateSet(boolean output, boolean evalWithPunc)
    		throws IOException {
    	
    	if (pruner != null) pruner.resetPruningStats();
    	
    	DependencyReader reader = DependencyReader.createDependencyReader(options);
    	reader.startReading(options.testFile);
    	

    	DependencyWriter writer = null;
    	if (output && options.outFile != null) {
    		writer = DependencyWriter.createDependencyWriter(options, pipe);
    		writer.startWriting(options.outFile);
    	}
    	
    	DependencyDecoder decoder = DependencyDecoder.createDependencyDecoder(options);   	
    	
    	Evaluator eval = new Evaluator(options, pipe);
    	
		long start = System.currentTimeMillis();
    	
    	DependencyInstance inst = pipe.createInstance(reader);    	
    	while (inst != null) {
    		LocalFeatureData lfd = new LocalFeatureData(inst, this, true, false);
    		GlobalFeatureData gfd = new GlobalFeatureData(lfd); 

    		    		
            DependencyInstance predInst = decoder.decode(inst, lfd, gfd, false);
            if (options.learnLabel)
            	lfd.predictLabels(predInst.heads, predInst.deplbids, false);
            
            eval.add(inst, predInst, evalWithPunc);
    		
    		if (writer != null) {
    			//inst.heads = predInst.heads;
    			//inst.deplbids = predInst.deplbids;
    			writer.writeInstance(inst, predInst);
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
    	if (options.pruning && options.learningMode != LearningMode.Basic && pruner != null) {
    		pruner.printPruningStats();
    		if (pruner.pruningRecall() < 0.99) {
    			System.out.printf("%nWARNING: Pruning recall is less than 99%%!%n"
    					+"Current pruning-weight=%.2f. Consider using a smaller value between (0,1)%n%n",
    					options.pruningCoeff);
    		}
    	}
    	
        decoder.shutdown();

        return eval.UAS();
    }
    
    public double evaluateWithConvergeNum(int converge) throws IOException, CloneNotSupportedException 
    {
    	
    	if (pruner != null) pruner.resetPruningStats();
    	
    	Options options = (Options) this.options.clone();
    	options.numTestConverge = converge;
    	DependencyReader reader = DependencyReader.createDependencyReader(options);
    	reader.startReading(options.testFile);

    	DependencyDecoder decoder = DependencyDecoder.createDependencyDecoder(options);   	
    	
    	Evaluator eval = new Evaluator(options, pipe);
    	
    	DependencyInstance inst = pipe.createInstance(reader);    	
    	while (inst != null) {
    		LocalFeatureData lfd = new LocalFeatureData(inst, this, true, false);
    		GlobalFeatureData gfd = new GlobalFeatureData(lfd); 
    		
            DependencyInstance predInst = decoder.decode(inst, lfd, gfd, false);
            
            eval.add(inst, predInst, false);
    		
    		inst = pipe.createInstance(reader);
    	}
    	
    	reader.close();
    	
        decoder.shutdown();

        return eval.UAS();
    }
}
