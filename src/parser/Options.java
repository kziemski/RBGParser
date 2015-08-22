package parser;

import java.io.Serializable;


public class Options implements Cloneable, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public enum LearningMode {
		Basic,			// 1st order arc factored model
		Second,			// 2nd order grandparent
	}
		
	public String trainFile = null;
	public String testFile = null;
	public String predFile = null;
	public String unimapFile = null;
	public String outFile = null;
	public boolean train = false;
	public boolean test = false;
	public String wordVectorFile = null;
	public String modelFile = "model.out";
    public String format = "CONLL-X";
    
	public int maxNumSent = -1;
    public int numPretrainIters = 1;
	public int maxNumIters = 10;
	public boolean initTensorWithPretrain = true;
	
	public LearningMode learningMode = LearningMode.Second;
	
	public boolean average = true;
	public double C = 0.01;
	public double gammaLabel = 0;
	public int R = 50, R2 = 30;
	
	// feature set
	public int bits = 30;
	public boolean useGP = true;		// use grandparent
	
	// CoNLL language specific info
	// used only in Full learning mode
	public enum PossibleLang {
		Arabic,
		Bulgarian,
		Chinese,
		Czech,
		Danish,
		Dutch,
		English,
		German,
		Japanese,
		Portuguese,
		Slovene,
		Spanish,
		Swedish,
		Turkish,
		Unknown,
	}
	PossibleLang lang;
	
	final static String langString[] = {"arabic", "bulgarian", "chinese", "czech", "danish", "dutch",
			"english", "german", "japanese", "portuguese", "slovene", "spanish",
			"swedish", "turkish"};
	
	
	public Options() {
		
	}
	
	@Override
	public Object clone() throws CloneNotSupportedException 
	{
		return super.clone();
	}
	
    public void processArguments(String[] args) {
    	
    	for (String arg : args) {
    		if (arg.equals("train")) {
    			train = true;
    		}
    		else if (arg.equals("test")) {
    			test = true;
    		}
            else if (arg.startsWith("average:")) {
            	average = Boolean.parseBoolean(arg.split(":")[1]);
            }
            else if (arg.startsWith("pretrain:")) {
            	initTensorWithPretrain = Boolean.parseBoolean(arg.split(":")[1]);
            }
    		else if (arg.startsWith("train-file:")) {
    			trainFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("test-file:")) {
    			testFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("pred-file:")) {
    			predFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("unimap-file:")) {
    			unimapFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("output-file:")) {
    			outFile = arg.split(":")[1];
    		}
    		else if (arg.startsWith("model-file:")) {
    			modelFile = arg.split(":")[1];
    		}
            else if (arg.startsWith("max-sent:")) {
                maxNumSent = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("C:")) {
            	C = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("gammaLabel:")) {
            	gammaLabel = Double.parseDouble(arg.split(":")[1]);
            }
            else if (arg.startsWith("R:")) {
                R = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("R2:")) {
                R2 = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("word-vector:")) {
            	wordVectorFile = arg.split(":")[1];
            }
            else if (arg.startsWith("iters:")) {
                maxNumIters = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("pre-iters:")) {
                numPretrainIters = Integer.parseInt(arg.split(":")[1]);
            }
            else if (arg.startsWith("bits:")) {
            	bits = Integer.parseInt(arg.split(":")[1]);
            	if (bits < 20) bits = 20;
            	if (bits > 31) bits = 31;
            }
            else if (arg.startsWith("model:")) {
            	String str = arg.split(":")[1];
            	if (str.equals("basic"))
            		learningMode = LearningMode.Basic;
            	else if (str.equals("second"))
            		learningMode = LearningMode.Second;
            }
            else if (arg.startsWith("format:")) {
            	format = arg.split(":")[1];
            }
    	}    	
        
        //gammaLabel = 1.0;

    	switch (learningMode) {
    		case Basic:
    			useGP = false;
    			break;
    		case Second:
    			break;
    		default:
    			break;
    	}
    	
    	lang = findLang(trainFile != null ? trainFile : testFile);
    }
    
    public void printOptions() {
    	System.out.println("------\nFLAGS\n------");
    	System.out.println("train-file: " + trainFile);
    	System.out.println("test-file: " + testFile);
    	System.out.println("pred-file: " + predFile);
    	System.out.println("model-name: " + modelFile);
        System.out.println("output-file: " + outFile);
    	System.out.println("train: " + train);
    	System.out.println("test: " + test);
        System.out.println("iters: " + maxNumIters);
        System.out.println("max-sent: " + maxNumSent);   
        System.out.println("gammaLabel: " + gammaLabel);
        System.out.println("C: " + C);
        System.out.println("R: " + R);
        System.out.println("R2: " + R2);
        System.out.println("word-vector:" + wordVectorFile);
        System.out.println("file format: " + format);
        System.out.println("feature hash bits: " + bits);
        
        System.out.println();
        System.out.println("use grandparent: " + useGP);
        System.out.println("model: " + learningMode.name());

    	System.out.println("------\n");
    }
    
    PossibleLang findLang(String file) {
    	for (PossibleLang lang : PossibleLang.values())
    		if (lang != PossibleLang.Unknown && file.indexOf(langString[lang.ordinal()]) != -1) {
    			return lang;
    		}
    	System.out.println("Warning: unknow language");
    	return PossibleLang.Unknown;
    }
    
}

