package parser.io;

import java.io.IOException;
import java.util.regex.Pattern;

import parser.DependencyInstance;
import parser.DependencyPipe;
import parser.Options;

public class Conll09Writer extends DependencyWriter {
	
	public static Pattern puncRegex = Pattern.compile("[\\p{Punct}]+", Pattern.UNICODE_CHARACTER_CLASS);
	
	public Conll09Writer(Options options, DependencyPipe pipe) {
		this.options = options;
		this.labels = pipe.types;
	}
	
	@Override
	public void writeInstance(DependencyInstance gold, int[] predDeps, int[] predLabs) throws IOException {
		
		if (first) 
			first = false;
		else
			writer.write("\n");
		
		String[] forms = gold.forms;
		String[] lemmas = gold.lemmas;
		String[] cpos = gold.cpostags;
		String[] pos = gold.postags;
		
	    /*
	     * CoNLL 2009 format:
		    0 ID
		    1 FORM
		    2 LEMMA (not used)
		    3 PLEMMA 
		    4 POS (not used)
		    5 PPOS   
		    6 FEAT (not used)
		    7 PFEAT  
		    8 HEAD
		    9 PHEAD 
		    10 DEPREL 
		    11 PDEPREL 
		    12 FILLPRED 
		    13 PRED
		    14... APREDn
	   	*/
	    
	    // 11  points  point   point   NNS NNS _   _   8   8   PMOD    PMOD    Y   point.02    _   _   _   _	    
	    // 1   杩�  杩�  杩�  DT  DT  _   _   6   4   DMOD    ADV _   _   _   _   _   _
		
		for (int i = 1, N = gold.length; i < N; ++i) {
			writer.write(i + "\t");
			writer.write(forms[i] + "\t");
			writer.write((lemmas != null && lemmas[i] != "" ? lemmas[i] : "_") + "\t");
			writer.write((lemmas != null && lemmas[i] != "" ? lemmas[i] : "_") + "\t");
			writer.write(pos[i] + "\t");
            writer.write(pos[i] + "\t");
			writer.write("_\t");
			writer.write("_\t");
			writer.write(predDeps[i] + "\t");
			writer.write("_\t");
			writer.write(labels[predLabs[i]]);
			writer.write("\n");
		}
	}

}
