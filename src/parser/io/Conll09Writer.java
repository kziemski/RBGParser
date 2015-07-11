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
	public void writeInstance(DependencyInstance gold, DependencyInstance pred) throws IOException {
		
		if (first) 
			first = false;
		else
			writer.write("\n");
		
		String[] forms = gold.forms;
		String[] lemmas = gold.lemmas;
		String[] cpos = gold.cpostags;
		String[] pos = gold.postags;
		int[] heads = pred.heads;
		int[] labelids = pred.deplbids;
		
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
			if (!puncRegex.matcher(forms[i]).matches() || 
				(gold.heads[i] == pred.heads[i] && gold.deprels[i].equals(labels[pred.deplbids[i]])))
				continue;
			
			writer.write(i + "\t");
			writer.write(forms[i] + "\t");
			writer.write((lemmas != null && lemmas[i] != "" ? lemmas[i] : "_") + "\t");
			writer.write((lemmas != null && lemmas[i] != "" ? lemmas[i] : "_") + "\t");
			writer.write(pos[i] + "\t");
            writer.write(pos[i] + "\t");
			writer.write("_\t");
			writer.write("_\t");
			writer.write(heads[i] + "\t");
			writer.write("_\t");
			writer.write((isLabeled ? labels[labelids[i]] : "_"));
			
			writer.write("_\t");
			writer.write(gold.heads[i] + "\t");
			writer.write("_\t");
			writer.write((isLabeled ? gold.deprels[i] : "_"));
			
			writer.write("\n");
		}
	}

}
