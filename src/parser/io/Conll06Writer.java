package parser.io;

import java.io.IOException;

import parser.DependencyInstance;
import parser.DependencyPipe;
import parser.Options;

public class Conll06Writer extends DependencyWriter {
	
	
	public Conll06Writer(Options options, DependencyPipe pipe) {
		this.options = options;
		this.labels = pipe.types;
	}
	
	@Override
	public void writeInstance(DependencyInstance gold, DependencyInstance pred) throws IOException {
		
		//if (first) 
		//	first = false;
		//else
		//	writer.write("\n");
		
		String[] forms = gold.forms;
		String[] lemmas = gold.lemmas;
		String[] cpos = gold.cpostags;
		String[] pos = gold.postags;
		int[] heads = pred.heads;
		int[] labelids = pred.deplbids;
		
	    // 3 eles ele pron pron-pers M|3P|NOM 4 SUBJ _ _
	    // ID FORM LEMMA COURSE-POS FINE-POS FEATURES HEAD DEPREL PHEAD PDEPREL
		for (int i = 1, N = gold.length; i < N; ++i) {
			writer.write(i + "\t");
			writer.write(forms[i] + "\t");
			writer.write((lemmas != null && lemmas[i] != "" ? lemmas[i] : "_") + "\t");
			writer.write(cpos[i] + "\t");
			writer.write(pos[i] + "\t");
			writer.write("_\t");
			writer.write(heads[i] + "\t");
			writer.write((isLabeled ? labels[labelids[i]] : "_") + "\t_\t_");
			writer.write("\n");
		}
		
		writer.write("\n");
	}

}
