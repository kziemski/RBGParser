package parser.io;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import parser.DependencyInstance;
import parser.DependencyPipe;
import parser.Options;

public abstract class DependencyWriter {
	BufferedWriter writer;
	Options options;
	String[] labels;
	boolean first, isLabeled;
	
	public static DependencyWriter createDependencyWriter(Options options, DependencyPipe pipe) {
		String format = options.format;
		if (format.equalsIgnoreCase("CONLL06") || format.equalsIgnoreCase("CONLL-06")) {
			return new Conll06Writer(options, pipe);
		} else if (format.equalsIgnoreCase("CONLLX") || format.equalsIgnoreCase("CONLL-X")) {
			return new Conll06Writer(options, pipe);
		} else if (format.equalsIgnoreCase("CONLL09") || format.equalsIgnoreCase("CONLL-09")) {
			return new Conll09Writer(options, pipe);
		} else {
			System.out.printf("!!!!! Unsupported file format: %s%n", format);
			return new Conll06Writer(options, pipe);
		}
	}
	
	public abstract void writeInstance(DependencyInstance inst) throws IOException;
	
	public void writeDifference(DependencyInstance inst, DependencyInstance predInst) throws IOException {
		
		int length = inst.length;
		String[] forms = inst.forms;
		String[] pos = inst.postags;
		int[] heads = inst.heads;
		int[] pheads = predInst.heads;
		int[] childs = new int[length];
		int[] pchilds = new int[length];
		
		Arrays.fill(childs, -1);
		Arrays.fill(pchilds, -1);
		for (int i = 1; i < length; ++i) {
			if (childs[heads[i]] == -1 || Math.abs(i-heads[i]) < Math.abs(childs[heads[i]]-heads[i]))
				childs[heads[i]] = i;
			if (pchilds[pheads[i]] == -1 || Math.abs(i-pheads[i]) < Math.abs(pchilds[pheads[i]]-pheads[i]))
				pchilds[pheads[i]] = i;
		}
			
		for (int i = 1; i < length; ++i) {
			if (heads[i] != pheads[i] && pos[i].equals("IN"))
				writer.write("gold: "+forms[heads[i]]+" "+forms[childs[i]] + "\tpredict: "+forms[pheads[i]]+" "+forms[pchilds[i]]+"\n");
		}
	}
	
	public void startWriting(String file) throws IOException {
		writer = new BufferedWriter(new FileWriter(file));
		first = true;
		isLabeled = options.learnLabel;
	}
	
	public void close() throws IOException {
		if (writer != null) writer.close();
	}
	
}
