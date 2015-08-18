package parser;

import java.util.regex.Pattern;

public class Evaluator
{
	int uas, las, tot;
	int whole, nsents;
	
	String[] labels;
	
	public static Pattern puncRegex = Pattern.compile("[\\p{Punct}]+", Pattern.UNICODE_CHARACTER_CLASS);
	//public static Pattern puncRegex = Pattern.compile("[-!\"%&'()*,./:;?@\\[\\]_{}、，。！]+");
	
	public Evaluator(Options options, DependencyPipe pipe)
	{
		uas = las = tot = 0;
		whole = nsents = 0;
		labels = pipe.types;
	}
	
	
	public double UAS()
	{
		return uas/(tot+1e-20);
	}
	
	public double LAS()
	{
		return las/(tot+1e-20);
	}
	
	public double CAS()
	{
		return whole/(nsents+1e-20);
	}
	
	
	public void add(DependencyInstance gold, int[] predDeps, int[] predLabs, boolean evalWithPunc)
	{
		evaluateDependencies(gold, predDeps, predLabs, evalWithPunc);
	}
	
    public void evaluateDependencies(DependencyInstance gold, 
    		int[] predDeps, int[] predLabs, boolean evalWithPunc) 
    {
    	++nsents;
    	int tt = 0, ua = 0, la = 0;
    	for (int i = 1, N = gold.length; i < N; ++i) {

            if (!evalWithPunc)
            	if (puncRegex.matcher(gold.forms[i]).matches()) continue;
            	//if (gold.forms[i].matches("[-!\"%&'()*,./:;?@\\[\\]_{}、]+")) continue;

            ++tt;
    		if (gold.heads[i] == predDeps[i]) {
    			++ua;
    			if (gold.deprels[i].equals(labels[predLabs[i]])) ++la;
    		}
    	
    	}    		
    	
    	tot += tt;
    	uas += ua;
    	las += la;
    	whole += (tt == ua) && (tt == la) ? 1 : 0;
    }
    
}
