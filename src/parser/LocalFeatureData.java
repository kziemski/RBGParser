package parser;

import java.util.Arrays;
import parser.feature.SyntacticFeatureFactory;
import utils.FeatureVector;
import utils.ScoreCollector;

public class LocalFeatureData {
    
    public static long calcScoreTime = 0;
    public static long calcDpTime = 0;
    public static long memAllocTime = 0;
    public static long projTime = 0;

	DependencyInstance inst;
	DependencyPipe pipe;
	SyntacticFeatureFactory synFactory;
	Options options;
	Parameters parameters;
	
	final int len;					// sentence length
	final int ntypes;				// number of label types
	final int sizeL;						
	final int rank, rank2;								
	final float gammaL;
	
	FeatureVector[] wordFvs;		// word feature vector
	float[][] wpU, wpV;			// word projections U\phi and V\phi
	float[][] wpU2, wpV2, wpW2;	// word projections U2\phi, V2\phi and W2\phi
	
	float[][] f;
	float[][][] labScores;
	
	public LocalFeatureData(DependencyInstance inst,
			DependencyParser parser) 
	{
        long start = System.currentTimeMillis();

		this.inst = inst;
		pipe = parser.pipe;
		synFactory = pipe.synFactory;
		options = parser.options;
		parameters = parser.parameters;
		
		len = inst.length;
		ntypes = pipe.types.length;
		rank = options.R;
		rank2 = options.R2;
		sizeL = synFactory.numLabeledArcFeats+1;
		gammaL = options.gammaLabel;
		
		wordFvs = new FeatureVector[len];
		wpU = new float[len][rank];
		wpV = new float[len][rank];
		if (options.useGP) {
			wpU2 = new float[len][rank2];
			wpV2 = new float[len][rank2];
			wpW2 = new float[len][rank2];
		}
		
		f = new float[len][ntypes];
		labScores = new float[len][ntypes][ntypes];
	    
        memAllocTime += System.currentTimeMillis()-start;
        

		for (int i = 0; i < len; ++i) {
			wordFvs[i] = synFactory.createWordFeatures(inst, i);

            start = System.currentTimeMillis();
			parameters.projectU(wordFvs[i], wpU[i]);
			parameters.projectV(wordFvs[i], wpV[i]);
			if (options.useGP) {
				parameters.projectU2(wordFvs[i], wpU2[i]);
				parameters.projectV2(wordFvs[i], wpV2[i]);
				parameters.projectW2(wordFvs[i], wpW2[i]);
			}
            projTime += System.currentTimeMillis()-start;
		}
        
   	}

	private FeatureVector getLabelFeature(int[] heads, int[] types, int mod, int order)
	{
		FeatureVector fv = new FeatureVector(sizeL);
		synFactory.createLabelFeatures(fv, inst, heads, types, mod, order);
		return fv;
	}
	
	private float getLabelScoreTheta(int[] heads, int[] types, int mod, int order)
	{
		ScoreCollector col = new ScoreCollector(parameters.paramsL);
		synFactory.createLabelFeatures(col, inst, heads, types, mod, order);
		return col.score;
	}
	
	void treeDP(int i, DependencyArcList arcLis, int lab0)
	{
		Arrays.fill(f[i], 0);
		int st = arcLis.startIndex(i);
		int ed = arcLis.endIndex(i);
		for (int l = st; l < ed ; ++l) {
			int j = arcLis.get(l);
			treeDP(j, arcLis, lab0);
			for (int p = lab0; p < ntypes; ++p) {
				float best = Float.NEGATIVE_INFINITY;
				for (int q = lab0; q < ntypes; ++q) {
					float s = f[j][q] + labScores[j][q][p];
					if (s > best)
						best = s;
				}
				f[i][p] += best;
			}
		}
	}
	
	void getType(int i, DependencyArcList arcLis, int[] types, int lab0)
	{
		int p = types[i];
		int st = arcLis.startIndex(i);
		int ed = arcLis.endIndex(i);
		for (int l = st; l < ed ; ++l) {
			int j = arcLis.get(l);
			int bestq = 0;
			float best = Float.NEGATIVE_INFINITY;
			for (int q = lab0; q < ntypes; ++q) {
				float s = f[j][q] + labScores[j][q][p];
				if (s > best) {
					best = s;
					bestq = q;
				}
			}
			types[j] = bestq;
			getType(j, arcLis, types, lab0);
		}
	}
	
	public void predictLabelsDP(int[] heads, int[] deplbids, boolean addLoss, DependencyArcList arcLis) {

        long start = System.currentTimeMillis();
        
		int lab0 = addLoss ? 0 : 1;
		
		for (int mod = 1; mod < len; ++mod) {
			int head = heads[mod];
			int dir = head > mod ? 1 : 2;
			int gp = heads[head];
			int pdir = gp > head ? 1 : 2;
			for (int p = lab0; p < ntypes; ++p) {
				if (pipe.pruneLabel[inst.postagids[head]][inst.postagids[mod]][p]) {
					deplbids[mod] = p;
					float s1 = 0;
					if (gammaL > 0)
						s1 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 1);
					if (gammaL < 1)
						s1 += (1-gammaL) * parameters.dotProductL(wpU[head], wpV[mod], p, dir);
					for (int q = lab0; q < ntypes; ++q) {
						float s2 = 0;
						if (options.useGP && gp != -1) {
							if (pipe.pruneLabel[inst.postagids[gp]][inst.postagids[head]][q]) {
								deplbids[head] = q;
								if (gammaL > 0)
									s2 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 2);
								if (gammaL < 1)
									s2 += (1-gammaL) * parameters.dotProduct2L(wpU2[gp], wpV2[head], wpW2[mod], q, p, pdir, dir);
							}
							else s2 = Float.NEGATIVE_INFINITY;
						}
						labScores[mod][p][q] = s1 + s2 + (addLoss && inst.deplbids[mod] != p ? 1.0f : 0.0f);
					}
				}
				else Arrays.fill(labScores[mod][p], Float.NEGATIVE_INFINITY);
			}
		}
	    
        calcScoreTime += (System.currentTimeMillis()-start);

        start = System.currentTimeMillis();
		
        treeDP(0, arcLis, lab0);
		deplbids[0] = inst.deplbids[0];
		getType(0, arcLis, deplbids, lab0);

	    calcDpTime += (System.currentTimeMillis()-start);
		
//		float s = 0;
//		for (int i = 1; i < len; ++i) {
//			int h = heads[i];
//			s += labScores[i][deplbids[i]][deplbids[h]];
//		}
//		if (Math.abs(s-f[0][1]) > 1e-7)
//			System.out.println(s + " " + f[0][1]);
	}
	
	void predictLabelsGreedy(int i, int[] heads, int[] types, boolean addLoss, DependencyArcList arcLis)
	{
		int st = arcLis.startIndex(i);
		int ed = arcLis.endIndex(i);
		for (int l = st; l < ed ; ++l) {
			int j = arcLis.get(l);
			predictLabelsGreedy(j, heads, types, addLoss, arcLis);
		}
		
		int k = heads[i];
		if (k == -1)
			return;
		int dir = k > i ? 1 : 2;
		
		int lab0 = addLoss ? 0 : 1;
		int bestp = 0;
		float best = Float.NEGATIVE_INFINITY;
		for (int p = lab0; p < ntypes; ++p) {
			if (pipe.pruneLabel[inst.postagids[k]][inst.postagids[i]][p]) {
				types[i] = p;
				float s = 0;
				if (gammaL > 0)
					s += gammaL * getLabelScoreTheta(heads, types, i, 1);
				if (gammaL < 1)
					s += (1-gammaL) * parameters.dotProductL(wpU[k], wpV[i], p, dir);
				for (int l = st; l < ed ; ++l) {
					int j = arcLis.get(l);
					int cdir = i > j ? 1 : 2;
					if (gammaL > 0)
						s += gammaL * getLabelScoreTheta(heads, types, j, 2);
					if (gammaL < 1)
						s += (1-gammaL) * parameters.dotProduct2L(wpU2[k], wpV2[i], wpW2[j], p, types[j], dir, cdir);
				}
				if (s > best) {
					best = s;
					bestp = p;
				}
			}
		}
		types[i] = bestp;
	}
	
	public int predictLabels(int[] heads, int[] deplbids, boolean addLoss)
	{
		DependencyArcList arcLis = new DependencyArcList(heads);
		
		predictLabelsDP(heads, deplbids, addLoss, arcLis);
		//predictLabelsGreedy(0, heads, deplbids, addLoss, arcLis);
		
		int lab0 = addLoss ? 0 : 1;
		int total = 0;
		for (int mod = 1; mod < len; ++mod) {
			int head = heads[mod];
			for (int p = lab0; p < ntypes; ++p) {
				if (pipe.pruneLabel[inst.postagids[head]][inst.postagids[mod]][p]) {
					total++;
				}
			}
		}
		return total;
	}
	
	public FeatureVector getLabeledFeatureDifference(DependencyInstance gold, 
			int[] predDeps, int[] predLabs)
	{
		assert(gold.heads == predDeps);
		
		FeatureVector dlfv = new FeatureVector(sizeL);

    	int N = inst.length;
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	
    	for (int mod = 1; mod < N; ++mod) {
    		int head = actDeps[mod];
    		if (actLabs[mod] != predLabs[mod]) {
    			dlfv.addEntries(getLabelFeature(actDeps, actLabs, mod, 1));
        		dlfv.addEntries(getLabelFeature(predDeps, predLabs, mod, 1), -1.0f);
    		}
    		if (actLabs[mod] != predLabs[mod] || actLabs[head] != predLabs[head]) {
    			dlfv.addEntries(getLabelFeature(actDeps, actLabs, mod, 2));
    			dlfv.addEntries(getLabelFeature(predDeps, predLabs, mod, 2), -1.0f);
    		}
    	}
		
		return dlfv;
	}
	
}
