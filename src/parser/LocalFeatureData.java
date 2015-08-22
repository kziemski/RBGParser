package parser;

import java.util.Arrays;
import parser.feature.SyntacticFeatureFactory;
import utils.FeatureVector;
import utils.ScoreCollector;

public class LocalFeatureData {
	
	DependencyInstance inst;
	DependencyPipe pipe;
	SyntacticFeatureFactory synFactory;
	Options options;
	Parameters parameters;
	
	final int len;					// sentence length
	final int ntypes;				// number of label types
	final int sizeL;						
	final int rank, rank2;								
	final double gammaL;
	
	FeatureVector[] wordFvs;		// word feature vector
	double[][] wpU, wpV;			// word projections U\phi and V\phi
	double[][] wpU2, wpV2, wpW2;	// word projections U2\phi, V2\phi and W2\phi
	
	int[] numPossibleLabs;
	int[][] possibleLabs;
	
	double[][] f;
	double[][] labScores1;
	double[][][] labScores2;
	
	public LocalFeatureData(DependencyInstance inst,
			DependencyParser parser) 
	{
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
		wpU = new double[len][rank];
		wpV = new double[len][rank];
		if (options.useGP) {
			wpU2 = new double[len][rank2];
			wpV2 = new double[len][rank2];
			wpW2 = new double[len][rank2];
		}
		
		numPossibleLabs = new int[len];
		possibleLabs = new int[len][ntypes];
		
		f = new double[len][ntypes];
		labScores1 = new double[len][ntypes];
		labScores2 = new double[len][ntypes][ntypes];
		
		for (int i = 0; i < len; ++i) {
			wordFvs[i] = synFactory.createWordFeatures(inst, i);
			parameters.projectU(wordFvs[i], wpU[i]);
			parameters.projectV(wordFvs[i], wpV[i]);
			if (options.useGP) {
				parameters.projectU2(wordFvs[i], wpU2[i]);
				parameters.projectV2(wordFvs[i], wpV2[i]);
				parameters.projectW2(wordFvs[i], wpW2[i]);
			}
		}
	}

	private FeatureVector getLabelFeature(int[] heads, int[] types, int mod, int order)
	{
		FeatureVector fv = new FeatureVector(sizeL);
		synFactory.createLabelFeatures(fv, inst, heads, types, mod, order);
		return fv;
	}
	
	private double getLabelScoreTheta(int[] heads, int[] types, int mod, int order)
	{
		ScoreCollector col = new ScoreCollector(parameters.paramsL);
		synFactory.createLabelFeatures(col, inst, heads, types, mod, order);
		return col.score;
	}
	
	void treeDP(int i, DependencyArcList arcLis)
	{
		int st = arcLis.startIndex(i);
		int ed = arcLis.endIndex(i);
		for (int l = st; l < ed ; ++l) {
			int j = arcLis.get(l);
			treeDP(j, arcLis);
		}
		
		for (int pp = 0; pp < numPossibleLabs[i]; ++pp) {
			int p = possibleLabs[i][pp];
			f[i][p] = 0;
			for (int l = st; l < ed ; ++l) {
				int j = arcLis.get(l);
				double best = Double.NEGATIVE_INFINITY;
				for (int qq = 0; qq < numPossibleLabs[j]; ++qq) {
					int q = possibleLabs[j][qq];
					double s = f[j][q] + labScores2[j][q][p];
					if (s > best)
						best = s;
				}
				f[i][p] += best;
			}
		}
	}
	
	void getType(int i, DependencyArcList arcLis, int[] types)
	{
		int p = types[i];
		int st = arcLis.startIndex(i);
		int ed = arcLis.endIndex(i);
		for (int l = st; l < ed ; ++l) {
			int j = arcLis.get(l);
			int bestq = 0;
			double best = Double.NEGATIVE_INFINITY;
			for (int qq = 0; qq < numPossibleLabs[j]; ++qq) {
				int q = possibleLabs[j][qq];
				double s = f[j][q] + labScores2[j][q][p];
				if (s > best) {
					best = s;
					bestq = q;
				}
			}
			types[j] = bestq;
			getType(j, arcLis, types);
		}
	}
	
	public int predictLabels(int[] heads, int[] deplbids, boolean addLoss)
	{
		assert(heads == inst.heads);
		
		int lab0 = addLoss ? 0 : 1;
		int total = 0;
		Arrays.fill(numPossibleLabs, 0);
		for (int mod = 1; mod < len; ++mod) {
			int head = heads[mod];
			for (int p = lab0; p < ntypes; ++p) {
				if (pipe.pruneLabel[inst.postagids[head]][inst.postagids[mod]][p]) {
					possibleLabs[mod][numPossibleLabs[mod]++] = p;
					total++;
				}
			}
		}
		numPossibleLabs[0] = 1;
		possibleLabs[0][0] = inst.deplbids[0];
		
		for (int i = 0; i < len; ++i) {
			Arrays.fill(f[i], Double.NEGATIVE_INFINITY);
			Arrays.fill(labScores1[i], Double.NEGATIVE_INFINITY);
			for (int j = 0; j < ntypes; ++j)
				Arrays.fill(labScores2[i][j], Double.NEGATIVE_INFINITY);
		}
		for (int mod = 1; mod < len; ++mod) {
			int head = heads[mod];
			int dir = head > mod ? 1 : 2;
			int gp = heads[head];
			int pdir = gp > head ? 1 : 2;
			for (int pp = 0; pp < numPossibleLabs[mod]; ++pp) {
				int p = possibleLabs[mod][pp];
				deplbids[mod] = p;
				double s1 = 0;
				if (gammaL > 0)
					s1 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 1);
				if (gammaL < 1)
					s1 += (1-gammaL) * parameters.dotProductL(wpU[head], wpV[mod], p, dir);
				if (options.useGP) {
					for (int qq = 0; qq < numPossibleLabs[head]; ++qq) {
						int q = possibleLabs[head][qq];
						double s2 = 0;
						if (gp != -1) {
							deplbids[head] = q;
							if (gammaL > 0)
								s2 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 2);
							if (gammaL < 1)
								s2 += (1-gammaL) * parameters.dotProduct2L(wpU2[gp], wpV2[head], wpW2[mod], q, p, pdir, dir);
						}
						labScores2[mod][p][q] = s1 + s2 + (addLoss && inst.deplbids[mod] != p ? 1.0 : 0.0);
					}
				}
				else labScores1[mod][p] = s1 + (addLoss && inst.deplbids[mod] != p ? 1.0 : 0.0);
			}
		}
		
		DependencyArcList arcLis = new DependencyArcList(heads);
		treeDP(0, arcLis);
		deplbids[0] = inst.deplbids[0];
		getType(0, arcLis, deplbids);
		
		
//		double s = 0;
//		for (int i = 1; i < len; ++i) {
//			int h = heads[i];
//			s += labScores[i][deplbids[i]][deplbids[h]];
//		}
//		if (Math.abs(s-f[0][1]) > 1e-7)
//			System.out.println(s + " " + f[0][1]);
		
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
        		dlfv.addEntries(getLabelFeature(predDeps, predLabs, mod, 1), -1.0);
    		}
    		if (actLabs[mod] != predLabs[mod] || actLabs[head] != predLabs[head]) {
    			dlfv.addEntries(getLabelFeature(actDeps, actLabs, mod, 2));
    			dlfv.addEntries(getLabelFeature(predDeps, predLabs, mod, 2), -1.0);
    		}
    	}
		
		return dlfv;
	}
	
}
