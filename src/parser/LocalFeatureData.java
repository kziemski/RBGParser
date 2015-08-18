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
	
	double[][] f;
	double[][][] labScores;
	
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
		
		f = new double[len][ntypes];
		labScores = new double[len][ntypes][ntypes];
		
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
	
	void treeDP(int i, DependencyArcList arcLis, int lab0)
	{
		Arrays.fill(f[i], 0);
		int st = arcLis.startIndex(i);
		int ed = arcLis.endIndex(i);
		for (int l = st; l < ed ; ++l) {
			int j = arcLis.get(l);
			treeDP(j, arcLis, lab0);
			for (int p = lab0; p < ntypes; ++p) {
				double best = Double.NEGATIVE_INFINITY;
				for (int q = lab0; q < ntypes; ++q) {
					double s = f[j][q] + labScores[j][q][p];
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
			double best = Double.NEGATIVE_INFINITY;
			for (int q = lab0; q < ntypes; ++q) {
				double s = f[j][q] + labScores[j][q][p];
				if (s > best) {
					best = s;
					bestq = q;
				}
			}
			types[j] = bestq;
			getType(j, arcLis, types, lab0);
		}
	}
	
	public void predictLabels(int[] heads, int[] deplbids, boolean addLoss)
	{
		assert(heads == inst.heads);
		int lab0 = addLoss ? 0 : 1;
		
		for (int mod = 1; mod < len; ++mod) {
			int head = heads[mod];
			int dir = head > mod ? 1 : 2;
			int gp = heads[head];
			int pdir = gp > head ? 1 : 2;
			for (int p = lab0; p < ntypes; ++p) {
				if (pipe.pruneLabel[inst.postagids[head]][inst.postagids[mod]][p]) {
					deplbids[mod] = p;
					double s1 = 0;
					if (gammaL > 0)
						s1 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 1);
					if (gammaL < 1)
						s1 += (1-gammaL) * parameters.dotProductL(wpU[head], wpV[mod], p, dir);
					for (int q = lab0; q < ntypes; ++q) {
						double s2 = 0;
						if (options.useGP && gp != -1) {
							if (pipe.pruneLabel[inst.postagids[gp]][inst.postagids[head]][q]) {
								deplbids[head] = q;
								if (gammaL > 0)
									s2 += gammaL * getLabelScoreTheta(heads, deplbids, mod, 2);
								if (gammaL < 1)
									s2 += (1-gammaL) * parameters.dotProduct2L(wpU2[gp], wpV2[head], wpW2[mod], q, p, pdir, dir);
							}
							else s2 = Double.NEGATIVE_INFINITY;
						}
						labScores[mod][p][q] = s1 + s2 + (addLoss && inst.deplbids[mod] != p ? 1.0 : 0.0);
					}
				}
				else Arrays.fill(labScores[mod][p], Double.NEGATIVE_INFINITY);
			}
		}
		
		DependencyArcList arcLis = new DependencyArcList(heads);
		treeDP(0, arcLis, lab0);
		deplbids[0] = inst.deplbids[0];
		getType(0, arcLis, deplbids, lab0);
		
		
//		double s = 0;
//		for (int i = 1; i < len; ++i) {
//			int h = heads[i];
//			s += labScores[i][deplbids[i]][deplbids[h]];
//		}
//		if (Math.abs(s-f[0][1]) > 1e-7)
//			System.out.println(s + " " + f[0][1]);
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
