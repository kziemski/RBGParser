package parser.feature;


import static parser.feature.FeatureTemplate.Arc.*;
import static parser.feature.FeatureTemplate.Word.*;
import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.hash.TLongHashSet;

import java.io.Serializable;

import parser.DependencyInstance;
import parser.LowRankTensor;
import parser.Options;
import parser.Parameters;
import parser.DependencyInstance.SpecialPos;
import utils.Alphabet;
import utils.Collector;
import utils.FeatureVector;
import utils.Utils;

public class SyntacticFeatureFactory implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public int TOKEN_START = 1;
	public int TOKEN_END = 2;
	public int TOKEN_MID = 3;

	// for punc
	public int TOKEN_QUOTE = 4;
	public int TOKEN_RRB = 5;
	public int TOKEN_LRB = 6;
	
	public Options options;
	
	public double[][] wordVectors = null;
	public double[] unknownWv = null;
	
	public int tagNumBits, wordNumBits, depNumBits, disNumBits = 4;
	public int flagBits;
	
	public int ccDepType;
	
	public final int numLabeledArcFeats;
	public int numWordFeats;			// number of word features
	
	private boolean stoppedGrowth;
	private transient TLongHashSet featureHashSet;
	private Alphabet wordAlphabet;		// the alphabet of word features (e.g. \phi_h, \phi_m)
	
	public SyntacticFeatureFactory(Options options)
	{
		this.options = options;
		
		wordAlphabet = new Alphabet();
		
		stoppedGrowth = false;
		featureHashSet = new TLongHashSet(100000);
		
		numWordFeats = 0;
		numLabeledArcFeats = (int ) ((1L << (options.bits-2))-1);
	}
	
	public void closeAlphabets()
	{
		wordAlphabet.stopGrowth();
		stoppedGrowth = true;
	}
	
	public void checkCollisions()
	{
		long[] codes = featureHashSet.toArray();
		int nfeats = codes.length;
		int ncols = 0;
		TIntHashSet idhash = new TIntHashSet();
		for (long code : codes) {
			int id = hashcode2int(code) & numLabeledArcFeats;
			if (idhash.contains(id))
				++ncols;
			else
				idhash.add(id);
		}
		System.out.printf("Hash collision: %.4f%% (%d / %d)%n",
				ncols / (nfeats + 1e-30) * 100,
				ncols,
				nfeats
			);		
	}
	
    public void initFeatureAlphabets(DependencyInstance inst) 
    {
    	LazyCollector col = new LazyCollector();
    	
        int n = inst.length;
        
        for (int i = 0; i < n; ++i)
            createWordFeatures(inst, i);
    	
		for (int m = 1; m < n; ++m) {
			createLabelFeatures(col, inst, inst.heads, inst.deplbids, m, 0);
		}
    }
    
    /************************************************************************
     * Region start #
     * 
     *  Functions that create feature vectors of a specific word in the 
     *  sentence
     *  
     ************************************************************************/
    
    public FeatureVector createWordFeatures(DependencyInstance inst, int i) 
    {
    	
    	int[] pos = inst.postagids;
        int[] posA = inst.cpostagids;
        int[] toks = inst.formids;
    	int[] lemma = inst.lemmaids;
        
    	int p0 = pos[i];
    	int pp = i > 0 ? pos[i-1] : TOKEN_START;
    	int pn = i < pos.length-1 ? pos[i+1] : TOKEN_END;
    	
    	int c0 = posA[i];
    	int cp = i > 0 ? posA[i-1] : TOKEN_START;
    	int cn = i < posA.length-1 ? posA[i+1] : TOKEN_END;
    	
        int w0 = toks[i];
        int wp = i == 0 ? TOKEN_START : toks[i-1];
    	int wn = i == inst.length - 1 ? TOKEN_END : toks[i+1];
    	
        int l0 = 0, lp = 0, ln = 0;
        if (lemma != null) {
        	l0 =  lemma[i];
        	lp = i == 0 ? TOKEN_START : lemma[i-1];
	    	ln = i == inst.length - 1 ? TOKEN_END : lemma[i+1];
        }
        
        FeatureVector fv = new FeatureVector(wordAlphabet.size());
    	
    	long code = 0;
        
    	code = createWordCodeP(WORDFV_BIAS, 0);
    	addWordFeature(code, fv);

    	code = createWordCodeW(WORDFV_W0, w0);
    	addWordFeature(code, fv);
    	code = createWordCodeW(WORDFV_Wp, wp);
    	addWordFeature(code, fv);
    	code = createWordCodeW(WORDFV_Wn, wn);
    	addWordFeature(code, fv);

    	
		if (l0 != 0) {
    		code = createWordCodeW(WORDFV_W0, l0);
    		addWordFeature(code, fv);
	    	code = createWordCodeW(WORDFV_Wp, lp);
	    	addWordFeature(code, fv);
	    	code = createWordCodeW(WORDFV_Wn, ln);
	    	addWordFeature(code, fv);
		}
		
		code = createWordCodeP(WORDFV_P0, p0);
    	addWordFeature(code, fv);
    	code = createWordCodeP(WORDFV_Pp, pp);
    	addWordFeature(code, fv);
    	code = createWordCodeP(WORDFV_Pn, pn);
    	addWordFeature(code, fv);
    	
    	code = createWordCodeP(WORDFV_P0, c0);
		addWordFeature(code, fv);
		code = createWordCodeP(WORDFV_Pp, cp);
		addWordFeature(code, fv);
		code = createWordCodeP(WORDFV_Pn, cn);
		addWordFeature(code, fv);
		
		code = createWordCodePP(WORDFV_PpP0, pp, p0);
    	addWordFeature(code, fv);
    	code = createWordCodePP(WORDFV_P0Pn, p0, pn);
    	addWordFeature(code, fv);
    	code = createWordCodePP(WORDFV_PpPn, pp, pn);
    	addWordFeature(code, fv);
    	code = createWordCodePPP(WORDFV_PpP0Pn, pp, p0, pn);
    	addWordFeature(code, fv);
    	
		code = createWordCodePP(WORDFV_PpP0, cp, c0);
    	addWordFeature(code, fv);
    	code = createWordCodePP(WORDFV_P0Pn, c0, cn);
    	addWordFeature(code, fv);
    	code = createWordCodePP(WORDFV_PpPn, cp, cn);
    	addWordFeature(code, fv);
    	code = createWordCodePPP(WORDFV_PpP0Pn, cp, c0, cn);
    	addWordFeature(code, fv);
		
    	code = createWordCodeWP(WORDFV_W0P0, w0, p0);
		addWordFeature(code, fv);
		
//		code = createWordCodeWP(WORDFV_W0Pp, w0, pp);
//		addWordFeature(code, fv);
//		
//		code = createWordCodeWP(WORDFV_W0Pn, w0, pn);
//		addWordFeature(code, fv);
		
//		code = createWordCodeWP(WORDFV_WpPp, wp, pp);
//		addWordFeature(code, fv);
//		
//		code = createWordCodeWP(WORDFV_WnPn, wn, pn);
//		addWordFeature(code, fv);
		
		code = createWordCodeWP(WORDFV_W0P0, w0, c0);
		addWordFeature(code, fv);
		
//		code = createWordCodeWP(WORDFV_W0Pp, w0, cp);
//		addWordFeature(code, fv);
//		
//		code = createWordCodeWP(WORDFV_W0Pn, w0, cn);
//		addWordFeature(code, fv);
		
//		code = createWordCodeWP(WORDFV_WpPp, wp, cp);
//		addWordFeature(code, fv);
//		
//		code = createWordCodeWP(WORDFV_WnPn, wn, cn);
//		addWordFeature(code, fv);
		
		if (l0 != 0) {
			code = createWordCodeWP(WORDFV_W0P0, l0, p0);
			addWordFeature(code, fv);
			
//			code = createWordCodeWP(WORDFV_W0Pp, l0, pp);
//			addWordFeature(code, fv);
//			
//			code = createWordCodeWP(WORDFV_W0Pn, l0, pn);
//			addWordFeature(code, fv);
			
//			code = createWordCodeWP(WORDFV_WpPp, lp, pp);
//			addWordFeature(code, fv);
//			
//			code = createWordCodeWP(WORDFV_WnPn, ln, pn);
//			addWordFeature(code, fv);
			
			code = createWordCodeWP(WORDFV_W0P0, l0, c0);
			addWordFeature(code, fv);
			
			code = createWordCodeWP(WORDFV_W0Pp, l0, cp);
			addWordFeature(code, fv);
			
			code = createWordCodeWP(WORDFV_W0Pn, l0, cn);
			addWordFeature(code, fv);
			
			code = createWordCodeWP(WORDFV_WpPp, lp, cp);
			addWordFeature(code, fv);
			
			code = createWordCodeWP(WORDFV_WnPn, ln, cn);
			addWordFeature(code, fv);
		}
    	
		int[][] feats = inst.featids;
		if (feats[i] != null) {
    		for (int u = 0; u < feats[i].length; ++u) {
    			int f = feats[i][u];
    			
    			code = createWordCodeP(WORDFV_P0, f);
    			addWordFeature(code, fv);
    			
                if (l0 != 0) {
                	code = createWordCodeWP(WORDFV_W0P0, l0, f);
                	addWordFeature(code, fv);
                }
                
            }
		}
    		    
    	if (wordVectors != null) {
    		addWordVectorFeatures(inst, i, 0, fv);
    		//addWordVectorFeatures(inst, i, -1, fv);
    		//addWordVectorFeatures(inst, i, 1, fv);	
    	}
    	
    	return fv;
    }
    
    public void addWordVectorFeatures(DependencyInstance inst, int i, int dis, FeatureVector fv) {
    	
    	int d = Utils.getBinnedDistance(dis);
    	double [] v = unknownWv;
    	int pos = i + dis;
    	
    	if (pos >= 0 && pos < inst.length) {
    		int wvid = inst.wordVecIds[pos];
    		if (wvid > 0) v = wordVectors[wvid];
    	}
    	
		//if (v == unknownWv) ++wvMiss; else ++wvHit;
		
		if (v != null) {
			for (int j = 0; j < v.length; ++j) {
				long code = createWordCodeW(WORDFV_EMB, j);
				addWordFeature(code | d, (float) v[j], fv);
			}
		}
    }

    /************************************************************************
     *  Region end #
     ************************************************************************/
    
    
    
    /************************************************************************
     * Region start #
     * 
     *  Functions that create feature vectors for labeled arcs
     *  
     ************************************************************************/
    
    public void createLabelFeatures(Collector fv, DependencyInstance inst,
    		int[] heads, int[] types, int mod, int order)
    {
    	int head = heads[mod];
    	int type = types[mod];
    	if (order != 2)
    		createLabeledArcFeatures(fv, inst, head, mod, type);
    	
    	int gp = heads[head];
    	int ptype = types[head];
    	if (order != 1 && options.useGP && gp != -1) {
    		createLabeledGPCFeatureVector(fv, inst, gp, head, mod, type, ptype);
    	}
    }
    
    public void createLabeledArcFeatures(Collector fv, DependencyInstance inst, int h, int c, int type) 
    {
    	int attDist;
    	//if (preTrain)
    		attDist = h > c ? 1 : 2;
    	//else attDist = getBinnedDistance(h-c);   
    		
    	addBasic1OFeatures(fv, inst, h, c, attDist, type);
    	
    	addCore1OPosFeatures(fv, inst, h, c, attDist, type);
    		    		
    	addCore1OBigramFeatures(fv, inst.formids[h], inst.postagids[h], 
    			inst.formids[c], inst.postagids[c], attDist, type);
    	    		
		if (inst.lemmaids != null)
			addCore1OBigramFeatures(fv, inst.lemmaids[h], inst.postagids[h], 
					inst.lemmaids[c], inst.postagids[c], attDist, type);
		
		addCore1OBigramFeatures(fv, inst.formids[h], inst.cpostagids[h], 
    			inst.formids[c], inst.cpostagids[c], attDist, type);
		
		if (inst.lemmaids != null)
			addCore1OBigramFeatures(fv, inst.lemmaids[h], inst.cpostagids[h], 
					inst.lemmaids[c], inst.cpostagids[c], attDist, type);
    	
    	if (inst.featids[h] != null && inst.featids[c] != null) {
    		for (int i = 0, N = inst.featids[h].length; i < N; ++i)
    			for (int j = 0, M = inst.featids[c].length; j < M; ++j) {
    				
    				addCore1OBigramFeatures(fv, inst.formids[h], inst.featids[h][i], 
    						inst.formids[c], inst.featids[c][j], attDist, type);
    				
    				if (inst.lemmas != null)
    					addCore1OBigramFeatures(fv, inst.lemmaids[h], inst.featids[h][i], 
    							inst.lemmaids[c], inst.featids[c][j], attDist, type);
    			}
    	}

    }
    
    public void addBasic1OFeatures(Collector fv, DependencyInstance inst, 
    		int h, int m, int attDist, int type) 
    {
    	
    	long code = 0; 			// feature code
    	
    	int[] forms = inst.formids, lemmas = inst.lemmaids, postags = inst.postagids;
    	int[] cpostags = inst.cpostagids;
    	int[][] feats = inst.featids;
    	
    	int tid = type << 4;

    	code = createArcCodeW(CORE_HEAD_WORD, forms[h]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	    	    	
    	code = createArcCodeW(CORE_MOD_WORD, forms[m]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeWW(HW_MW, forms[h], forms[m]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	int pHF = h == 0 ? TOKEN_START : (h == m+1 ? TOKEN_MID : forms[h-1]);
    	int nHF = h == inst.length - 1 ? TOKEN_END : (h+1 == m ? TOKEN_MID : forms[h+1]);
    	int pMF = m == 0 ? TOKEN_START : (m == h+1 ? TOKEN_MID : forms[m-1]);
    	int nMF = m == inst.length - 1 ? TOKEN_END : (m+1 == h ? TOKEN_MID : forms[m+1]);
    	
    	code = createArcCodeW(CORE_HEAD_pWORD, pHF) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeW(CORE_HEAD_nWORD, nHF) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeW(CORE_MOD_pWORD, pMF) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeW(CORE_MOD_nWORD, nMF) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
	
		
    	code = createArcCodeP(CORE_HEAD_POS, postags[h]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeP(CORE_HEAD_POS, cpostags[h]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeP(CORE_MOD_POS, postags[m]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeP(CORE_MOD_POS, cpostags[m]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodePP(HP_MP, postags[h], postags[m]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodePP(HP_MP, cpostags[h], cpostags[m]) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	     	
    	if (lemmas != null) {
    		code = createArcCodeW(CORE_HEAD_WORD, lemmas[h]) | tid;
        	addLabeledArcFeature(code, fv);
        	addLabeledArcFeature(code | attDist, fv);
        	
    		code = createArcCodeW(CORE_MOD_WORD, lemmas[m]) | tid;
        	addLabeledArcFeature(code, fv);
        	addLabeledArcFeature(code | attDist, fv);
        	
        	code = createArcCodeWW(HW_MW, lemmas[h], lemmas[m]) | tid;
        	addLabeledArcFeature(code, fv);
        	addLabeledArcFeature(code | attDist, fv);
        	
	    	int pHL = h == 0 ? TOKEN_START : (h == m+1 ? TOKEN_MID : lemmas[h-1]);
	    	int nHL = h == inst.length - 1 ? TOKEN_END : (h+1 == m ? TOKEN_MID : lemmas[h+1]);
	    	int pML = m == 0 ? TOKEN_START : (m == h+1 ? TOKEN_MID : lemmas[m-1]);
	    	int nML = m == inst.length - 1 ? TOKEN_END : (m+1 == h ? TOKEN_MID : lemmas[m+1]);
	    	
	    	code = createArcCodeW(CORE_HEAD_pWORD, pHL) | tid;
	    	addLabeledArcFeature(code, fv);
	    	addLabeledArcFeature(code | attDist, fv);
	    	
	    	code = createArcCodeW(CORE_HEAD_nWORD, nHL) | tid;
	    	addLabeledArcFeature(code, fv);
	    	addLabeledArcFeature(code | attDist, fv);
	    	
	    	code = createArcCodeW(CORE_MOD_pWORD, pML) | tid;
	    	addLabeledArcFeature(code, fv);
	    	addLabeledArcFeature(code | attDist, fv);
	    	
	    	code = createArcCodeW(CORE_MOD_nWORD, nML) | tid;
	    	addLabeledArcFeature(code, fv);
	    	addLabeledArcFeature(code | attDist, fv);
    	}
    	
		if (feats[h] != null)
			for (int i = 0, N = feats[h].length; i < N; ++i) {
				code = createArcCodeP(CORE_HEAD_POS, feats[h][i]) | tid;
	        	addLabeledArcFeature(code, fv);
	        	addLabeledArcFeature(code | attDist, fv);
			}
		
		if (feats[m] != null)
			for (int i = 0, N = feats[m].length; i < N; ++i) {
				code = createArcCodeP(CORE_MOD_POS, feats[m][i]) | tid;
	        	addLabeledArcFeature(code, fv);
	        	addLabeledArcFeature(code | attDist, fv);
			}
		
		if (feats[h] != null && feats[m] != null) {
			for (int i = 0, N = feats[h].length; i < N; ++i)
				for (int j = 0, M = feats[m].length; j < M; ++j) {
			    	code = createArcCodePP(HP_MP, feats[h][i], feats[m][j]) | tid;
			    	addLabeledArcFeature(code, fv);
			    	addLabeledArcFeature(code | attDist, fv);
				}
		}
		
		if (wordVectors != null) {
			
			int wvid = inst.wordVecIds[h];
			double [] v = wvid > 0 ? wordVectors[wvid] : unknownWv;
			if (v != null) {
				for (int i = 0; i < v.length; ++i) {
					code = createArcCodeW(HEAD_EMB, i) | tid;
					addLabeledArcFeature(code, (float) v[i], fv);
					addLabeledArcFeature(code | attDist, (float) v[i], fv);
				}
			}
			
			wvid = inst.wordVecIds[m];
			v = wvid > 0 ? wordVectors[wvid] : unknownWv;
			if (v != null) {
				for (int i = 0; i < v.length; ++i) {
					code = createArcCodeW(MOD_EMB, i) | tid;
					addLabeledArcFeature(code, (float) v[i], fv);
					addLabeledArcFeature(code | attDist, (float) v[i], fv);
				}
			}
		}
    }
    
    public void addCore1OPosFeatures(Collector fv, DependencyInstance inst, 
    		int h, int c, int attDist, int type) 
    {  	
    	
    	int[] pos = inst.postagids;
    	int[] posA = inst.cpostagids;
	
    	int tid = type << 4;
    	
    	int pHead = pos[h], pHeadA = posA[h];
    	int pMod = pos[c], pModA = posA[c];
    	int pHeadLeft = h > 0 ? (h-1 == c ? TOKEN_MID : pos[h-1]) : TOKEN_START;    	
    	int pModRight = c < pos.length-1 ? (c+1 == h ? TOKEN_MID : pos[c+1]) : TOKEN_END;
    	int pHeadRight = h < pos.length-1 ? (h+1 == c ? TOKEN_MID: pos[h+1]) : TOKEN_END;
    	int pModLeft = c > 0 ? (c-1 == h ? TOKEN_MID : pos[c-1]) : TOKEN_START;
    	int pHeadLeftA = h > 0 ? (h-1 == c ? TOKEN_MID : posA[h-1]) : TOKEN_START;    	
    	int pModRightA = c < posA.length-1 ? (c+1 == h ? TOKEN_MID : posA[c+1]) : TOKEN_END;
    	int pHeadRightA = h < posA.length-1 ? (h+1 == c ? TOKEN_MID: posA[h+1]) : TOKEN_END;
    	int pModLeftA = c > 0 ? (c-1 == h ? TOKEN_MID : posA[c-1]) : TOKEN_START;
    	
    	    	
    	long code = 0;
    	
    	// feature posR posMid posL
    	int small = h < c ? h : c;
    	int large = h > c ? h : c;
    	
    	SpecialPos[] spos = inst.specialPos;
    	int num_verb = 0, num_conj = 0, num_punc = 0;
    	for(int i = small+1; i < large; ++i)
    		if (spos[i] == SpecialPos.C)
    			++num_conj;
    		else if (spos[i] == SpecialPos.V)
    			++num_verb;
    		else if (spos[i] == SpecialPos.PNX)
    			++num_punc;
    	int max_num = 15 < (1 << tagNumBits) ? 15 : (1 << tagNumBits)-1;
    	num_verb = num_verb > max_num ? max_num : num_verb;
    	num_conj = num_conj > max_num ? max_num : num_conj;
    	num_punc = num_punc > max_num ? max_num : num_punc;
    	
//    	for(int i = small+1; i < large; i++) {    		
//    		code = createArcCodePPP(HP_BP_MP, pHead, pos[i], pMod) | tid;
//    		addLabeledArcFeature(code, fv);
//    		addLabeledArcFeature(code | attDist, fv);
//    		
//    		code = createArcCodePPP(HP_BP_MP, pHeadA, posA[i], pModA) | tid;
//    		addLabeledArcFeature(code, fv);
//    		addLabeledArcFeature(code | attDist, fv);
//    	}
//    	
//		code = createArcCodePPP(HP_BCC_MP, pHeadA, num_conj, pModA) | tid;
//		addLabeledArcFeature(code, fv);
//		addLabeledArcFeature(code | attDist, fv);
//		
//		code = createArcCodePPP(HP_BVB_MP, pHeadA, num_verb, pModA) | tid;
//		addLabeledArcFeature(code, fv);
//		addLabeledArcFeature(code | attDist, fv);
//		
//		code = createArcCodePPP(HP_BPN_MP, pHeadA, num_punc, pModA) | tid;
//		addLabeledArcFeature(code, fv);
//		addLabeledArcFeature(code | attDist, fv);
//		
//		code = createArcCodePPP(HP_BCC_MP, 0, num_conj, 0) | tid;
//		addLabeledArcFeature(code, fv);
//		addLabeledArcFeature(code | attDist, fv);
//		
//		code = createArcCodePPP(HP_BVB_MP, 0, num_verb, 0) | tid;
//		addLabeledArcFeature(code, fv);
//		addLabeledArcFeature(code | attDist, fv);
//		
//		code = createArcCodePPP(HP_BPN_MP, 0, num_punc, 0) | tid;
//		addLabeledArcFeature(code, fv);
//		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(HPp_HP, pHeadLeft, pHead) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(HP_HPn, pHead, pHeadRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HPp_HP_HPn, pHeadLeft, pHead, pHeadRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(MPp_MP, pModLeft, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(MP_MPn, pMod, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(MPp_MP_MPn, pModLeft, pMod, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(HPp_HP, pHeadLeftA, pHeadA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(HP_HPn, pHeadA, pHeadRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HPp_HP_HPn, pHeadLeftA, pHeadA, pHeadRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(MPp_MP, pModLeftA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePP(MP_MPn, pModA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(MPp_MP_MPn, pModLeftA, pModA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	
    	// feature posL-1 posL posR posR+1
    	code = createArcCodePPPP(HPp_HP_MP_MPn, pHeadLeft, pHead, pMod, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HP_MP_MPn, pHead, pMod, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HPp_HP_MP, pHeadLeft, pHead, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HPp_MP_MPn, pHeadLeft, pMod, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HPp_HP_MPn, pHeadLeft, pHead, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);

    	code = createArcCodePPPP(HPp_HP_MP_MPn, pHeadLeftA, pHeadA, pModA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HP_MP_MPn, pHeadA, pModA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HPp_HP_MP, pHeadLeftA, pHeadA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HPp_MP_MPn, pHeadLeftA, pModA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	code = createArcCodePPP(HPp_HP_MPn, pHeadLeftA, pHeadA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    	
    	// feature posL posL+1 posR-1 posR
		code = createArcCodePPPP(HP_HPn_MPp_MP, pHead, pHeadRight, pModLeft, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HP_MPp_MP, pHead, pModLeft, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HP_HPn_MP, pHead, pHeadRight, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HPn_MPp_MP, pHeadRight, pModLeft, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HP_HPn_MPp, pHead, pHeadRight, pModLeft) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPPP(HP_HPn_MPp_MP, pHeadA, pHeadRightA, pModLeftA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HP_MPp_MP, pHeadA, pModLeftA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HP_HPn_MP, pHeadA, pHeadRightA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HPn_MPp_MP, pHeadRightA, pModLeftA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPP(HP_HPn_MPp, pHeadA, pHeadRightA, pModLeftA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
	
    	
		// feature posL-1 posL posR-1 posR
		// feature posL posL+1 posR posR+1
		code = createArcCodePPPP(HPp_HP_MPp_MP, pHeadLeft, pHead, pModLeft, pMod) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPPP(HP_HPn_MP_MPn, pHead, pHeadRight, pMod, pModRight) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPPP(HPp_HP_MPp_MP, pHeadLeftA, pHeadA, pModLeftA, pModA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
		code = createArcCodePPPP(HP_HPn_MP_MPn, pHeadA, pHeadRightA, pModA, pModRightA) | tid;
		addLabeledArcFeature(code, fv);
		addLabeledArcFeature(code | attDist, fv);
		
    }

    public void addCore1OBigramFeatures(Collector fv, int head, int headP, 
    		int mod, int modP, int attDist, int type) 
    {
    	
    	long code = 0;
    	
    	int tid = type << 4;
    	
    	code = createArcCodeWWPP(HW_MW_HP_MP, head, mod, headP, modP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeWPP(MW_HP_MP, mod, headP, modP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeWPP(HW_HP_MP, head, headP, modP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeWP(MW_HP, mod, headP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeWP(HW_MP, head, modP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	    	
    	code = createArcCodeWP(HW_HP, head, headP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
    	
    	code = createArcCodeWP(MW_MP, mod, modP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | attDist, fv);
      
    }
    
    public void createLabeledGPCFeatureVector(Collector fv, DependencyInstance inst, 
    		int gp, int par, int c, int type, int ptype) 
    {
    	
    	int[] pos = inst.postagids;
    	int[] posA = inst.cpostagids;
    	//int[] lemma = inst.lemmaids;
    	int[] lemma = inst.lemmaids != null ? inst.lemmaids : inst.formids;
    	
    	int flag = (((((gp > par ? 0 : 1) << 1) | (par > c ? 0 : 1)) << 1) | 1);
    	int tid = ((ptype << depNumBits) | type) << 4;
    	
    	int GP = pos[gp];
    	int HP = pos[par];
    	int MP = pos[c];
    	int GC = posA[gp];
    	int HC = posA[par];
    	int MC = posA[c];
    	long code = 0;

    	code = createArcCodePPP(GP_HP_MP, GP, HP, MP) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | flag, fv);

    	code = createArcCodePPP(GC_HC_MC, GC, HC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | flag, fv);
    
        int GL = lemma[gp];
        int HL = lemma[par];
        int ML = lemma[c];

        code = createArcCodeWPP(GL_HC_MC, GL, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWPP(GC_HL_MC, HL, GC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWPP(GC_HC_ML, ML, GC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);
        
        code = createArcCodePP(GC_HC, GC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

    	code = createArcCodePP(GC_MC, GC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | flag, fv);

    	code = createArcCodePP(HC_MC, HC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | flag, fv);
    	
    	code = createArcCodeWWP(GL_HL_MC, GL, HL, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWP(GL_HC_ML, GL, ML, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWP(GC_HL_ML, HL, ML, GC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWWW(GL_HL_ML, GL, HL, ML) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GL_HC, GL, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GC_HL, HL, GC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWW(GL_HL, GL, HL) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GL_MC, GL, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(GC_ML, ML, GC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWW(GL_ML, GL, ML) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(HL_MC, HL, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWP(HC_ML, ML, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);

        code = createArcCodeWW(HL_ML, HL, ML) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | flag, fv);
        
        addLabeledTurboGPC(inst, gp, par, c, flag, tid, fv);
    }
    
    void addLabeledTurboGPC(DependencyInstance inst, int gp, int par, int c, 
    		int dirFlag, int tid, Collector fv) {
    	int[] posA = inst.cpostagids;
    	//int[] lemma = inst.lemmaids;
    	int[] lemma = inst.lemmaids != null ? inst.lemmaids : inst.formids;
    	int len = posA.length;

    	int GC = posA[gp];
    	int HC = posA[par];
    	int MC = posA[c];

    	int pGC = gp > 0 ? posA[gp - 1] : TOKEN_START;
    	int nGC = gp < len - 1 ? posA[gp + 1] : TOKEN_END;
    	int pHC = par > 0 ? posA[par - 1] : TOKEN_START;
    	int nHC = par < len - 1 ? posA[par + 1] : TOKEN_END;
    	int pMC = c > 0 ? posA[c - 1] : TOKEN_START;
    	int nMC = c < len - 1 ? posA[c + 1] : TOKEN_END;

    	long code = 0;

    	// CCC
    	code = createArcCodePPPP(pGC_GC_HC_MC, pGC, GC, HC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPP(GC_nGC_HC_MC, GC, nGC, HC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPP(GC_pHC_HC_MC, GC, pHC, HC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPP(GC_HC_nHC_MC, GC, HC, nHC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPP(GC_HC_pMC_MC, GC, HC, pMC, MC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPP(GC_HC_MC_nMC, GC, HC, MC, nMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_pGC_pHC, GC, HC, MC, pGC, pHC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_pGC_pMC, GC, HC, MC , pGC, pMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_pHC_pMC, GC, HC, MC, pHC, pMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_nGC_nHC, GC, HC, MC, nGC, nHC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_nGC_nMC, GC, HC, MC, nGC, nMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_nHC_nMC, GC, HC, MC, nHC, nMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_pGC_nHC, GC, HC, MC, pGC, nHC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_pGC_nMC, GC, HC, MC, pGC, nMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_pHC_nMC, GC, HC, MC, pHC, nMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_nGC_pHC, GC, HC, MC, nGC, pHC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_nGC_pMC, GC, HC, MC, nGC, pMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);

    	code = createArcCodePPPPP(GC_HC_MC_nHC_pMC, GC, HC, MC, nHC, pMC) | tid;
    	addLabeledArcFeature(code, fv);
    	addLabeledArcFeature(code | dirFlag, fv);
        
        int GL = lemma[gp];
        int HL = lemma[par];
        int ML = lemma[c];

        // LCC
        code = createArcCodeWPPP(pGC_GL_HC_MC, GL, pGC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_nGC_HC_MC, GL, nGC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_pHC_HC_MC, GL, pHC, HC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_HC_nHC_MC, GL, HC, nHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_HC_pMC_MC, GL, HC, pMC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GL_HC_MC_nMC, GL, HC, MC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        // CLC
        code = createArcCodeWPPP(pGC_GC_HL_MC, HL, pGC, GC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_nGC_HL_MC, HL, GC, nGC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_pHC_HL_MC, HL, GC, pHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HL_nHC_MC, HL, GC, nHC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HL_pMC_MC, HL, GC, pMC, MC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HL_MC_nMC, HL, GC, MC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        // CCL
        code = createArcCodeWPPP(pGC_GC_HC_ML, ML, pGC, GC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_nGC_HC_ML, ML, GC, nGC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_pHC_HC_ML, ML, GC, pHC, HC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HC_nHC_ML, ML, GC, HC, nHC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HC_pMC_ML, ML, GC, HC, pMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);

        code = createArcCodeWPPP(GC_HC_ML_nMC, ML, GC, HC, nMC) | tid;
        addLabeledArcFeature(code, fv);
        addLabeledArcFeature(code | dirFlag, fv);
    }

    /************************************************************************
     *  Region end #
     ************************************************************************/

   
    /************************************************************************
     * Region start #
     * 
     *  Functions that add feature codes into feature vectors and alphabets
     *  
     ************************************************************************/
    
    static final int C1 = 0xcc9e2d51;
    static final int C2 = 0x1b873593;
    private final int hashcode2int(long code)
    {
    	int k1 = (int) (code & 0xffffffff);
    	int k2 = (int) (code >>> 32);
    	int h = 0;
    	
    	k1 *= C1;
    	k1 = (k1 << 15) | (k1 >>> 17); // ROTL32(k1,15);
    	k1 *= C2;
    	h ^= k1;
    	h = (h << 13) | (h >>> 19); // ROTL32(h1,13);
    	h = h * 5 + 0xe6546b64;
    	
    	k2 *= C1;
    	k2 = (k2 << 15) | (k2 >>> 17); // ROTL32(k1,15);
    	k2 *= C2;
    	h ^= k2;
    	h = (h << 13) | (h >>> 19); // ROTL32(h1,13);
    	h = h * 5 + 0xe6546b64;
    	
    	// finalizer
    	h ^= h >> 16;
    	h *= 0x85ebca6b;
    	h ^= h >> 13;
    	h *= 0xc2b2ae35;
    	h ^= h >> 16;
        		
        //return (int) (0xFFFFFFFFL & h) % 115911564;
        //return (int) ((numArcFeats-1) & h);
        return h;
    }
    
//    private final int hashcode2int(long code)
//    {
//    	long hash = (code ^ (code&0xffffffff00000000L) >>> 32)*31;
//    	int id = (int)((hash < 0 ? -hash : hash) % numArcFeats);
//    	return id;
//    }
    
//    private final int hashcode2int(long code)
//    {
//    	int id = (int)((code < 0 ? -code : code) % 536870909);
//    	return id;
//    }
    
    
//    private final int hashcode2int(long l) {
//        long r= l;// 27
//        l = (l>>13)&0xffffffffffffe000L;
//        r ^= l;   // 40
//        l = (l>>11)&0xffffffffffff0000L;
//        r ^= l;   // 51
//        l = (l>>9)& 0xfffffffffffc0000L; //53
//        r ^= l;  // 60
//        l = (l>>7)& 0xfffffffffff00000L; //62
//        r ^=l;    //67
//        //int x = ((int)r) % 115911563;
//        int x = (int) (r % 115911563);
//    
//        return x >= 0 ? x : -x ; 
//    }
    
    public final void addLabeledArcFeature(long code, Collector mat) {
    	int id = hashcode2int(code) & numLabeledArcFeats;	
    	mat.addEntry(id);
    	if (!stoppedGrowth)
    		featureHashSet.add(code);
    }
    
    public final void addLabeledArcFeature(long code, float value, Collector mat) {
    	int id = hashcode2int(code) & numLabeledArcFeats; 	
    	mat.addEntry(id, value);
    	if (!stoppedGrowth)
    		featureHashSet.add(code);
    }
    
    public final void addWordFeature(long code, FeatureVector mat) {
    	int id = wordAlphabet.lookupIndex(code, numWordFeats);
    	if (id >= 0) {
    		mat.addEntry(id);
    		if (id == numWordFeats) ++numWordFeats;
    	}
    }
    
    public final void addWordFeature(long code, float value, FeatureVector mat) {
    	int id = wordAlphabet.lookupIndex(code, numWordFeats);
    	if (id >= 0) {
    		mat.addEntry(id, value);
    		if (id == numWordFeats) ++numWordFeats;
    	}
    }
    
    /************************************************************************
     *  Region end #
     ************************************************************************/
    
    
    
    /************************************************************************
     * Region start #
     * 
     *  Functions to create or parse 64-bit feature code
     *  
     *  A feature code is like:
     *  
     *    X1 X2 .. Xk TEMP DIST
     *  
     *  where Xi   is the integer id of a word, pos tag, etc.
     *        TEMP is the integer id of the feature template
     *        DIST is the integer binned length  (4 bits)
     ************************************************************************/
    
    private final long extractArcTemplateCode(long code) {
    	return (code >> flagBits) & ((1 << numArcFeatBits)-1);
    }
    
    private final long extractDistanceCode(long code) {
    	return code & 15;
    }
    
    private final long extractLabelCode(long code) {
    	return (code >> 4) & ((1 << depNumBits)-1);
    }
    
    private final long extractPLabelCode(long code) {
    	return (code >> (depNumBits+4)) & ((1 << depNumBits)-1);
    }
    
    private final void extractArcCodeP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[0] = (int) (code & ((1 << tagNumBits)-1));
    }
    
    private final void extractArcCodePP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << tagNumBits)-1));
    }
    
    private final void extractArcCodePPP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << tagNumBits)-1));
    }
    
    private final void extractArcCodePPPP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[3] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << tagNumBits)-1));
    }
    
    private final void extractArcCodePPPPP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
    	x[4] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
    	x[3] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << tagNumBits)-1));
    }
    
    private final void extractArcCodeWPPP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[3] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeW(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeWW(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[1] = (int) (code & ((1 << wordNumBits)-1));
	    code = code >> wordNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeWP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeWPP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeWWP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << wordNumBits)-1));
	    code = code >> wordNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeWWW(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[2] = (int) (code & ((1 << wordNumBits)-1));
	    code = code >> wordNumBits;
	    x[1] = (int) (code & ((1 << wordNumBits)-1));
	    code = code >> wordNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    private final void extractArcCodeWWPP(long code, int[] x) {
    	code = (code >> flagBits) >> numArcFeatBits;
	    x[3] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[2] = (int) (code & ((1 << tagNumBits)-1));
	    code = code >> tagNumBits;
	    x[1] = (int) (code & ((1 << wordNumBits)-1));
	    code = code >> wordNumBits;
	    x[0] = (int) (code & ((1 << wordNumBits)-1));
    }
    
    public final long createArcCodeP(FeatureTemplate.Arc temp, long x) {
    	return ((x << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodePP(FeatureTemplate.Arc temp, long x, long y) {
    	return ((((x << tagNumBits) | y) << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodePPP(FeatureTemplate.Arc temp, long x, long y, long z) {
    	return ((((((x << tagNumBits) | y) << tagNumBits) | z) << numArcFeatBits)
    			| temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodePPPP(FeatureTemplate.Arc temp, long x, long y, long u, long v) {
    	return ((((((((x << tagNumBits) | y) << tagNumBits) | u) << tagNumBits) | v)
    			<< numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodePPPPP(FeatureTemplate.Arc temp, long x, long y, long u, long v, long w) {
    	return ((((((((((x << tagNumBits) | y) << tagNumBits) | u) << tagNumBits) | v) << tagNumBits) | w)
    			<< numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeW(FeatureTemplate.Arc temp, long x) {
    	return ((x << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWW(FeatureTemplate.Arc temp, long x, long y) {
    	return ((((x << wordNumBits) | y) << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWWW(FeatureTemplate.Arc temp, long x, long y, long z) {
    	return ((((((x << wordNumBits) | y) << wordNumBits) | z) << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWP(FeatureTemplate.Arc temp, long x, long y) {
    	return ((((x << tagNumBits) | y) << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWPP(FeatureTemplate.Arc temp, long x, long y, long z) {
    	return ((((((x << tagNumBits) | y) << tagNumBits) | z) << numArcFeatBits)
    			| temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWPPP(FeatureTemplate.Arc temp, long x, long y, long u, long v) {
    	return ((((((((x << tagNumBits) | y) << tagNumBits) | u) << tagNumBits) | v) << numArcFeatBits)
    			| temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWWP(FeatureTemplate.Arc temp, long x, long y, long z) {
    	return ((((((x << wordNumBits) | y) << tagNumBits) | z) << numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createArcCodeWWPP(FeatureTemplate.Arc temp, long x, long y, long u, long v) {
    	return ((((((((x << wordNumBits) | y) << tagNumBits) | u) << tagNumBits) | v)
    			<< numArcFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createWordCodeW(FeatureTemplate.Word temp, long x) {
    	return ((x << numWordFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createWordCodeP(FeatureTemplate.Word temp, long x) {
    	return ((x << numWordFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createWordCodePP(FeatureTemplate.Word temp, long x, long y) {
    	return ((((x << tagNumBits) | y) << numWordFeatBits) | temp.ordinal()) << flagBits;
    }
    
    public final long createWordCodePPP(FeatureTemplate.Word temp, long x, long y, long z) {
    	return ((((((x << tagNumBits) | y) << tagNumBits) | z) << numWordFeatBits)
    			| temp.ordinal()) << flagBits;
    }
    
    public final long createWordCodeWP(FeatureTemplate.Word temp, long x, long y) {
    	return ((((x << tagNumBits) | y) << numWordFeatBits) | temp.ordinal()) << flagBits;
    }
    
    /************************************************************************
     *  Region end #
     ************************************************************************/
    
    public void clearFeatureHashSet()
    {
    	featureHashSet = null;
    }
    
    public void fillParameters(LowRankTensor tensor, LowRankTensor tensor2, Parameters params) {
        //System.out.println(arcAlphabet.size());	
    	long[] codes = //arcAlphabet.toArray();
    					featureHashSet.toArray();
    	clearFeatureHashSet();
    	int[] x = new int[5];
    	
    	for (long code : codes) {
    		
    		int dist = (int) extractDistanceCode(code);
    		int temp = (int) extractArcTemplateCode(code);
    		
    		int label = (int) extractLabelCode(code);
    		int plabel = (int) extractPLabelCode(code);
    		
    		long head = -1, mod = -1, gp = -1;
    		
    		//code = createArcCodePP(HPp_HP, pHeadLeft, pHead) | tid;
    		if (temp == HPp_HP.ordinal()) {
    			extractArcCodePP(code, x);
    			head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		//code = createArcCodePP(HP_HPn, pHead, pHeadRight) | tid;
    		if (temp == HP_HPn.ordinal()) {
    			extractArcCodePP(code, x);
    			head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		//code = createArcCodePPP(HPp_HP_HPn, pHeadLeft, pHead, pHeadRight) | tid;
    		if (temp == HPp_HP_HPn.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodePPP(WORDFV_PpP0Pn, x[0], x[1], x[2]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		//code = createArcCodePP(MPp_MP, pModLeft, pMod) | tid;
    		if (temp == MPp_MP.ordinal()) {
    			extractArcCodePP(code, x);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    		}
    		
    		//code = createArcCodePP(MP_MPn, pMod, pModRight) | tid;
    		if (temp == MP_MPn.ordinal()) {
    			extractArcCodePP(code, x);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    		}
    		
    		//code = createArcCodePPP(MPp_MP_MPn, pModLeft, pMod, pModRight) | tid;
    		if (temp == MPp_MP_MPn.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodePPP(WORDFV_PpP0Pn, x[0], x[1], x[2]);
    		}
    		
        	//code = createArcCodePPPP(CORE_POS_PT0, pHeadLeft, pHead, pMod, pModRight);
    		if (temp == HPp_HP_MP_MPn.ordinal()) {
    			extractArcCodePPPP(code, x);
    			head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
    		}
    		
        	//code = createArcCodePPP(CORE_POS_PT1, pHead, pMod, pModRight);
    		else if (temp == HP_MP_MPn.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodeP(WORDFV_P0, x[0]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
    		}

        	//code = createArcCodePPP(CORE_POS_PT2, pHeadLeft, pHead, pMod);
    		else if (temp == HPp_HP_MP.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
        	//code = createArcCodePPP(CORE_POS_PT3, pHeadLeft, pMod, pModRight);
    		else if (temp == HPp_MP_MPn.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodeP(WORDFV_Pp, x[0]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
    		}
    		
        	//code = createArcCodePPP(CORE_POS_PT4, pHeadLeft, pHead, pModRight);
    		else if (temp == HPp_HP_MPn.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_Pn, x[2]);
    		}
        	        	
    		//code = createArcCodePPPP(CORE_POS_APT0, pHead, pHeadRight, pModLeft, pMod);
    		else if (temp == HP_HPn_MPp_MP.ordinal()) {
    			extractArcCodePPPP(code, x);
    			head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
    		}
    		
    		//code = createArcCodePPP(CORE_POS_APT1, pHead, pModLeft, pMod);
    		else if (temp == HP_MPp_MP.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodeP(WORDFV_P0, x[0]);
    			mod = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
    		}
    		
    		//code = createArcCodePPP(CORE_POS_APT2, pHead, pHeadRight, pMod);
    		else if (temp == HP_HPn_MP.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		//code = createArcCodePPP(CORE_POS_APT3, pHeadRight, pModLeft, pMod);
    		else if (temp == HPn_MPp_MP.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodeP(WORDFV_Pn, x[0]);
    			mod = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
    		}
    		
    		//code = createArcCodePPP(CORE_POS_APT4, pHead, pHeadRight, pModLeft);
    		else if (temp == HP_HPn_MPp.ordinal()) {
    			extractArcCodePPP(code, x);
    			head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_Pp, x[2]);
    		}

    		//code = createArcCodePPPP(CORE_POS_BPT, pHeadLeft, pHead, pModLeft, pMod);
    		else if (temp == HPp_HP_MPp_MP.ordinal()) {
    			extractArcCodePPPP(code, x);
    			head = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
    		}
    		
    		//code = createArcCodePPPP(CORE_POS_CPT, pHead, pHeadRight, pMod, pModRight);
    		else if (temp == HP_HPn_MP_MPn.ordinal()) {
    			extractArcCodePPPP(code, x);
    			head = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
    		}
    		
        	//code = createArcCodeWWPP(CORE_BIGRAM_A, head, mod, headP, modP);
    		else if (temp == HW_MW_HP_MP.ordinal()) {
    			extractArcCodeWWPP(code, x);
    			head = createWordCodeWP(WORDFV_W0P0, x[0], x[2]);
    			mod = createWordCodeWP(WORDFV_W0P0, x[1], x[3]);
    		}
        	
        	//code = createArcCodeWPP(CORE_BIGRAM_B, mod, headP, modP);
    		else if (temp == MW_HP_MP.ordinal()) {
    			extractArcCodeWPP(code, x);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeWP(WORDFV_W0P0, x[0], x[2]);
    		}
        	
        	//code = createArcCodeWPP(CORE_BIGRAM_C, head, headP, modP);
    		else if (temp == HW_HP_MP.ordinal()) {
    			extractArcCodeWPP(code, x);
    			head = createWordCodeWP(WORDFV_W0P0, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
        	
        	//code = createArcCodeWP(CORE_BIGRAM_D, mod, headP);
    		else if (temp == MW_HP.ordinal()) {
    			extractArcCodeWP(code, x);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
        	
        	//code = createArcCodeWP(CORE_BIGRAM_E, head, modP);
    		else if (temp == HW_MP.ordinal()) {
    			extractArcCodeWP(code, x);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_P0, x[1]);
    		}
        	
            //code = createArcCodeWW(CORE_BIGRAM_F, head, mod);
    		else if (temp == HW_MW.ordinal()) {
    			extractArcCodeWW(code, x);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeW(WORDFV_W0, x[1]);
    		}
        	
            //code = createArcCodePP(CORE_BIGRAM_G, headP, modP);
    		else if (temp == HP_MP.ordinal()) {
    			extractArcCodePP(code, x);
    			head = createWordCodeW(WORDFV_P0, x[0]);
    			mod = createWordCodeW(WORDFV_P0, x[1]);
    		}
    		
        	//code = createArcCodeWP(CORE_BIGRAM_H, head, headP);
    		else if (temp == HW_HP.ordinal()) {
    			extractArcCodeWP(code, x);
    			head = createWordCodeWP(WORDFV_W0P0, x[0], x[1]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
        	//code = createArcCodeWP(CORE_BIGRAM_K, mod, modP);
    		else if (temp == MW_MP.ordinal()) {
    			extractArcCodeWP(code, x);    			
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodeWP(WORDFV_W0P0, x[0], x[1]);
    		}
    		
    		else if (temp == CORE_HEAD_WORD.ordinal()) {
    			extractArcCodeW(code, x);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == CORE_HEAD_POS.ordinal()) {
    			extractArcCodeP(code, x);
    			head = createWordCodeP(WORDFV_P0, x[0]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);	
    		}
    		
    		else if (temp == CORE_MOD_WORD.ordinal()) {
    			extractArcCodeW(code, x);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodeW(WORDFV_W0, x[0]);    			
    		}
    		
    		else if (temp == CORE_MOD_POS.ordinal()) {
    			extractArcCodeP(code, x);
    			head = createWordCodeP(WORDFV_BIAS, 0);	
    			mod = createWordCodeP(WORDFV_P0, x[0]);
    		}
    		
    		else if (temp == CORE_HEAD_pWORD.ordinal()) {
    			extractArcCodeW(code, x);
    			head = createWordCodeW(WORDFV_Wp, x[0]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == CORE_HEAD_nWORD.ordinal()) {
    			extractArcCodeW(code, x);
    			head = createWordCodeW(WORDFV_Wn, x[0]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == CORE_MOD_pWORD.ordinal()) {
    			extractArcCodeW(code, x);
    			mod = createWordCodeW(WORDFV_Wp, x[0]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == CORE_MOD_nWORD.ordinal()) {
    			extractArcCodeW(code, x);
    			mod = createWordCodeW(WORDFV_Wn, x[0]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == HEAD_EMB.ordinal()) {
    			extractArcCodeW(code, x);
    			head = createWordCodeW(WORDFV_EMB, x[0]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == MOD_EMB.ordinal()) {
    			extractArcCodeW(code, x);
    			mod = createWordCodeW(WORDFV_EMB, x[0]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		// second order
    		else if (temp == GP_HP_MP.ordinal()) {
    			extractArcCodePPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HC_MC.ordinal()) {
    			extractArcCodePPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GL_HC_MC.ordinal()) {
    			extractArcCodeWPP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HL_MC.ordinal()) {
    			extractArcCodeWPP(code, x);		// HL, GC, MC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HC_ML.ordinal()) {
    			extractArcCodeWPP(code, x);		// ML, GC, HC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}

    		else if (temp == GL_HL_MC.ordinal()) {
    			extractArcCodeWWP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeW(WORDFV_W0, x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GL_HC_ML.ordinal()) {
    			extractArcCodeWWP(code, x);		// GL, ML, HC
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeW(WORDFV_W0, x[1]);
    		}
    		
    		else if (temp == GC_HL_ML.ordinal()) {
    			extractArcCodeWWP(code, x);		// HL, ML, GC
    			gp = createWordCodeP(WORDFV_P0, x[2]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeW(WORDFV_W0, x[1]);
    		}
    		
    		else if (temp == GL_HL_ML.ordinal()) {
    			extractArcCodeWWW(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeW(WORDFV_W0, x[1]);
    			mod = createWordCodeW(WORDFV_W0, x[2]);
    		}

    		else if (temp == GC_HC.ordinal()) {
    			extractArcCodePP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == GL_HC.ordinal()) {
    			extractArcCodeWP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == GC_HL.ordinal()) {
    			extractArcCodeWP(code, x);		// HL, GC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}
    		
    		else if (temp == GL_HL.ordinal()) {
    			extractArcCodeWW(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeW(WORDFV_W0, x[1]);
    			mod = createWordCodeP(WORDFV_BIAS, 0);
    		}

    		else if (temp == GC_MC.ordinal()) {
    			extractArcCodePP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodeP(WORDFV_P0, x[1]);
    		}
    		
    		else if (temp == GL_MC.ordinal()) {
    			extractArcCodeWP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodeP(WORDFV_P0, x[1]);
    		}
    		
    		else if (temp == GC_ML.ordinal()) {
    			extractArcCodeWP(code, x);		// ML, GC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
    		
    		else if (temp == GL_ML.ordinal()) {
    			extractArcCodeWW(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_BIAS, 0);
    			mod = createWordCodeW(WORDFV_W0, x[1]);
    		}
    		
    		else if (temp == HC_MC.ordinal()) {
    			extractArcCodePP(code, x);
    			gp = createWordCodeP(WORDFV_BIAS, 0);
    			head = createWordCodeP(WORDFV_P0, x[0]);
    			mod = createWordCodeP(WORDFV_P0, x[1]);
    		}
    		
    		else if (temp == HL_MC.ordinal()) {
    			extractArcCodeWP(code, x);
    			gp = createWordCodeP(WORDFV_BIAS, 0);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_P0, x[1]);
    		}
    		
    		else if (temp == HC_ML.ordinal()) {
    			extractArcCodeWP(code, x);		// ML, HC
    			gp = createWordCodeP(WORDFV_BIAS, 0);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
    		
    		else if (temp == HL_ML.ordinal()) {
    			extractArcCodeWW(code, x);
    			gp = createWordCodeP(WORDFV_BIAS, 0);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeW(WORDFV_W0, x[1]);
    		}

    		else if (temp == pGC_GC_HC_MC.ordinal()) {
    			extractArcCodePPPP(code, x);
    			gp = createWordCodePP(WORDFV_PpP0, x[0], x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_nGC_HC_MC.ordinal()) {
    			extractArcCodePPPP(code, x);
    			gp = createWordCodePP(WORDFV_P0Pn, x[0], x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_pHC_HC_MC.ordinal()) {
    			extractArcCodePPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_HC_nHC_MC.ordinal()) {
    			extractArcCodePPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_HC_pMC_MC.ordinal()) {
    			extractArcCodePPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
    		}
    		
    		else if (temp == GC_HC_MC_nMC.ordinal()) {
    			extractArcCodePPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
    		}

    		else if (temp == pGC_GL_HC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// GL, pGC, HC, MC
    			gp = createWordCodeWP(WORDFV_W0Pp, x[0], x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GL_nGC_HC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);
    			gp = createWordCodeWP(WORDFV_W0Pn, x[0], x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GL_pHC_HC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GL_HC_nHC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GL_HC_pMC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
    		}
    		
    		else if (temp == GL_HC_MC_nMC.ordinal()) {
    			extractArcCodeWPPP(code, x);
    			gp = createWordCodeW(WORDFV_W0, x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
    		}

    		else if (temp == pGC_GC_HL_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// HL, pGC, GC, MC
    			gp = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_nGC_HL_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// HL, GC, nGC, MC
    			gp = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_pHC_HL_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// HL, GC, pHC, MC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeWP(WORDFV_W0Pp, x[0], x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_HL_nHC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// HL, GC, nHC, MC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeWP(WORDFV_W0Pn, x[0], x[2]);
    			mod = createWordCodeP(WORDFV_P0, x[3]);
    		}
    		
    		else if (temp == GC_HL_pMC_MC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// HL, GC, pMC, MC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
    		}
    		
    		else if (temp == GC_HL_MC_nMC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// HL, GC, MC, nMC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeW(WORDFV_W0, x[0]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
    		}

    		else if (temp == pGC_GC_HC_ML.ordinal()) {
    			extractArcCodeWPPP(code, x);		// ML, pGC, GC, HC
    			gp = createWordCodePP(WORDFV_PpP0, x[1], x[2]);
    			head = createWordCodeP(WORDFV_P0, x[3]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
    		
    		else if (temp == GC_nGC_HC_ML.ordinal()) {
    			extractArcCodeWPPP(code, x);		// ML, GC, nGC, HC
    			gp = createWordCodePP(WORDFV_P0Pn, x[1], x[2]);
    			head = createWordCodeP(WORDFV_P0, x[3]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
    		
    		else if (temp == GC_pHC_HC_ML.ordinal()) {
    			extractArcCodeWPPP(code, x);		// ML, GC, pHC, HC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodePP(WORDFV_PpP0, x[2], x[3]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
    		
    		else if (temp == GC_HC_nHC_ML.ordinal()) {
    			extractArcCodeWPPP(code, x);		// ML, GC, HC, nHC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodePP(WORDFV_P0Pn, x[2], x[3]);
    			mod = createWordCodeW(WORDFV_W0, x[0]);
    		}
    		
    		else if (temp == GC_HC_pMC_ML.ordinal()) {
    			extractArcCodeWPPP(code, x);		// ML, GC, HC, pMC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeWP(WORDFV_W0Pp, x[0], x[3]);
    		}
    		
    		else if (temp == GC_HC_ML_nMC.ordinal()) {
    			extractArcCodeWPPP(code, x);		// ML, GC, HC, nMC
    			gp = createWordCodeP(WORDFV_P0, x[1]);
    			head = createWordCodeP(WORDFV_P0, x[2]);
    			mod = createWordCodeWP(WORDFV_W0Pn, x[0], x[3]);
    		}

    		else if (temp == GC_HC_MC_pGC_pHC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
    			head = createWordCodePP(WORDFV_PpP0, x[4], x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HC_MC_pGC_pMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[4], x[2]);
    		}
    		
    		else if (temp == GC_HC_MC_pHC_pMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodePP(WORDFV_PpP0, x[3], x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[4], x[2]);
    		}
    		
    		else if (temp == GC_HC_MC_nGC_nHC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
    			head = createWordCodePP(WORDFV_P0Pn, x[1], x[4]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HC_MC_nGC_nMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
    		}
    		
    		else if (temp == GC_HC_MC_nHC_nMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodePP(WORDFV_P0Pn, x[1], x[3]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
    		}
    		
    		else if (temp == GC_HC_MC_pGC_nHC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
    			head = createWordCodePP(WORDFV_P0Pn, x[1], x[4]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HC_MC_pGC_nMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_PpP0, x[3], x[0]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
    		}
    		
    		else if (temp == GC_HC_MC_pHC_nMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodeP(WORDFV_P0, x[0]);
    			head = createWordCodePP(WORDFV_PpP0, x[3], x[1]);
    			mod = createWordCodePP(WORDFV_P0Pn, x[2], x[4]);
    		}
    		
    		else if (temp == GC_HC_MC_nGC_pHC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
    			head = createWordCodePP(WORDFV_PpP0, x[4], x[1]);
    			mod = createWordCodeP(WORDFV_P0, x[2]);
    		}
    		
    		else if (temp == GC_HC_MC_nGC_pMC.ordinal()) {
    			extractArcCodePPPPP(code, x);
    			gp = createWordCodePP(WORDFV_P0Pn, x[0], x[3]);
    			head = createWordCodeP(WORDFV_P0, x[1]);
    			mod = createWordCodePP(WORDFV_PpP0, x[4], x[2]);
    		}
    		
    		else {
    			//System.out.println(temp);
    			continue;
    		}
    		
    		int headId, modId, gpId = 0;
    		int dir = 0, pdir = 0;
    		headId = wordAlphabet.lookupIndex(head);
    		modId = wordAlphabet.lookupIndex(mod);
    		if (gp != -1) {
    			gpId = wordAlphabet.lookupIndex(gp);
    			if (dist != 0) {
					dir = ((dist>>1) & 1) + 1;
					pdir = ((dist>>2) & 1) + 1;
				}
    		}
    		
    		if (headId >= 0 && modId >= 0 && gpId >= 0) {
				int id = hashcode2int(code) & numLabeledArcFeats;
				if (id < 0) continue;
				float value = params.paramsL[id];
				if (gp == -1) {
					int[] y = {headId, modId, dist*params.T+label};
					tensor.add(y, value);
				}
				else {
					int[] y = {gpId, headId, modId, pdir*params.T+plabel, dir*params.T+label};
					tensor2.add(y, value);
				}
            }
    	}
    	
    }
}

// do nothing
class LazyCollector implements Collector
{

	@Override
	public void addEntry(int x) {

	}

	@Override
	public void addEntry(int x, float va) {

	}
	
}
