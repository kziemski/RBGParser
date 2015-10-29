package parser;

import java.io.Serializable;
import java.util.Arrays;

import utils.FeatureVector;
import utils.Utils;

public class Parameters implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public static final int d = 7;
	
	public transient Options options;
	public final int labelLossType;
	public float C, gamma, gammaLabel;
	public int size, sizeL;
	public int rank;
	public int N, M, T, D;
	
	public float[] params, paramsL;
	public float[][] U, V, W;
	public transient float[] total, totalL;
	public transient float[][] totalU, totalV, totalW;
	
	public transient FeatureVector[] dU, dV, dW;
	
	public Parameters(DependencyPipe pipe, Options options) 
	{
		 //T = pipe.types.length;
        D = d * 2 + 1;
		size = pipe.synFactory.numArcFeats+1;		
		params = new float[size];
		total = new float[size];
		
		if (options.learnLabel) {
			sizeL = pipe.synFactory.numLabeledArcFeats+1;
			paramsL = new float[sizeL];
			totalL = new float[sizeL];
		}
		
		this.options = options;
		this.labelLossType = options.labelLossType;
		C = options.C;
		gamma = options.gamma;
		gammaLabel = options.gammaLabel;
		rank = options.R;
		
		N = pipe.synFactory.numWordFeats;
		M = N;
		U = new float[rank][N];		
		V = new float[rank][M];
		W = new float[rank][D];
		totalU = new float[rank][N];
		totalV = new float[rank][M];
		totalW = new float[rank][D];
		dU = new FeatureVector[rank];
		dV = new FeatureVector[rank];
		dW = new FeatureVector[rank];

	}
	
	public void randomlyInitUVW() 
	{
		for (int i = 0; i < rank; ++i) {
			U[i] = Utils.getRandomUnitVector(N);
			V[i] = Utils.getRandomUnitVector(M);
			W[i] = Utils.getRandomUnitVector(D);
			totalU[i] = U[i].clone();
			totalV[i] = V[i].clone();
			totalW[i] = W[i].clone();
		}
	}
	
	public void averageParameters(int T) 
	{
		
		for (int i = 0; i < size; ++i) {
			params[i] += total[i]/T;
		}		

		for (int i = 0; i < sizeL; ++i) {
			paramsL[i] += totalL[i]/T;
		}		

		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < N; ++j) {
				U[i][j] += totalU[i][j]/T;
			}

		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < M; ++j) {
				V[i][j] += totalV[i][j]/T;
			}

		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < D; ++j) {
				W[i][j] += totalW[i][j]/T;
			}
	}
	
	public void unaverageParameters(int T) 
	{
		
		for (int i = 0; i < size; ++i) {
			params[i] -= total[i]/T;
		}	
		
		for (int i = 0; i < sizeL; ++i) {
			paramsL[i] -= totalL[i]/T;
		}	
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < N; ++j) {
				U[i][j] -= totalU[i][j]/T;
			}
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < M; ++j) {
				V[i][j] -= totalV[i][j]/T;
			}
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < D; ++j) {				
				W[i][j] -= totalW[i][j]/T;
			}
	}
	
	public void clearUVW() 
	{
		for (int i = 0; i < rank; ++i) {
			Arrays.fill(U[i], 0);
			Arrays.fill(V[i], 0);
			Arrays.fill(W[i], 0);
			Arrays.fill(totalU[i], 0);
			Arrays.fill(totalV[i], 0);
			Arrays.fill(totalW[i], 0);
			
		}
	}
	
	public void clearTheta() 
	{
		Arrays.fill(params, 0);
		Arrays.fill(total, 0);
		if (options.learnLabel) {
			Arrays.fill(paramsL, 0);
			Arrays.fill(totalL, 0);
		}
	}
	
	public void printUStat() 
	{
		float sum = 0;
		float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(U[i]);
			min = Math.min(min, Utils.min(U[i]));
			max = Math.max(max, Utils.max(U[i]));
		}
		System.out.printf(" |U|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printVStat() 
	{
		float sum = 0;
		float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(V[i]);
			min = Math.min(min, Utils.min(V[i]));
			max = Math.max(max, Utils.max(V[i]));
		}
		System.out.printf(" |V|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printWStat() 
	{
		float sum = 0;
		float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(W[i]);
			min = Math.min(min, Utils.min(W[i]));
			max = Math.max(max, Utils.max(W[i]));
		}
		System.out.printf(" |W|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printThetaStat() 
	{
		float sum = Utils.squaredSum(params);
		float min = Utils.min(params);
		float max = Utils.max(params);		
		System.out.printf(" |\u03b8|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void projectU(FeatureVector fv, float[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(U[r]);
	}
	
	public void projectV(FeatureVector fv, float[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(V[r]);
	}
	
	public float dotProduct(FeatureVector fv)
	{
		return fv.dotProduct(params);
	}
	
	public float dotProductL(FeatureVector fv)
	{
		return fv.dotProduct(paramsL);
	}
	
	public float dotProduct(float[] proju, float[] projv, int dist)
	{
		float sum = 0;
		int binDist = getBinnedDistance(dist);
		for (int r = 0; r < rank; ++r)
			sum += proju[r] * projv[r] * (W[r][binDist] + W[r][0]);
		return sum;
	}
	
	public float updateLabel(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, GlobalFeatureData gfd,
			int updCnt, int offset)
	{
		int N = gold.length;
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	float Fi = getLabelDis(actDeps, actLabs, predDeps, predLabs);
        	
    	FeatureVector dtl = lfd.getLabeledFeatureDifference(gold, pred);
    	float loss = - dtl.dotProduct(paramsL) + Fi;
        float l2norm = dtl.Squaredl2NormUnsafe();
    	
        float alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		float coeff = alpha;
    		float coeff2 = coeff * (1-updCnt);
    		for (int i = 0, K = dtl.size(); i < K; ++i) {
	    		int x = dtl.x(i);
	    		float z = dtl.value(i);
	    		paramsL[x] += coeff * z;
	    		totalL[x] += coeff2 * z;
    		}
    	}
    	
    	return loss;
	}
	
	
	public float update(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, GlobalFeatureData gfd,
			int updCnt, int offset)
	{
		int N = gold.length;
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	float Fi = getHammingDis(actDeps, actLabs, predDeps, predLabs);
    	
    	FeatureVector dt = lfd.getFeatureDifference(gold, pred);
    	dt.addEntries(gfd.getFeatureDifference(gold, pred));
    	    	
        float loss = - dt.dotProduct(params)*gamma + Fi;
        float l2norm = dt.Squaredl2NormUnsafe() * gamma * gamma;
    	
        int updId = (updCnt + offset) % 3;
        //if ( updId == 1 ) {
        	// update U
        	for (int k = 0; k < rank; ++k) {        		
        		FeatureVector dUk = getdU(k, lfd, actDeps, predDeps);
            	l2norm += dUk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
            	loss -= dUk.dotProduct(U[k]) * (1-gamma);
            	dU[k] = dUk;
        	}
        //} else if ( updId == 2 ) {
        	// update V
        	for (int k = 0; k < rank; ++k) {
        		FeatureVector dVk = getdV(k, lfd, actDeps, predDeps);
            	l2norm += dVk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
            	//loss -= dVk.dotProduct(V[k]) * (1-gamma);
            	dV[k] = dVk;
        	}        	
        //} else {
        	// update W
        	for (int k = 0; k < rank; ++k) {
        		FeatureVector dWk = getdW(k, lfd, actDeps, predDeps);
            	l2norm += dWk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
            	//loss -= dWk.dotProduct(W[k]) * (1-gamma);
            	dW[k] = dWk;
        	}   
        //}
        
        float alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		
    		{
    			// update theta
	    		float coeff = alpha * gamma;
	    		float coeff2 = coeff * (1-updCnt);
	    		for (int i = 0, K = dt.size(); i < K; ++i) {
		    		int x = dt.x(i);
		    		float z = dt.value(i);
		    		params[x] += coeff * z;
		    		total[x] += coeff2 * z;
	    		}
    		}
    		
    		//if ( updId == 1 ) 
    		{
    			// update U
    			float coeff = alpha * (1-gamma);
    			float coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dUk = dU[k];
            		for (int i = 0, K = dUk.size(); i < K; ++i) {
            			int x = dUk.x(i);
            			float z = dUk.value(i);
            			U[k][x] += coeff * z;
            			totalU[k][x] += coeff2 * z;
            		}
            	}
    		}	
    		//else if ( updId == 2 ) 
    		{
    			// update V
    			float coeff = alpha * (1-gamma);
    			float coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dVk = dV[k];
            		for (int i = 0, K = dVk.size(); i < K; ++i) {
            			int x = dVk.x(i);
            			float z = dVk.value(i);
            			V[k][x] += coeff * z;
            			totalV[k][x] += coeff2 * z;
            		}
            	}            	
    		} 
            //else 
    		{
    			// update W
    			float coeff = alpha * (1-gamma);
    			float coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dWk = dW[k];
            		for (int i = 0, K = dWk.size(); i < K; ++i) {
            			int x = dWk.x(i);
            			float z = dWk.value(i);
            			W[k][x] += coeff * z;
            			totalW[k][x] += coeff2 * z;
            		}
            	}  
    		}
    	}
    	
        return loss;
	}
	
	public void updateTheta(FeatureVector gold, FeatureVector pred, float loss,
			int updCnt) 
	{
		FeatureVector fv = new FeatureVector(size);
		fv.addEntries(gold);
		fv.addEntries(pred, -1.0f);
		
		float l2norm = fv.Squaredl2NormUnsafe();
		float alpha = loss/l2norm;
	    alpha = Math.min(C, alpha);
	    if (alpha > 0) {
			// update theta
    		float coeff = alpha;
    		float coeff2 = coeff * (1-updCnt);
    		for (int i = 0, K = fv.size(); i < K; ++i) {
	    		int x = fv.x(i);
	    		float z = fv.value(i);
	    		params[x] += coeff * z;
	    		total[x] += coeff2 * z;
    		}
	    }
	}
	
    private FeatureVector getdU(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	float[][] wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		float dotv = wpV[mod][k]; //wordFvs[mod].dotProduct(V[k]);    		
    		dU.addEntries(wordFvs[head], dotv * (W[k][0] + W[k][d]));
    		dU.addEntries(wordFvs[head2], - dotv * (W[k][0] + W[k][d2]));
    	}
    	return dU;
    }
    
    private FeatureVector getdV(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU = lfd.wpU;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV = new FeatureVector(M);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		float dotu2 = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		dV.addEntries(wordFvs[mod], dotu  * (W[k][0] + W[k][d])
    									- dotu2 * (W[k][0] + W[k][d2]));    		
    	}
    	return dV;
    }
    
    private FeatureVector getdW(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU = lfd.wpU, wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	float[] dW = new float[D];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		float dotu2 = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		float dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
    		dW[0] += (dotu - dotu2) * dotv;
    		dW[d] += dotu * dotv;
    		dW[d2] -= dotu2 * dotv;
    	}
    	
    	FeatureVector dW2 = new FeatureVector(D);
    	for (int i = 0; i < D; ++i)
    		dW2.addEntry(i, dW[i]);
    	return dW2;
    }
    
	public float getHammingDis(int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) 
	{
		float dis = 0;
		for (int i = 1; i < actDeps.length; ++i)
			//if (options.learnLabel) {
			//	if (labelLossType == 0) {
			//		if (actDeps[i] != predDeps[i]) dis += 1.0;
			//		if (actLabs[i] != predLabs[i]) dis += 1.0;
			//	} else if (actDeps[i] != predDeps[i] || actLabs[i] != predLabs[i]) dis += 1;
			//} else {
				if (actDeps[i] != predDeps[i]) dis += 1;
			//}
		return dis;
    }
	
	public float getLabelDis(int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) 
	{
		float dis = 0;
		for (int i = 1; i < actLabs.length; ++i) {
			assert(actDeps[i] == predDeps[i]);
			if (actLabs[i] != predLabs[i]) dis += 1;
		}
		return dis;
    }
    public int getBinnedDistance(int x) {
    	int y = x > 0 ? x : -x;
    	int dis = 0;
    	if (y > 10)
    		dis = 7;
    	else if (y > 5)
    		dis = 6;
    	else dis = y;
    	if (dis > d) dis = d;
    	return x > 0 ? dis : dis + d;
    }
}
