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
	public double C, gamma, gammaL;
	public int size, sizeL;
	public int rank;
	public int N, M, T, D;
	
	public float[] params, paramsL;
	public double[][] U, V, W, WL;
	public transient float[] total, totalL;
	public transient double[][] totalU, totalV, totalW, totalWL;
	
	public transient FeatureVector[] dU, dV, dW, dWL;
	
	public Parameters(DependencyPipe pipe, Options options) 
	{
        D = d * 2 + 1;
        T = pipe.types.length;
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
		gammaL = options.gammaLabel;
		rank = options.R;
		
		N = pipe.synFactory.numWordFeats;
		M = N;
		U = new double[rank][N];		
		V = new double[rank][M];
		W = new double[rank][D];
		WL = new double[rank][T];
		totalU = new double[rank][N];
		totalV = new double[rank][M];
		totalW = new double[rank][D];
		totalWL = new double[rank][T];
		dU = new FeatureVector[rank];
		dV = new FeatureVector[rank];
		dW = new FeatureVector[rank];
		dWL = new FeatureVector[rank];
	}
	
	public void randomlyInitUVWWL() 
	{
		for (int i = 0; i < rank; ++i) {
			U[i] = Utils.getRandomUnitVector(N);
			V[i] = Utils.getRandomUnitVector(M);
			W[i] = Utils.getRandomUnitVector(D);
			WL[i] = Utils.getRandomUnitVector(T);
			totalU[i] = U[i].clone();
			totalV[i] = V[i].clone();
			totalW[i] = W[i].clone();
			totalWL[i] = WL[i].clone();
		}
	}
	
	public void averageParameters(int L) 
	{
		
		for (int i = 0; i < size; ++i) {
			params[i] += total[i]/L;
		}		

		for (int i = 0; i < sizeL; ++i) {
			paramsL[i] += totalL[i]/L;
		}		

		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < N; ++j) {
				U[i][j] += totalU[i][j]/L;
			}

		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < M; ++j) {
				V[i][j] += totalV[i][j]/L;
			}

		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < D; ++j) {
				W[i][j] += totalW[i][j]/L;
			}
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < T; ++j) {
				WL[i][j] += totalWL[i][j]/L;
			}
	}
	
	public void unaverageParameters(int L) 
	{
		
		for (int i = 0; i < size; ++i) {
			params[i] -= total[i]/L;
		}	
		
		for (int i = 0; i < sizeL; ++i) {
			paramsL[i] -= totalL[i]/L;
		}	
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < N; ++j) {
				U[i][j] -= totalU[i][j]/L;
			}
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < M; ++j) {
				V[i][j] -= totalV[i][j]/L;
			}
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < D; ++j) {				
				W[i][j] -= totalW[i][j]/L;
			}
		
		for (int i = 0; i < rank; ++i)
			for (int j = 0; j < T; ++j) {
				WL[i][j] -= totalWL[i][j]/L;
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
			if (options.learnLabel) {
				Arrays.fill(WL[i], 0);
				Arrays.fill(totalWL[i], 0);
			}
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
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(U[i]);
			min = Math.min(min, Utils.min(U[i]));
			max = Math.max(max, Utils.max(U[i]));
		}
		System.out.printf(" |U|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printVStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(V[i]);
			min = Math.min(min, Utils.min(V[i]));
			max = Math.max(max, Utils.max(V[i]));
		}
		System.out.printf(" |V|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printWStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(W[i]);
			min = Math.min(min, Utils.min(W[i]));
			max = Math.max(max, Utils.max(W[i]));
		}
		System.out.printf(" |W|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printWLStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(WL[i]);
			min = Math.min(min, Utils.min(WL[i]));
			max = Math.max(max, Utils.max(WL[i]));
		}
		System.out.printf(" |WL|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	private double printdWStat() 
	{
		double sum = 0;
		for (int i = 0; i < rank; ++i) {
			sum += dW[i].Squaredl2NormUnsafe();
		}
		//System.out.printf(" |dW|^2: %f\n", sum);
		return sum;
	}
	
	private double printdWLStat() 
	{
		double sum = 0;
		for (int i = 0; i < rank; ++i) {
			sum += dWL[i].Squaredl2NormUnsafe();
		}
		//System.out.printf(" |dWL|^2: %f\n", sum);
		return sum;
	}
	
	private double WNorm()
	{
		double sum = 0;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(W[i]);
		}
		return sum;
	}
	
	private double WLNorm()
	{
		double sum = 0;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(WL[i]);
		}
		return sum;
	}
	
	public void printThetaStat() 
	{
		double sum = Utils.squaredSum(params);
		double min = Utils.min(params);
		double max = Utils.max(params);		
		System.out.printf(" |\u03b8|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void projectU(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(U[r]);
	}
	
	public void projectV(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(V[r]);
	}
	
	public double dotProduct(FeatureVector fv)
	{
		return fv.dotProduct(params);
	}
	
	public double dotProductL(FeatureVector fv)
	{
		return fv.dotProduct(paramsL);
	}
	
	public double dotProduct(double[] proju, double[] projv, int dist)
	{
		double sum = 0;
		int binDist = getBinnedDistance(dist);
		for (int r = 0; r < rank; ++r)
			sum += proju[r] * projv[r] * (W[r][binDist] + W[r][0]);
		return sum;
	}
	
	public double updateLabel(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, GlobalFeatureData gfd,
			int updCnt)
	{
		int N = gold.length;
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	double Fi = getLabelDis(actDeps, actLabs, predDeps, predLabs);
        	
    	FeatureVector dtl = lfd.getLabeledFeatureDifference(gold, pred);
    	double loss = - dtl.dotProduct(paramsL)*gammaL + Fi;
        double l2norm = dtl.Squaredl2NormUnsafe() * gammaL * gammaL;
    	
        // update U
    	for (int k = 0; k < rank; ++k) {        		
    		FeatureVector dUk = getdUL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dUk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);            	
        	loss -= dUk.dotProduct(U[k]) * (1-gammaL);
        	dU[k] = dUk;
    	}
    	// update V
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dVk = getdVL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dVk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
        	//loss -= dVk.dotProduct(V[k]) * (1-gammaL);
        	dV[k] = dVk;
    	}        	
        // update WL
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dWLk = getdWL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dWLk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
        	//loss -= dWLk.dotProduct(WL[k]) * (1-gammaL);
        	dWL[k] = dWLk;
    	}
        
        double alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		
    		{
    			// update thetaL
	    		double coeff = alpha * gammaL;
	    		double coeff2 = coeff * (1-updCnt);
	    		for (int i = 0, K = dtl.size(); i < K; ++i) {
		    		int x = dtl.x(i);
		    		double z = dtl.value(i);
		    		paramsL[x] += coeff * z;
		    		totalL[x] += coeff2 * z;
	    		}
    		}
    		
    		{
    			// update U
    			double coeff = alpha * (1-gammaL);
    			double coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dUk = dU[k];
            		for (int i = 0, K = dUk.size(); i < K; ++i) {
            			int x = dUk.x(i);
            			double z = dUk.value(i);
            			U[k][x] += coeff * z;
            			totalU[k][x] += coeff2 * z;
            		}
            	}
    		}	
    		{
    			// update V
    			double coeff = alpha * (1-gammaL);
    			double coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dVk = dV[k];
            		for (int i = 0, K = dVk.size(); i < K; ++i) {
            			int x = dVk.x(i);
            			double z = dVk.value(i);
            			V[k][x] += coeff * z;
            			totalV[k][x] += coeff2 * z;
            		}
            	}            	
    		} 
    		{
	    		// update WL
				double coeff = alpha * (1-gammaL);
				double coeff2 = coeff * (1-updCnt);
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dWLk = dWL[k];
	        		for (int i = 0, K = dWLk.size(); i < K; ++i) {
	        			int x = dWLk.x(i);
	        			double z = dWLk.value(i);
	        			WL[k][x] += coeff * z;
	        			totalWL[k][x] += coeff2 * z;
	        		}
	        	}
    		}
    	}
    	return loss;
	}
	
	
	public double update(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, GlobalFeatureData gfd,
			int updCnt, int offset)
	{
		int N = gold.length;
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	double Fi = getHammingDis(actDeps, actLabs, predDeps, predLabs);
    	
    	FeatureVector dt = lfd.getFeatureDifference(gold, pred);
    	dt.addEntries(gfd.getFeatureDifference(gold, pred));
    	    	
        double loss = - dt.dotProduct(params)*gamma + Fi;
        double l2norm = dt.Squaredl2NormUnsafe() * gamma * gamma;
    	
    	// update U
    	for (int k = 0; k < rank; ++k) {        		
    		FeatureVector dUk = getdU(k, lfd, actDeps, predDeps);
        	l2norm += dUk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
        	loss -= dUk.dotProduct(U[k]) * (1-gamma);
        	dU[k] = dUk;
    	}
    	// update V
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dVk = getdV(k, lfd, actDeps, predDeps);
        	l2norm += dVk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
        	//loss -= dVk.dotProduct(V[k]) * (1-gamma);
        	dV[k] = dVk;
    	}        	
    	// update W
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dWk = getdW(k, lfd, actDeps, predDeps);
        	l2norm += dWk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
        	//loss -= dWk.dotProduct(W[k]) * (1-gamma);
        	dW[k] = dWk;
    	}   
        
        double alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		
    		{
    			// update theta
	    		double coeff = alpha * gamma;
	    		double coeff2 = coeff * (1-updCnt);
	    		for (int i = 0, K = dt.size(); i < K; ++i) {
		    		int x = dt.x(i);
		    		double z = dt.value(i);
		    		params[x] += coeff * z;
		    		total[x] += coeff2 * z;
	    		}
    		}
    		
    		{
    			// update U
    			double coeff = alpha * (1-gamma);
    			double coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dUk = dU[k];
            		for (int i = 0, K = dUk.size(); i < K; ++i) {
            			int x = dUk.x(i);
            			double z = dUk.value(i);
            			U[k][x] += coeff * z;
            			totalU[k][x] += coeff2 * z;
            		}
            	}
    		}	
    		{
    			// update V
    			double coeff = alpha * (1-gamma);
    			double coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dVk = dV[k];
            		for (int i = 0, K = dVk.size(); i < K; ++i) {
            			int x = dVk.x(i);
            			double z = dVk.value(i);
            			V[k][x] += coeff * z;
            			totalV[k][x] += coeff2 * z;
            		}
            	}            	
    		} 
    		{
    			// update W
    			double coeff = alpha * (1-gamma);
    			double coeff2 = coeff * (1-updCnt);
            	for (int k = 0; k < rank; ++k) {
            		FeatureVector dWk = dW[k];
            		for (int i = 0, K = dWk.size(); i < K; ++i) {
            			int x = dWk.x(i);
            			double z = dWk.value(i);
            			W[k][x] += coeff * z;
            			totalW[k][x] += coeff2 * z;
            		}
            	}  
    		}
    	}
        return loss;
	}
	
	public void updateTheta(FeatureVector gold, FeatureVector pred, double loss,
			int updCnt) 
	{
		FeatureVector fv = new FeatureVector(size);
		fv.addEntries(gold);
		fv.addEntries(pred, -1.0);
		
		double l2norm = fv.Squaredl2NormUnsafe();
		double alpha = loss/l2norm;
	    alpha = Math.min(C, alpha);
	    if (alpha > 0) {
			// update theta
    		double coeff = alpha;
    		double coeff2 = coeff * (1-updCnt);
    		for (int i = 0, K = fv.size(); i < K; ++i) {
	    		int x = fv.x(i);
	    		double z = fv.value(i);
	    		params[x] += coeff * z;
	    		total[x] += coeff2 * z;
    		}
	    }
	}
	
    private FeatureVector getdU(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		double dotv = wpV[mod][k]; //wordFvs[mod].dotProduct(V[k]);    		
    		dU.addEntries(wordFvs[head], dotv * (W[k][0] + W[k][d]));
    		dU.addEntries(wordFvs[head2], - dotv * (W[k][0] + W[k][d2]));
    	}
    	return dU;
    }
    
    private FeatureVector getdUL(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	double[][] wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		assert(actDeps[mod] == predDeps[mod]);
    		int head  = actDeps[mod];
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		if (lab == lab2) continue;
    		double dotv = wpV[mod][k]; //wordFvs[mod].dotProduct(V[k]);    		
    		dU.addEntries(wordFvs[head], dotv*WL[k][lab] - dotv*WL[k][lab2]);
    	}
    	return dU;
    }
    
    private FeatureVector getdV(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	double[][] wpU = lfd.wpU;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV = new FeatureVector(M);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		double dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		double dotu2 = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		dV.addEntries(wordFvs[mod], dotu  * (W[k][0] + W[k][d])
    									- dotu2 * (W[k][0] + W[k][d2]));    		
    	}
    	return dV;
    }
    
    private FeatureVector getdVL(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	double[][] wpU = lfd.wpU;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV = new FeatureVector(M);
    	for (int mod = 1; mod < L; ++mod) {
    		assert(actDeps[mod] == predDeps[mod]);
    		int head  = actDeps[mod];
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		if (lab == lab2) continue;
    		double dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		dV.addEntries(wordFvs[mod], dotu*WL[k][lab] - dotu*WL[k][lab2]);    		
    	}
    	return dV;
    }
    
    private FeatureVector getdW(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	double[][] wpU = lfd.wpU, wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	double[] dW = new double[D];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		double dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		double dotu2 = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		double dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
    		dW[0] += (dotu - dotu2) * dotv;
    		dW[d] += dotu * dotv;
    		dW[d2] -= dotu2 * dotv;
    	}
    	
    	FeatureVector dW2 = new FeatureVector(D);
    	for (int i = 0; i < D; ++i)
    		dW2.addEntry(i, dW[i]);
    	return dW2;
    }
    
    private FeatureVector getdWL(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	double[][] wpU = lfd.wpU, wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	double[] dWL = new double[T];
    	for (int mod = 1; mod < L; ++mod) {
    		assert(actDeps[mod] == predDeps[mod]);
    		int head = actDeps[mod];
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		if (lab == lab2) continue;
    		double dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		double dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
    		dWL[lab] += dotu * dotv;
    		dWL[lab2] -= dotu * dotv;
    	}
    	
    	FeatureVector dWL2 = new FeatureVector(T);
    	for (int i = 0; i < T; ++i)
    		dWL2.addEntry(i, dWL[i]);
    	return dWL2;
    }
    
	public double getHammingDis(int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) 
	{
		double dis = 0;
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
	
	public double getLabelDis(int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) 
	{
		double dis = 0;
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
