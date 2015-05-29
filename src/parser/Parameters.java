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
	public double C, gamma, gammaLabel;
	public int size, sizeL;
	public int rank;
	public int N, M, T, D;
	
	public float[] params, paramsL;
	public double[][] U, V, W;
	public double[][] U2, V2, W2, X2;
	public transient float[] total, totalL;
	public transient double[][] totalU, totalV, totalW;
	public transient double[][] totalU2, totalV2, totalW2, totalX2;
	
	public transient FeatureVector[] dU, dV, dW;
	public transient FeatureVector[] dU2, dV2, dW2, dX2;
	
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
		U = new double[rank][N];		
		V = new double[rank][M];
		W = new double[rank][D];
		totalU = new double[rank][N];
		totalV = new double[rank][M];
		totalW = new double[rank][D];
		dU = new FeatureVector[rank];
		dV = new FeatureVector[rank];
		dW = new FeatureVector[rank];
		
		if (options.useGP) {
			U2 = new double[rank][N];
			V2 = new double[rank][N];
			W2 = new double[rank][D];
			X2 = new double[rank][N];
			totalU2 = new double[rank][N];
			totalV2 = new double[rank][N];
			totalW2 = new double[rank][D];
			totalX2 = new double[rank][N];
			dU2 = new FeatureVector[rank];
			dV2 = new FeatureVector[rank];
			dW2 = new FeatureVector[rank];
			dX2 = new FeatureVector[rank];
		}
	}
	
	public void randomlyInitTensor() 
	{
 		for (int i = 0; i < rank; ++i) {
			//U[i] = Utils.getRandomNormVector(N, 1);
			//V[i] = Utils.getRandomNormVector(M, 1);
			//W[i] = Utils.getRandomNormVector(D, 1);
 			U[i] = Utils.getRandomRangeVector(N,0.1);
			V[i] = Utils.getRandomRangeVector(M,0.1);
			W[i] = Utils.getRandomRangeVector(D,0.1);
			totalU[i] = U[i].clone();
			totalV[i] = V[i].clone();
			totalW[i] = W[i].clone();
			
			if (options.useGP) {
				//U2[i] = Utils.getRandomNormVector(N, 10);
				//V2[i] = Utils.getRandomNormVector(N, 10);
				//W2[i] = Utils.getRandomNormVector(D, 1);
				//X2[i] = Utils.getRandomNormVector(N, 10);
				U2[i] = Utils.getRandomRangeVector(N,0.1);
				V2[i] = Utils.getRandomRangeVector(N,0.1);
				W2[i] = Utils.getRandomRangeVector(D,0.1);
				X2[i] = Utils.getRandomRangeVector(N,0.1);
				totalU2[i] = U2[i].clone();
				totalV2[i] = V2[i].clone();
				totalW2[i] = W2[i].clone();
				totalX2[i] = X2[i].clone();
			}
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
		
		if (options.useGP) {
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					U2[i][j] += totalU2[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					V2[i][j] += totalV2[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < D; ++j) {
					W2[i][j] += totalW2[i][j]/T;
				}
			
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					X2[i][j] += totalX2[i][j]/T;
				}
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
		
		if (options.useGP) {
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					U2[i][j] -= totalU2[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					V2[i][j] -= totalV2[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < D; ++j) {
					W2[i][j] -= totalW2[i][j]/T;
				}
			
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					X2[i][j] -= totalX2[i][j]/T;
				}
		}
	}
	
	public void clearTensor() 
	{
		for (int i = 0; i < rank; ++i) {
			Arrays.fill(U[i], 0);
			Arrays.fill(V[i], 0);
			Arrays.fill(W[i], 0);
			Arrays.fill(totalU[i], 0);
			Arrays.fill(totalV[i], 0);
			Arrays.fill(totalW[i], 0);
			
			if (options.useGP) {
				Arrays.fill(U2[i], 0);
				Arrays.fill(V2[i], 0);
				Arrays.fill(W2[i], 0);
				Arrays.fill(X2[i], 0);
				Arrays.fill(totalU2[i], 0);
				Arrays.fill(totalV2[i], 0);
				Arrays.fill(totalW2[i], 0);
				Arrays.fill(totalX2[i], 0);
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
	
	public void printU2Stat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(U2[i]);
			min = Math.min(min, Utils.min(U2[i]));
			max = Math.max(max, Utils.max(U2[i]));
		}
		System.out.printf(" |U2|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printV2Stat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(V2[i]);
			min = Math.min(min, Utils.min(V2[i]));
			max = Math.max(max, Utils.max(V2[i]));
		}
		System.out.printf(" |V2|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printW2Stat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(W2[i]);
			min = Math.min(min, Utils.min(W2[i]));
			max = Math.max(max, Utils.max(W2[i]));
		}
		System.out.printf(" |W2|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printX2Stat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(X2[i]);
			min = Math.min(min, Utils.min(X2[i]));
			max = Math.max(max, Utils.max(X2[i]));
		}
		System.out.printf(" |X2|^2: %f min: %f\tmax: %f%n", sum, min, max);
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
	
	public void projectU2(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(U2[r]);
	}
	
	public void projectV2(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(V2[r]);
	}
	
	public void projectX2(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(X2[r]);
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
	
	public double dotProduct2(double[] proju, double[] projv, int dist, double[] projx)
	{
		double sum = 0;
		int binDist = getBinnedDistance(dist);
		for (int r = 0; r < rank; ++r)
			sum += proju[r] * projv[r] * (W2[r][binDist] + W2[r][0]) * projx[r];
		return sum;
	}
	
	public double updateLabel(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, GlobalFeatureData gfd,
			int updCnt, int offset)
	{
		int N = gold.length;
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	double Fi = getLabelDis(actDeps, actLabs, predDeps, predLabs);
        	
    	FeatureVector dtl = lfd.getLabeledFeatureDifference(gold, pred);
    	double loss = - dtl.dotProduct(paramsL) + Fi;
        double l2norm = dtl.Squaredl2NormUnsafe();
    	
        double alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		double coeff = alpha;
    		double coeff2 = coeff * (1-updCnt);
    		for (int i = 0, K = dtl.size(); i < K; ++i) {
	    		int x = dtl.x(i);
	    		double z = dtl.value(i);
	    		paramsL[x] += coeff * z;
	    		totalL[x] += coeff2 * z;
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
        	dV[k] = dVk;
    	}        	

    	// update W
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dWk = getdW(k, lfd, actDeps, predDeps);
        	l2norm += dWk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
        	dW[k] = dWk;
    	}   
    	
    	if (options.useGP) {
	    	// update U2
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dU2k = getdU2(k, lfd, actDeps, predDeps);
	        	l2norm += dU2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
	        	loss -= dU2k.dotProduct(U2[k]) * (1-gamma);
	        	dU2[k] = dU2k;
	    	}
	    	
	    	// update V2
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dV2k = getdV2(k, lfd, actDeps, predDeps);
	        	l2norm += dV2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dV2[k] = dV2k;
	    	}        	
	
	    	// update W2
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dW2k = getdW2(k, lfd, actDeps, predDeps);
	        	l2norm += dW2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dW2[k] = dW2k;
	    	}
	    	
	    	// update X2
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dX2k = getdX2(k, lfd, actDeps, predDeps);
	        	l2norm += dX2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dX2[k] = dX2k;
	    	}
    	}

        double alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		double coeff, coeff2;

    		coeff = alpha * gamma;
    		coeff2 = coeff * (1-updCnt);
    		
			// update theta
    		for (int i = 0, K = dt.size(); i < K; ++i) {
	    		int x = dt.x(i);
	    		double z = dt.value(i);
	    		params[x] += coeff * z;
	    		total[x] += coeff2 * z;
    		}

    		coeff = alpha * (1-gamma);
			coeff2 = coeff * (1-updCnt);
			
			// update U
        	for (int k = 0; k < rank; ++k) {
        		FeatureVector dUk = dU[k];
        		for (int i = 0, K = dUk.size(); i < K; ++i) {
        			int x = dUk.x(i);
        			double z = dUk.value(i);
        			U[k][x] += coeff * z;
        			totalU[k][x] += coeff2 * z;
        		}
        	}

			// update V
        	for (int k = 0; k < rank; ++k) {
        		FeatureVector dVk = dV[k];
        		for (int i = 0, K = dVk.size(); i < K; ++i) {
        			int x = dVk.x(i);
        			double z = dVk.value(i);
        			V[k][x] += coeff * z;
        			totalV[k][x] += coeff2 * z;
        		}
        	}            	

			// update W
        	for (int k = 0; k < rank; ++k) {
        		FeatureVector dWk = dW[k];
        		for (int i = 0, K = dWk.size(); i < K; ++i) {
        			int x = dWk.x(i);
        			double z = dWk.value(i);
        			W[k][x] += coeff * z;
        			totalW[k][x] += coeff2 * z;
        		}
        	}
        	
        	if (options.useGP) {
	        	// update U2
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dU2k = dU2[k];
	        		for (int i = 0, K = dU2k.size(); i < K; ++i) {
	        			int x = dU2k.x(i);
	        			double z = dU2k.value(i);
	        			U2[k][x] += coeff * z;
	        			totalU2[k][x] += coeff2 * z;
	        		}
	        	}
	        	
	        	// update V2
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dV2k = dV2[k];
	        		for (int i = 0, K = dV2k.size(); i < K; ++i) {
	        			int x = dV2k.x(i);
	        			double z = dV2k.value(i);
	        			V2[k][x] += coeff * z;
	        			totalV2[k][x] += coeff2 * z;
	        		}
	        	} 
	        	
	        	// update W2
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dW2k = dW2[k];
	        		for (int i = 0, K = dW2k.size(); i < K; ++i) {
	        			int x = dW2k.x(i);
	        			double z = dW2k.value(i);
	        			W2[k][x] += coeff * z;
	        			totalW2[k][x] += coeff2 * z;
	        		}
	        	}
	        	
	        	// update X2
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dX2k = dX2[k];
	        		for (int i = 0, K = dX2k.size(); i < K; ++i) {
	        			int x = dX2k.x(i);
	        			double z = dX2k.value(i);
	        			X2[k][x] += coeff * z;
	        			totalX2[k][x] += coeff2 * z;
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
    
    private FeatureVector getdU2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpV2 = lfd.wpV2, wpX2 = lfd.wpX2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		if (gp != -1)
    			dU2.addEntries(wordFvs[head], wpV2[mod][k] * (W2[k][0] + W2[k][d]) * wpX2[gp][k]);
    		if (gp2 != -1)
    			dU2.addEntries(wordFvs[head2], - wpV2[mod][k] * (W2[k][0] + W2[k][d2]) * wpX2[gp2][k]);
    	}
    	return dU2;
    }
    
    private FeatureVector getdV2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpU2 = lfd.wpU2, wpX2 = lfd.wpX2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		if (gp != -1)
    			dV2.addEntries(wordFvs[mod], wpU2[head][k] * (W2[k][0] + W2[k][d]) * wpX2[gp][k]);
    		if (gp2 != -1)
    			dV2.addEntries(wordFvs[mod], - wpU2[head2][k] * (W2[k][0] + W2[k][d2]) * wpX2[gp2][k]);
    	}
    	return dV2;
    }
    
    private FeatureVector getdW2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	double[][] wpU2 = lfd.wpU2, wpV2 = lfd.wpV2, wpX2 = lfd.wpX2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	double[] dW2 = new double[D];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		if (gp != -1) {
    			dW2[0] += wpU2[head][k] * wpV2[mod][k] * wpX2[gp][k];
    			dW2[d] += wpU2[head][k] * wpV2[mod][k] * wpX2[gp][k];
    		}
    		if (gp2 != -1) {
    			dW2[0] -= wpU2[head2][k] * wpV2[mod][k] * wpX2[gp2][k];
    			dW2[d2] -= wpU2[head2][k] * wpV2[mod][k] * wpX2[gp2][k];
    		}
    	}
    	
    	FeatureVector fdW2 = new FeatureVector(D);
    	for (int i = 0; i < D; ++i)
    		fdW2.addEntry(i, dW2[i]);
    	return fdW2;
    }
    
    private FeatureVector getdX2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpU2 = lfd.wpU2, wpV2 = lfd.wpV2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dX2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = getBinnedDistance(head-mod);
    		int d2 = getBinnedDistance(head2-mod);
    		if (gp != -1)
    			dX2.addEntries(wordFvs[gp], wpU2[head][k] * wpV2[mod][k] * (W2[k][0] + W2[k][d]));
    		if (gp2 != -1)
    			dX2.addEntries(wordFvs[gp2], - wpU2[head2][k] * wpV2[mod][k] * (W2[k][0] + W2[k][d2]));
    	}
    	return dX2;
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
