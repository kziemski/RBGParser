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
	public int N, M, T, D, D2;
	
	public float[] params, paramsL;
	public double[][] U, V, W;
	public double[][] U2g, V2g, W2g, X2g;
	public double[][] U2s, V2s, W2s, X2s;
	public transient float[] total, totalL;
	public transient double[][] totalU, totalV, totalW;
	public transient double[][] totalU2g, totalV2g, totalW2g, totalX2g;
	public transient double[][] totalU2s, totalV2s, totalW2s, totalX2s;
	
	public transient FeatureVector[] dU, dV, dW;
	public transient FeatureVector[] dU2g, dV2g, dW2g, dX2g;
	public transient FeatureVector[] dU2s, dV2s, dW2s, dX2s;
	
	public Parameters(DependencyPipe pipe, Options options) 
	{
		 //T = pipe.types.length;
        D = d * 2 + 1;
        D2 = 5;
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
			U2g = new double[rank][N];
			V2g = new double[rank][N];
			W2g = new double[rank][D2];
			X2g = new double[rank][N];
			totalU2g = new double[rank][N];
			totalV2g = new double[rank][N];
			totalW2g = new double[rank][D2];
			totalX2g = new double[rank][N];
			dU2g = new FeatureVector[rank];
			dV2g = new FeatureVector[rank];
			dW2g = new FeatureVector[rank];
			dX2g = new FeatureVector[rank];
		}
		
		if (options.useCS) {
			U2s = new double[rank][N];
			V2s = new double[rank][N];
			W2s = new double[rank][D2];
			X2s = new double[rank][N];
			totalU2s = new double[rank][N];
			totalV2s = new double[rank][N];
			totalW2s = new double[rank][D2];
			totalX2s = new double[rank][N];
			dU2s = new FeatureVector[rank];
			dV2s = new FeatureVector[rank];
			dW2s = new FeatureVector[rank];
			dX2s = new FeatureVector[rank];
		}
	}
	
	public void randomlyInitTensor() 
	{
 		for (int i = 0; i < rank; ++i) {
			U[i] = Utils.getRandomNormVector(N, 1);
			V[i] = Utils.getRandomNormVector(M, 1);
			W[i] = Utils.getRandomNormVector(D, 1);
 			//U[i] = Utils.getRandomRangeVector(N,0.01);
			//V[i] = Utils.getRandomRangeVector(M,0.01);
			//W[i] = Utils.getRandomRangeVector(D,0.01);
			totalU[i] = U[i].clone();
			totalV[i] = V[i].clone();
			totalW[i] = W[i].clone();
			
			if (options.useGP) {
				U2g[i] = Utils.getRandomNormVector(N, 1);
				V2g[i] = Utils.getRandomNormVector(N, 1);
				W2g[i] = Utils.getRandomNormVector(D2, 1); 
				X2g[i] = Utils.getRandomNormVector(N, 1);
				//U2g[i] = Utils.getRandomRangeVector(N,0.01);
				//V2g[i] = Utils.getRandomRangeVector(N,0.01);
				//W2g[i] = Utils.getRandomRangeVector(D2,0.01);
				//X2g[i] = Utils.getRandomRangeVector(N,0.01);
				totalU2g[i] = U2g[i].clone();
				totalV2g[i] = V2g[i].clone();
				totalW2g[i] = W2g[i].clone();
				totalX2g[i] = X2g[i].clone();
			}
			
			if (options.useCS) {
				U2s[i] = Utils.getRandomNormVector(N, 1);
				V2s[i] = Utils.getRandomNormVector(N, 1);
				W2s[i] = Utils.getRandomNormVector(D2, 1); 
				X2s[i] = Utils.getRandomNormVector(N, 1);
				//U2s[i] = Utils.getRandomRangeVector(N,0.01);
				//V2s[i] = Utils.getRandomRangeVector(N,0.01);
				//W2s[i] = Utils.getRandomRangeVector(D2,0.01);
				//X2s[i] = Utils.getRandomRangeVector(N,0.01);
				totalU2s[i] = U2s[i].clone();
				totalV2s[i] = V2s[i].clone();
				totalW2s[i] = W2s[i].clone();
				totalX2s[i] = X2s[i].clone();
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
					U2g[i][j] += totalU2g[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					V2g[i][j] += totalV2g[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < D2; ++j) {
					W2g[i][j] += totalW2g[i][j]/T;
				}
			
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					X2g[i][j] += totalX2g[i][j]/T;
				}
		}
		
		if (options.useCS) {
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					U2s[i][j] += totalU2s[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					V2s[i][j] += totalV2s[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < D2; ++j) {
					W2s[i][j] += totalW2s[i][j]/T;
				}
			
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					X2s[i][j] += totalX2s[i][j]/T;
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
					U2g[i][j] -= totalU2g[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					V2g[i][j] -= totalV2g[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < D2; ++j) {
					W2g[i][j] -= totalW2g[i][j]/T;
				}
			
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					X2g[i][j] -= totalX2g[i][j]/T;
				}
		}
		
		if (options.useCS) {
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					U2s[i][j] -= totalU2s[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					V2s[i][j] -= totalV2s[i][j]/T;
				}
	
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < D2; ++j) {
					W2s[i][j] -= totalW2s[i][j]/T;
				}
			
			for (int i = 0; i < rank; ++i)
				for (int j = 0; j < N; ++j) {
					X2s[i][j] -= totalX2s[i][j]/T;
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
				Arrays.fill(U2g[i], 0);
				Arrays.fill(V2g[i], 0);
				Arrays.fill(W2g[i], 0);
				Arrays.fill(X2g[i], 0);
				Arrays.fill(totalU2g[i], 0);
				Arrays.fill(totalV2g[i], 0);
				Arrays.fill(totalW2g[i], 0);
				Arrays.fill(totalX2g[i], 0);
			}
			
			if (options.useCS) {
				Arrays.fill(U2s[i], 0);
				Arrays.fill(V2s[i], 0);
				Arrays.fill(W2s[i], 0);
				Arrays.fill(X2s[i], 0);
				Arrays.fill(totalU2s[i], 0);
				Arrays.fill(totalV2s[i], 0);
				Arrays.fill(totalW2s[i], 0);
				Arrays.fill(totalX2s[i], 0);
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
	
	public void printU2gStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(U2g[i]);
			min = Math.min(min, Utils.min(U2g[i]));
			max = Math.max(max, Utils.max(U2g[i]));
		}
		System.out.printf(" |U2g|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printV2gStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(V2g[i]);
			min = Math.min(min, Utils.min(V2g[i]));
			max = Math.max(max, Utils.max(V2g[i]));
		}
		System.out.printf(" |V2g|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printW2gStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(W2g[i]);
			min = Math.min(min, Utils.min(W2g[i]));
			max = Math.max(max, Utils.max(W2g[i]));
		}
		System.out.printf(" |W2g|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printX2gStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(X2g[i]);
			min = Math.min(min, Utils.min(X2g[i]));
			max = Math.max(max, Utils.max(X2g[i]));
		}
		System.out.printf(" |X2g|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printU2sStat()
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(U2s[i]);
			min = Math.min(min, Utils.min(U2s[i]));
			max = Math.max(max, Utils.max(U2s[i]));
		}
		System.out.printf(" |U2s|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printV2sStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(V2s[i]);
			min = Math.min(min, Utils.min(V2s[i]));
			max = Math.max(max, Utils.max(V2s[i]));
		}
		System.out.printf(" |V2s|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printW2sStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(W2s[i]);
			min = Math.min(min, Utils.min(W2s[i]));
			max = Math.max(max, Utils.max(W2s[i]));
		}
		System.out.printf(" |W2s|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void printX2sStat() 
	{
		double sum = 0;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < rank; ++i) {
			sum += Utils.squaredSum(X2s[i]);
			min = Math.min(min, Utils.min(X2s[i]));
			max = Math.max(max, Utils.max(X2s[i]));
		}
		System.out.printf(" |X2s|^2: %f min: %f\tmax: %f%n", sum, min, max);
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
	
	public void projectU2g(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(U2g[r]);
	}
	
	public void projectV2g(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(V2g[r]);
	}
	
	public void projectX2g(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(X2g[r]);
	}
	
	public void projectU2s(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(U2s[r]);
	}
	
	public void projectV2s(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(V2s[r]);
	}
	
	public void projectX2s(FeatureVector fv, double[] proj) 
	{
		for (int r = 0; r < rank; ++r) 
			proj[r] = fv.dotProduct(X2s[r]);
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
	
	public double dotProduct2g(double[] projU2g, double[] projV2g, int dirFlag, double[] projX2g)
	{
		double sum = 0;
		for (int r = 0; r < rank; ++r)
			sum += projU2g[r] * projV2g[r] * (W2g[r][dirFlag] + W2g[r][0]) * projX2g[r];
		return sum;
	}
	
	public double dotProduct2s(double[] projU2s, double[] projV2s, int dirFlag, double[] projX2s)
	{
		double sum = 0;
		for (int r = 0; r < rank; ++r)
			sum += projU2s[r] * projV2s[r] * (W2s[r][dirFlag] + W2s[r][0]) * projX2s[r];
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
	    	// update U2g
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dU2gk = getdU2g(k, lfd, actDeps, predDeps);
	        	l2norm += dU2gk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
	        	loss -= dU2gk.dotProduct(U2g[k]) * (1-gamma);
	        	dU2g[k] = dU2gk;
	    	}
	    	
	    	// update V2g
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dV2gk = getdV2g(k, lfd, actDeps, predDeps);
	        	l2norm += dV2gk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dV2g[k] = dV2gk;
	    	}        	
	
	    	// update W2g
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dW2gk = getdW2g(k, lfd, actDeps, predDeps);
	        	l2norm += dW2gk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dW2g[k] = dW2gk;
	    	}
	    	
	    	// update X2g
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dX2gk = getdX2g(k, lfd, actDeps, predDeps);
	        	l2norm += dX2gk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dX2g[k] = dX2gk;
	    	}
    	}
    	
    	if (options.useCS) {
	    	// update U2s
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dU2sk = getdU2s(k, lfd, actDeps, predDeps);
	        	l2norm += dU2sk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
	        	loss -= dU2sk.dotProduct(U2s[k]) * (1-gamma);
	        	dU2s[k] = dU2sk;
	    	}
	    	
	    	// update V2s
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dV2sk = getdV2s(k, lfd, actDeps, predDeps);
	        	l2norm += dV2sk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dV2s[k] = dV2sk;
	    	}        	
	
	    	// update W2s
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dW2sk = getdW2s(k, lfd, actDeps, predDeps);
	        	l2norm += dW2sk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dW2s[k] = dW2sk;
	    	}
	    	
	    	// update X2s
	    	for (int k = 0; k < rank; ++k) {
	    		FeatureVector dX2sk = getdX2s(k, lfd, actDeps, predDeps);
	        	l2norm += dX2sk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dX2s[k] = dX2sk;
	    	}
    	}
    	
    	
    	// check
//    	double goldScore = lfd.getScore(gold.heads);
//    	double predScore = lfd.getScore(pred.heads);
//    	if (Math.abs(loss-(predScore-goldScore+Fi)) > 1e-4)
//    		System.out.println("Oh, no!");

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
	        	// update U2g
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dU2gk = dU2g[k];
	        		for (int i = 0, K = dU2gk.size(); i < K; ++i) {
	        			int x = dU2gk.x(i);
	        			double z = dU2gk.value(i);
	        			U2g[k][x] += coeff * z;
	        			totalU2g[k][x] += coeff2 * z;
	        		}
	        	}
	        	
	        	// update V2g
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dV2gk = dV2g[k];
	        		for (int i = 0, K = dV2gk.size(); i < K; ++i) {
	        			int x = dV2gk.x(i);
	        			double z = dV2gk.value(i);
	        			V2g[k][x] += coeff * z;
	        			totalV2g[k][x] += coeff2 * z;
	        		}
	        	} 
	        	
	        	// update W2g
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dW2gk = dW2g[k];
	        		for (int i = 0, K = dW2gk.size(); i < K; ++i) {
	        			int x = dW2gk.x(i);
	        			double z = dW2gk.value(i);
	        			W2g[k][x] += coeff * z;
	        			totalW2g[k][x] += coeff2 * z;
	        		}
	        	}
	        	
	        	// update X2g
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dX2gk = dX2g[k];
	        		for (int i = 0, K = dX2gk.size(); i < K; ++i) {
	        			int x = dX2gk.x(i);
	        			double z = dX2gk.value(i);
	        			X2g[k][x] += coeff * z;
	        			totalX2g[k][x] += coeff2 * z;
	        		}
	        	}
        	}
        	
        	if (options.useCS) {
	        	// update U2s
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dU2sk = dU2s[k];
	        		for (int i = 0, K = dU2sk.size(); i < K; ++i) {
	        			int x = dU2sk.x(i);
	        			double z = dU2sk.value(i);
	        			U2s[k][x] += coeff * z;
	        			totalU2s[k][x] += coeff2 * z;
	        		}
	        	}
	        	
	        	// update V2s
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dV2sk = dV2s[k];
	        		for (int i = 0, K = dV2sk.size(); i < K; ++i) {
	        			int x = dV2sk.x(i);
	        			double z = dV2sk.value(i);
	        			V2s[k][x] += coeff * z;
	        			totalV2s[k][x] += coeff2 * z;
	        		}
	        	} 
	        	
	        	// update W2s
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dW2sk = dW2s[k];
	        		for (int i = 0, K = dW2sk.size(); i < K; ++i) {
	        			int x = dW2sk.x(i);
	        			double z = dW2sk.value(i);
	        			W2s[k][x] += coeff * z;
	        			totalW2s[k][x] += coeff2 * z;
	        		}
	        	}
	        	
	        	// update X2s
	        	for (int k = 0; k < rank; ++k) {
	        		FeatureVector dX2sk = dX2s[k];
	        		for (int i = 0, K = dX2sk.size(); i < K; ++i) {
	        			int x = dX2sk.x(i);
	        			double z = dX2sk.value(i);
	        			X2s[k][x] += coeff * z;
	        			totalX2s[k][x] += coeff2 * z;
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
    		double dotU2g = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		dV.addEntries(wordFvs[mod], dotu  * (W[k][0] + W[k][d])
    									- dotU2g * (W[k][0] + W[k][d2]));    		
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
    		double dotU2g = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		double dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
    		dW[0] += (dotu - dotU2g) * dotv;
    		dW[d] += dotu * dotv;
    		dW[d2] -= dotU2g * dotv;
    	}
    	
    	FeatureVector dW2g = new FeatureVector(D);
    	for (int i = 0; i < D; ++i)
    		dW2g.addEntry(i, dW[i]);
    	return dW2g;
    }
    
    private FeatureVector getdU2g(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpV2g = lfd.wpV2g, wpX2g = lfd.wpX2g;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU2g = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = (((gp < head ? 0 : 1) << 1) | (head < mod ? 0 : 1)) + 1;
    		int d2 = (((gp2 < head2 ? 0 : 1) << 1) | (head2 < mod ? 0 : 1)) + 1;
    		if (gp != -1)
    			dU2g.addEntries(wordFvs[head], wpV2g[mod][k] * (W2g[k][0] + W2g[k][d]) * wpX2g[gp][k]);
    		if (gp2 != -1)
    			dU2g.addEntries(wordFvs[head2], - wpV2g[mod][k] * (W2g[k][0] + W2g[k][d2]) * wpX2g[gp2][k]);
    	}
    	return dU2g;
    }
    
    private FeatureVector getdV2g(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpU2g = lfd.wpU2g, wpX2g = lfd.wpX2g;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV2g = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = (((gp < head ? 0 : 1) << 1) | (head < mod ? 0 : 1)) + 1;
    		int d2 = (((gp2 < head2 ? 0 : 1) << 1) | (head2 < mod ? 0 : 1)) + 1;
    		if (gp != -1)
    			dV2g.addEntries(wordFvs[mod], wpU2g[head][k] * (W2g[k][0] + W2g[k][d]) * wpX2g[gp][k]);
    		if (gp2 != -1)
    			dV2g.addEntries(wordFvs[mod], - wpU2g[head2][k] * (W2g[k][0] + W2g[k][d2]) * wpX2g[gp2][k]);
    	}
    	return dV2g;
    }
    
    private FeatureVector getdW2g(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	double[][] wpU2g = lfd.wpU2g, wpV2g = lfd.wpV2g, wpX2g = lfd.wpX2g;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	double[] dW2g = new double[D2];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = (((gp < head ? 0 : 1) << 1) | (head < mod ? 0 : 1)) + 1;
    		int d2 = (((gp2 < head2 ? 0 : 1) << 1) | (head2 < mod ? 0 : 1)) + 1;
    		if (gp != -1) {
    			dW2g[0] += wpU2g[head][k] * wpV2g[mod][k] * wpX2g[gp][k];
    			dW2g[d] += wpU2g[head][k] * wpV2g[mod][k] * wpX2g[gp][k];
    		}
    		if (gp2 != -1) {
    			dW2g[0] -= wpU2g[head2][k] * wpV2g[mod][k] * wpX2g[gp2][k];
    			dW2g[d2] -= wpU2g[head2][k] * wpV2g[mod][k] * wpX2g[gp2][k];
    		}
    	}
    	
    	FeatureVector fdW2g = new FeatureVector(D2);
    	for (int i = 0; i < D2; ++i)
    		fdW2g.addEntry(i, dW2g[i]);
    	return fdW2g;
    }
    
    private FeatureVector getdX2g(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	double[][] wpU2g = lfd.wpU2g, wpV2g = lfd.wpV2g;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dX2g = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int d = (((gp < head ? 0 : 1) << 1) | (head < mod ? 0 : 1)) + 1;
    		int d2 = (((gp2 < head2 ? 0 : 1) << 1) | (head2 < mod ? 0 : 1)) + 1;
    		if (gp != -1)
    			dX2g.addEntries(wordFvs[gp], wpU2g[head][k] * wpV2g[mod][k] * (W2g[k][0] + W2g[k][d]));
    		if (gp2 != -1)
    			dX2g.addEntries(wordFvs[gp2], - wpU2g[head2][k] * wpV2g[mod][k] * (W2g[k][0] + W2g[k][d2]));
    	}
    	return dX2g;
    }
    
    private FeatureVector getdU2s(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	DependencyArcList actArcLis = new DependencyArcList(actDeps, false);
    	DependencyArcList predArcLis = new DependencyArcList(predDeps, false);
    	double[][] wpV2s = lfd.wpV2s, wpX2s = lfd.wpX2s;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU2s = new FeatureVector(N);
    	for (int h = 0; h < L; ++h) {
    		int st, ed;
    		
    		st = actArcLis.startIndex(h);
			ed = actArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = actArcLis.get(p);
				int s = actArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
    			dU2s.addEntries(wordFvs[h], wpV2s[m][k] * (W2s[k][0] + W2s[k][d]) * wpX2s[s][k]);
			}
			
			st = predArcLis.startIndex(h);
			ed = predArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = predArcLis.get(p);
				int s = predArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
    			dU2s.addEntries(wordFvs[h], - wpV2s[m][k] * (W2s[k][0] + W2s[k][d]) * wpX2s[s][k]);
			}
    	}
    	return dU2s;
    }
    
    private FeatureVector getdV2s(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	DependencyArcList actArcLis = new DependencyArcList(actDeps, false);
    	DependencyArcList predArcLis = new DependencyArcList(predDeps, false);
    	double[][] wpU2s = lfd.wpU2s, wpX2s = lfd.wpX2s;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV2s = new FeatureVector(N);
    	for (int h = 0; h < L; ++h) {
    		int st, ed;
    		
    		st = actArcLis.startIndex(h);
			ed = actArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = actArcLis.get(p);
				int s = actArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
    			dV2s.addEntries(wordFvs[m], wpU2s[h][k] * (W2s[k][0] + W2s[k][d]) * wpX2s[s][k]);
			}
			
			st = predArcLis.startIndex(h);
			ed = predArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = predArcLis.get(p);
				int s = predArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
    			dV2s.addEntries(wordFvs[m], - wpU2s[h][k] * (W2s[k][0] + W2s[k][d]) * wpX2s[s][k]);
			}
    	}
    	return dV2s;
    }
    
    private FeatureVector getdW2s(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	DependencyArcList actArcLis = new DependencyArcList(actDeps, false);
    	DependencyArcList predArcLis = new DependencyArcList(predDeps, false);
    	double[][] wpU2s = lfd.wpU2s, wpV2s = lfd.wpV2s, wpX2s = lfd.wpX2s;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	double[] dW2s = new double[D2];
    	for (int h = 0; h < L; ++h) {
    		int st, ed;
    		
    		st = actArcLis.startIndex(h);
			ed = actArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = actArcLis.get(p);
				int s = actArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
				dW2s[0] += wpU2s[h][k] * wpV2s[m][k] * wpX2s[s][k];
				dW2s[d] += wpU2s[h][k] * wpV2s[m][k] * wpX2s[s][k];
			}
			
			st = predArcLis.startIndex(h);
			ed = predArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = predArcLis.get(p);
				int s = predArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
				dW2s[0] -= wpU2s[h][k] * wpV2s[m][k] * wpX2s[s][k];
				dW2s[d] -= wpU2s[h][k] * wpV2s[m][k] * wpX2s[s][k];
			}
    	}
    	
    	FeatureVector fdW2s = new FeatureVector(D2);
    	for (int i = 0; i < D2; ++i)
    		fdW2s.addEntry(i, dW2s[i]);
    	return fdW2s;
    }
    
    private FeatureVector getdX2s(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) 
    {
    	DependencyArcList actArcLis = new DependencyArcList(actDeps, false);
    	DependencyArcList predArcLis = new DependencyArcList(predDeps, false);
    	double[][] wpU2s = lfd.wpU2s, wpV2s = lfd.wpV2s;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dX2s = new FeatureVector(N);
    	for (int h = 0; h < L; ++h) {
    		int st, ed;
    		
    		st = actArcLis.startIndex(h);
			ed = actArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = actArcLis.get(p);
				int s = actArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
    			dX2s.addEntries(wordFvs[s], wpU2s[h][k] * wpV2s[m][k] * (W2s[k][0] + W2s[k][d]));
			}
			
			st = predArcLis.startIndex(h);
			ed = predArcLis.endIndex(h);
			for (int p = st; p+1 < ed; ++p) {
				int m = predArcLis.get(p);
				int s = predArcLis.get(p+1);
				int d = (((h < m ? 0 : 1) << 1) | (h < s ? 0 : 1)) + 1;
    			dX2s.addEntries(wordFvs[s], - wpU2s[h][k] * wpV2s[m][k] * (W2s[k][0] + W2s[k][d]));
			}
    	}
    	return dX2s;
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
