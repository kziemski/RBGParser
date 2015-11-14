package parser;

import java.io.Serializable;
import java.util.Arrays;

import utils.FeatureVector;
import utils.Utils;

public class Parameters implements Serializable {

	/**
	 * 
	 */
	//private static final long serialVersionUID = 1L;
	
	public static final int d = 7;
	
	public boolean learnLabel, useGP;
	public float C, gamma, gammaL;
	public int size, sizeL;
	public int rank, rank2;
	public int N, T, D, D2, DL;
	
	public float[] params, paramsL;
	public float[][] U, V, W, WL;
	public float[][] U2, V2, W2, X2, Y2, X2L, Y2L;
	public transient float[] total, totalL;
	public transient float[][] totalU, totalV, totalW, totalWL;
	public transient float[][] totalU2, totalV2, totalW2, totalX2, totalY2, totalX2L, totalY2L;
	
	public transient FeatureVector[] dU, dV, dW, dWL;
	public transient FeatureVector[] dU2, dV2, dW2, dX2, dY2, dX2L, dY2L;
	
	public Parameters(DependencyPipe pipe, Options options) 
	{
		N = pipe.synFactory.numWordFeats;
		T = pipe.types.length;
		D = d * 2 + 1;
		D2 = 3;
		DL = T * 3;
        learnLabel = options.learnLabel;
        useGP = options.useGP;
		C = options.C;
		gamma = options.gamma;
		gammaL = options.gammaLabel;
		rank = options.R;
		rank2 = options.R2;
        
		size = pipe.synFactory.numArcFeats+1;	
		params = new float[size];
		total = new float[size];
		if (learnLabel) {
			sizeL = pipe.synFactory.numLabeledArcFeats+1;
			paramsL = new float[sizeL];
			totalL = new float[sizeL];
		}
		
		U = new float[N][rank];		
		V = new float[N][rank];
		W = new float[D][rank];
		totalU = new float[N][rank];
		totalV = new float[N][rank];
		totalW = new float[D][rank];
		dU = new FeatureVector[rank];
		dV = new FeatureVector[rank];
		dW = new FeatureVector[rank];
		if (learnLabel) {
			WL = new float[DL][rank];
			totalWL = new float[DL][rank];
			dWL = new FeatureVector[rank];
		}
		
		if (useGP) {
			U2 = new float[N][rank2];
			V2 = new float[N][rank2];
			W2 = new float[N][rank2];
			X2 = new float[D2][rank2];
			Y2 = new float[D2][rank2];
			totalU2 = new float[N][rank2];
			totalV2 = new float[N][rank2];
			totalW2 = new float[N][rank2];
			totalX2 = new float[D2][rank2];
			totalY2 = new float[D2][rank2];
			dU2 = new FeatureVector[rank2];
			dV2 = new FeatureVector[rank2];
			dW2 = new FeatureVector[rank2];
			dX2 = new FeatureVector[rank2];
			dY2 = new FeatureVector[rank2];
			if (learnLabel) {
				X2L = new float[DL][rank2];
				Y2L = new float[DL][rank2];
				totalX2L = new float[DL][rank2];
				totalY2L = new float[DL][rank2];
				dX2L = new FeatureVector[rank2];
				dY2L = new FeatureVector[rank2];
			}
		}
	}
	
	public void assignTotal()
	{
		for (int i = 0; i < N; ++i) {
			totalU[i] = U[i].clone();
			totalV[i] = V[i].clone();
			if (useGP) {
				totalU2[i] = U2[i].clone();
				totalV2[i] = V2[i].clone();
				totalW2[i] = W2[i].clone();
			}
		}
		
		for (int i = 0; i < D; ++i)
			totalW[i] = W[i].clone();
		
		if (useGP) {
			for (int i = 0; i < D2; ++i) {
				totalX2[i] = X2[i].clone();
				totalY2[i] = Y2[i].clone();
			}
		}
		
		if (learnLabel) {
			for (int i = 0; i < DL; ++i) {
				totalWL[i] = WL[i].clone();
				if (useGP) {
					totalX2L[i] = X2L[i].clone();
					totalY2L[i] = Y2L[i].clone();
				}
			}
		}
	}
	
	private void assignColumn(float[][] mat, int col, float[] values)
	{
		for (int id = 0, tot=values.length; id < tot; ++id)
			mat[id][col] = values[id];
	}
	
	public void randomlyInit() 
	{
		for (int i = 0; i < rank; ++i) {
			assignColumn(U, i, Utils.getRandomUnitVector(N));
			assignColumn(V, i, Utils.getRandomUnitVector(N));
			assignColumn(W, i, Utils.getRandomUnitVector(D));
			if (learnLabel) {
				assignColumn(WL, i, Utils.getRandomUnitVector(DL));
			}
		}
		if (useGP) {
			for (int i = 0; i < rank2; ++i) {
				assignColumn(U2, i, Utils.getRandomUnitVector(N));
				assignColumn(V2, i, Utils.getRandomUnitVector(N));
				assignColumn(W2, i, Utils.getRandomUnitVector(N));
				assignColumn(X2, i, Utils.getRandomUnitVector(D2));
				assignColumn(Y2, i, Utils.getRandomUnitVector(D2));
				if (learnLabel) {
					assignColumn(X2L, i, Utils.getRandomUnitVector(DL));
					assignColumn(Y2L, i, Utils.getRandomUnitVector(DL));
				}
			}
		}
		assignTotal();
	}
	
	private void averageTheta(float[] a, float[] totala, int T, float c)
	{
		int n = a.length;
		for (int i = 0; i < n; ++i) {
			a[i] += c*totala[i]/T;
		}
	}
	
	private void averageTensor(float[][] a, float[][] totala, int T, float c)
	{
		int n = a.length;
		if (n == 0)
			return;
		int m = a[0].length;
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < m; ++j) {
				a[i][j] += c*totala[i][j]/T;
			}
	}
	
	public void averageParameters(int T, float c) 
	{
		averageTheta(params, total, T, c);
		if (learnLabel) {
			averageTheta(paramsL, totalL, T, c);
		}
		
		averageTensor(U, totalU, T, c);
		averageTensor(V, totalV, T, c);
		averageTensor(W, totalW, T, c);
		if (learnLabel) {
			averageTensor(WL, totalWL, T, c);
		}
		
		if (useGP) {
			averageTensor(U2, totalU2, T, c);
			averageTensor(V2, totalV2, T, c);
			averageTensor(W2, totalW2, T, c);
			averageTensor(X2, totalX2, T, c);
			averageTensor(Y2, totalY2, T, c);
			if (learnLabel) {
				averageTensor(X2L, totalX2L, T, c);
				averageTensor(Y2L, totalY2L, T, c);
			}
		}
	}
	
	private void clearMat(float[][] mat)
	{
		for (int i = 0, n = mat.length; i < n; ++i)
			Arrays.fill(mat[i], 0);
	}
	
	public void clearTensor() 
	{
		clearMat(U);
		clearMat(V);
		clearMat(W);
		clearMat(totalU);
		clearMat(totalV);
		clearMat(totalW);
		if (learnLabel) {
			clearMat(WL);
			clearMat(totalWL);
		}
		if (useGP) {
			clearMat(U2);
			clearMat(V2);
			clearMat(W2);
			clearMat(X2);
			clearMat(Y2);
			clearMat(totalU2);
			clearMat(totalV2);
			clearMat(totalW2);
			clearMat(totalX2);
			clearMat(totalY2);
			if (learnLabel) {
				clearMat(X2L);
				clearMat(Y2L);
				clearMat(totalX2L);
				clearMat(totalY2L);
			}
		}
	}
	
	public void clearTheta() 
	{
		Arrays.fill(params, 0);
		Arrays.fill(total, 0);
		if (learnLabel) {
			Arrays.fill(paramsL, 0);
			Arrays.fill(totalL, 0);
		}
	}
	
	private void printStat(float[][] a, String s) 
	{
		int n = a.length;
		float sum = 0;
		float min = Float.POSITIVE_INFINITY, max = Float.NEGATIVE_INFINITY;
		for (int i = 0; i < n; ++i) {
			sum += Utils.squaredSum(a[i]);
			min = Math.min(min, Utils.min(a[i]));
			max = Math.max(max, Utils.max(a[i]));
		}
		System.out.printf(" |%s|^2: %f min: %f\tmax: %f%n", s, sum, min, max);
	}
	
	public void printStat()
	{
		printStat(U, "U");
		printStat(V, "V");
		printStat(W, "W");
		if (learnLabel) {
			printStat(WL, "WL");
		}
		if (useGP) {
			printStat(U2, "U2");
			printStat(V2, "V2");
			printStat(W2, "W2");
			printStat(X2, "X2");
			printStat(Y2, "Y2");
			if (learnLabel) {
				printStat(X2L, "X2L");
				printStat(Y2L, "Y2L");
			}
		}
	}
	
	public void printThetaStat() 
	{
		float sum = Utils.squaredSum(params);
		float min = Utils.min(params);
		float max = Utils.max(params);		
		System.out.printf(" |\u03b8|^2: %f min: %f\tmax: %f%n", sum, min, max);
	}
	
	public void projectMat(float[][] mat, FeatureVector fv, float[] proj)
	{
		int rank = proj.length;
		Arrays.fill(proj, 0);
		for (int id = 0, n = fv.size(); id < n; ++id) {
			int i = fv.x(id);
			float w = fv.value(id);
			for (int j = 0; j < rank; ++j)
				proj[j] += mat[i][j] * w;
		}
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
		int binDist = Utils.getBinnedDistance(dist);
		for (int r = 0; r < rank; ++r)
			sum += proju[r] * projv[r] * (W[binDist][r] + W[0][r]);
		return sum;
	}
	
	public float dotProductL(float[] proju, float[] projv, int lab, int dir)
	{
		float sum = 0;
		for (int r = 0; r < rank; ++r)
			sum += proju[r] * projv[r] * (WL[lab][r] + WL[dir*T+lab][r]);
		return sum;
	}
	
	public float dotProduct2(float[] proju, float[] projv, float[] projw, int pdir, int dir)
	{
		float sum = 0;
		for (int r = 0; r < rank2; ++r)
			sum += proju[r] * projv[r] * projw[r] * (X2[0][r] + X2[pdir][r]) * (Y2[0][r] + Y2[dir][r]);
		return sum;
	}
	
	public float dotProduct2L(float[] proju, float[] projv, float[] projw,
			int plab, int lab, int pdir, int dir)
	{
		float sum = 0;
		for (int r = 0; r < rank2; ++r)
			sum += proju[r] * projv[r] * projw[r] * (X2L[plab][r] + X2L[pdir*T+plab][r])
					* (Y2L[lab][r] + Y2L[dir*T+lab][r]);
		return sum;
	}
	
	private void addTheta(float[] a, float[] totala, FeatureVector da,
			float coeff, float coeff2)
	{
		if (da == null)
			return;
		for (int i = 0, K = da.size(); i < K; ++i) {
    		int x = da.x(i);
    		float z = da.value(i);
    		a[x] += coeff * z;
    		totala[x] += coeff2 * z;
		}
	}
	
	private void addTensor(float[][] a, float[][] totala, FeatureVector[] da,
			float coeff, float coeff2)
	{
		int n = da.length;
		for (int k = 0; k < n; ++k) {
    		FeatureVector dak = da[k];
    		if (dak == null)
    			continue;
    		for (int i = 0, K = dak.size(); i < K; ++i) {
    			int x = dak.x(i);
    			double z = dak.value(i);
    			a[x][k] += coeff * z;
    			totala[x][k] += coeff2 * z;
    		}
    	}
	}
	
	public float updateLabel(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, int updCnt)
	{
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	float Fi = getLabelDis(actDeps, actLabs, predDeps, predLabs);
        	
    	FeatureVector dtl = lfd.getLabeledFeatureDifference(gold, pred);
    	float loss = - dtl.dotProduct(paramsL)*gammaL + Fi;
        float l2norm = dtl.Squaredl2NormUnsafe() * gammaL * gammaL;
    	
        // update U
    	for (int k = 0; k < rank; ++k) {        		
    		FeatureVector dUk = getdUL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dUk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);            	
        	for (int u = 0, n = dUk.size(); u < n; ++u)
        		loss -= U[dUk.x(u)][k] * dUk.value(u) * (1-gammaL);
        	dU[k] = dUk;
    	}
    	// update V
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dVk = getdVL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dVk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
        	dV[k] = dVk;
    	}        	
        // update WL
    	for (int k = 0; k < rank; ++k) {
    		FeatureVector dWLk = getdWL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dWLk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
        	dWL[k] = dWLk;
    	}
    	
    	if (useGP) {
	    	// update U2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dU2k = getdU2L(k, lfd, actDeps, actLabs, predDeps, predLabs);
	        	l2norm += dU2k.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);            	
	        	for (int u = 0, n = dU2k.size(); u < n; ++u)
	        		loss -= U2[dU2k.x(u)][k] * dU2k.value(u) * (1-gammaL);
	        	dU2[k] = dU2k;
	    	}
	    	// update V2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dV2k = getdV2L(k, lfd, actDeps, actLabs, predDeps, predLabs);
	        	l2norm += dV2k.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
	        	dV2[k] = dV2k;
	    	} 
	    	// update W2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dW2k = getdW2L(k, lfd, actDeps, actLabs, predDeps, predLabs);
	        	l2norm += dW2k.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
	        	dW2[k] = dW2k;
	    	}
	    	// update X2L
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dX2Lk = getdX2L(k, lfd, actDeps, actLabs, predDeps, predLabs);
	        	l2norm += dX2Lk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
	        	dX2L[k] = dX2Lk;
	    	}
	    	// update Y2L
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dY2Lk = getdY2L(k, lfd, actDeps, actLabs, predDeps, predLabs);
	        	l2norm += dY2Lk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);
	        	dY2L[k] = dY2Lk;
	    	}
    	}
        
        float alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		float coeff, coeff2;
    		
    		coeff = alpha * gammaL;
    		coeff2 = coeff * (1-updCnt);
    		addTheta(paramsL, totalL, dtl, coeff, coeff2);
    		
    		coeff = alpha * (1-gammaL);
			coeff2 = coeff * (1-updCnt);
			addTensor(U, totalU, dU, coeff, coeff2);
			addTensor(V, totalV, dV, coeff, coeff2);
			addTensor(WL, totalWL, dWL, coeff, coeff2);
			if (useGP) {
				addTensor(U2, totalU2, dU2, coeff, coeff2);
				addTensor(V2, totalV2, dV2, coeff, coeff2);
				addTensor(W2, totalW2, dW2, coeff, coeff2);
				addTensor(X2L, totalX2L, dX2L, coeff, coeff2);
				addTensor(Y2L, totalY2L, dY2L, coeff, coeff2);
			}
    	}
    	return loss;
	}
	
	public float update(DependencyInstance gold, DependencyInstance pred,
			LocalFeatureData lfd, GlobalFeatureData gfd,
			int updCnt, int offset)
	{
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	int[] predDeps = pred.heads;
    	int[] predLabs = pred.deplbids;
    	
    	float Fi = getHammingDis(actDeps, actLabs, predDeps, predLabs);
    	
    	FeatureVector dt = lfd.getFeatureDifference(gold, pred);
    	dt.addEntries(gfd.getFeatureDifference(gold, pred));
    	    	
        float loss = - dt.dotProduct(params)*gamma + Fi;
        float l2norm = dt.Squaredl2NormUnsafe() * gamma * gamma;
    	
    	// update U
    	for (int k = 0; k < rank; ++k) {        		
    		FeatureVector dUk = getdU(k, lfd, actDeps, predDeps);
        	l2norm += dUk.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
        	for (int u = 0, n = dUk.size(); u < n; ++u)
        		loss -= U[dUk.x(u)][k] * dUk.value(u) * (1-gamma);
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
    	
    	if (useGP) {
	    	// update U2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dU2k = getdU2(k, lfd, actDeps, predDeps);
	        	l2norm += dU2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);            	
	        	for (int u = 0, n = dU2k.size(); u < n; ++u)
	        		loss -= U2[dU2k.x(u)][k] * dU2k.value(u) * (1-gamma);
	        	dU2[k] = dU2k;
	    	}
	    	// update V2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dV2k = getdV2(k, lfd, actDeps, predDeps);
	        	l2norm += dV2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dV2[k] = dV2k;
	    	} 
	    	// update W2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dW2k = getdW2(k, lfd, actDeps, predDeps);
	        	l2norm += dW2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dW2[k] = dW2k;
	    	}
	    	// update X2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dX2k = getdX2(k, lfd, actDeps, predDeps);
	        	l2norm += dX2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dX2[k] = dX2k;
	    	}
	    	// update Y2
	    	for (int k = 0; k < rank2; ++k) {
	    		FeatureVector dY2k = getdY2(k, lfd, actDeps, predDeps);
	        	l2norm += dY2k.Squaredl2NormUnsafe() * (1-gamma) * (1-gamma);
	        	dY2[k] = dY2k;
	    	}
    	}
        
        float alpha = loss/l2norm;
    	alpha = Math.min(C, alpha);
    	if (alpha > 0) {
    		float coeff, coeff2;
    		
    		coeff = alpha * gamma;
    		coeff2 = coeff * (1-updCnt);
    		addTheta(params, total, dt, coeff, coeff2);

			coeff = alpha * (1-gamma);
			coeff2 = coeff * (1-updCnt);
			addTensor(U, totalU, dU, coeff, coeff2);
			addTensor(V, totalV, dV, coeff, coeff2);
			addTensor(W, totalW, dW, coeff, coeff2);
			if (useGP) {
				addTensor(U2, totalU2, dU2, coeff, coeff2);
				addTensor(V2, totalV2, dV2, coeff, coeff2);
				addTensor(W2, totalW2, dW2, coeff, coeff2);
				addTensor(X2, totalX2, dX2, coeff, coeff2);
				addTensor(Y2, totalY2, dY2, coeff, coeff2);
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
    		float coeff = alpha;
    		float coeff2 = coeff * (1-updCnt);
    		addTheta(params, total, fv, coeff, coeff2);
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
    		int d = Utils.getBinnedDistance(head-mod);
    		int d2 = Utils.getBinnedDistance(head2-mod);
    		float dotv = wpV[mod][k]; //wordFvs[mod].dotProduct(V[k]);    		
    		dU.addEntries(wordFvs[head], dotv * (W[0][k] + W[d][k]));
    		dU.addEntries(wordFvs[head2], - dotv * (W[0][k] + W[d2][k]));
    	}
    	return dU;
    }
    
    private FeatureVector getdUL(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		assert(actDeps[mod] == predDeps[mod]);
    		int head  = actDeps[mod];
    		int dir = head > mod ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		if (lab == lab2) continue;
    		float dotv = wpV[mod][k]; //wordFvs[mod].dotProduct(V[k]);    		
    		dU.addEntries(wordFvs[head], dotv * (WL[lab][k] + WL[dir*T+lab][k])
    									 - dotv * (WL[lab2][k] + WL[dir*T+lab2][k]));
    	}
    	return dU;
    }
    
    private FeatureVector getdV(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU = lfd.wpU;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		if (head == head2) continue;
    		int d = Utils.getBinnedDistance(head-mod);
    		int d2 = Utils.getBinnedDistance(head2-mod);
    		float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		float dotu2 = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		dV.addEntries(wordFvs[mod], dotu  * (W[0][k] + W[d][k])
    									- dotu2 * (W[0][k] + W[d2][k]));    		
    	}
    	return dV;
    }
    
    private FeatureVector getdVL(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpU = lfd.wpU;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		assert(actDeps[mod] == predDeps[mod]);
    		int head  = actDeps[mod];
    		int dir = head > mod ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		if (lab == lab2) continue;
    		float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		dV.addEntries(wordFvs[mod], dotu  * (WL[lab][k] + WL[dir*T+lab][k])
    									- dotu * (WL[lab2][k] + WL[dir*T+lab2][k]));    		
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
    		int d = Utils.getBinnedDistance(head-mod);
    		int d2 = Utils.getBinnedDistance(head2-mod);
    		float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		float dotu2 = wpU[head2][k]; //wordFvs[head2].dotProduct(U[k]);
    		float dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
    		dW[0] += (dotu - dotu2) * dotv;
    		dW[d] += dotu * dotv;
    		dW[d2] -= dotu2 * dotv;
    	}
    	
    	FeatureVector dWfv = new FeatureVector(D);
    	for (int i = 0; i < D; ++i)
    		dWfv.addEntry(i, dW[i]);
    	return dWfv;
    }
    
    private FeatureVector getdWL(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpU = lfd.wpU, wpV = lfd.wpV;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	float[] dWL = new float[DL];
    	for (int mod = 1; mod < L; ++mod) {
    		assert(actDeps[mod] == predDeps[mod]);
    		int head = actDeps[mod];
    		int dir = head > mod ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		if (lab == lab2) continue;
    		float dotu = wpU[head][k];   //wordFvs[head].dotProduct(U[k]);
    		float dotv = wpV[mod][k];  //wordFvs[mod].dotProduct(V[k]);
    		dWL[lab] += dotu * dotv;
    		dWL[dir*T+lab] += dotu * dotv;
    		dWL[lab2] -= dotu * dotv;
    		dWL[dir*T+lab2] -= dotu * dotv;
    	}
    	
    	FeatureVector dWLfv = new FeatureVector(DL);
    	for (int i = 0; i < DL; ++i)
    		dWLfv.addEntry(i, dWL[i]);
    	return dWLfv;
    }
    
    private FeatureVector getdU2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpV2 = lfd.wpV2;
    	float[][] wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int dir = head > mod ? 1 : 2;
    		int dir2 = head2 > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int pdir2 = gp2 > head2 ? 1 : 2;
    		if (gp != -1)
    			dU2.addEntries(wordFvs[gp], wpV2[head][k] * wpW2[mod][k] * (X2[0][k] + X2[pdir][k]) * (Y2[0][k] + Y2[dir][k]));
    		if (gp2 != -1)
    			dU2.addEntries(wordFvs[gp2], - wpV2[head2][k] * wpW2[mod][k] * (X2[0][k] + X2[pdir2][k]) * (Y2[0][k] + Y2[dir2][k]));
    	}
    	return dU2;
    }
    
    private FeatureVector getdU2L(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpV2 = lfd.wpV2;
    	float[][] wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dU2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int gp = actDeps[head];
    		if (gp == -1)
    			continue;
    		int dir = head > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		int plab = actLabs[head];
    		int plab2 = predLabs[head];
    		if (lab == lab2 && plab == plab2) continue;
    		float dotv2 = wpV2[head][k];
    		float dotw2 = wpW2[mod][k];
    		dU2.addEntries(wordFvs[gp], dotv2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
    								  - dotv2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
    	}
    	return dU2;
    }
    
    private FeatureVector getdV2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU2 = lfd.wpU2;
    	float[][] wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int dir = head > mod ? 1 : 2;
    		int dir2 = head2 > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int pdir2 = gp2 > head2 ? 1 : 2;
    		if (gp != -1)
    			dV2.addEntries(wordFvs[head], wpU2[gp][k] * wpW2[mod][k] * (X2[0][k] + X2[pdir][k]) * (Y2[0][k] + Y2[dir][k]));
    		if (gp2 != -1)
    			dV2.addEntries(wordFvs[head2], - wpU2[gp2][k] * wpW2[mod][k] * (X2[0][k] + X2[pdir2][k]) * (Y2[0][k] + Y2[dir2][k]));
    	}
    	return dV2;
    }
    
    private FeatureVector getdV2L(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpU2 = lfd.wpU2;
    	float[][] wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dV2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int gp = actDeps[head];
    		if (gp == -1)
    			continue;
    		int dir = head > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		int plab = actLabs[head];
    		int plab2 = predLabs[head];
    		if (lab == lab2 && plab == plab2) continue;
    		float dotu2 = wpU2[gp][k];
    		float dotw2 = wpW2[mod][k];
    		dV2.addEntries(wordFvs[head], dotu2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
    								    - dotu2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
    	}
    	return dV2;
    }
    
    private FeatureVector getdW2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU2 = lfd.wpU2;
    	float[][] wpV2 = lfd.wpV2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dW2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int dir = head > mod ? 1 : 2;
    		int dir2 = head2 > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int pdir2 = gp2 > head2 ? 1 : 2;
    		if (gp != -1)
    			dW2.addEntries(wordFvs[mod], wpU2[gp][k] * wpV2[head][k] * (X2[0][k] + X2[pdir][k]) * (Y2[0][k] + Y2[dir][k]));
    		if (gp2 != -1)
    			dW2.addEntries(wordFvs[mod], wpU2[gp2][k] * wpV2[head2][k] * (X2[0][k] + X2[pdir2][k]) * (Y2[0][k] + Y2[dir2][k]));
    	}
    	return dW2;
    }
    
    private FeatureVector getdW2L(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpU2 = lfd.wpU2;
    	float[][] wpV2 = lfd.wpV2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	FeatureVector dW2 = new FeatureVector(N);
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int gp = actDeps[head];
    		if (gp == -1)
    			continue;
    		int dir = head > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		int plab = actLabs[head];
    		int plab2 = predLabs[head];
    		if (lab == lab2 && plab == plab2) continue;
    		float dotu2 = wpU2[gp][k];
    		float dotv2 = wpV2[head][k];
    		dW2.addEntries(wordFvs[mod], dotu2 * dotv2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
    								  - dotu2 * dotv2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
    	}
    	return dW2;
    }
    
    private FeatureVector getdX2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU2 = lfd.wpU2, wpV2 = lfd.wpV2, wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	float[] dX2 = new float[D2];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int dir = head > mod ? 1 : 2;
    		int dir2 = head2 > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int pdir2 = gp2 > head2 ? 1 : 2;
    		if (gp != -1) {
    			float val = wpU2[gp][k] * wpV2[head][k] * wpW2[mod][k] * (Y2[0][k] + Y2[dir][k]);
    			dX2[0] += val;
        		dX2[pdir] += val;
    		}
    		if (gp2 != -1) {
	    		float val2 = wpU2[gp2][k] * wpV2[head2][k] * wpW2[mod][k] * (Y2[0][k] + Y2[dir2][k]);
	    		dX2[0] -= val2;
	    		dX2[pdir2] -= val2;
    		}
    	}
    	
    	FeatureVector dX2fv = new FeatureVector(D2);
    	for (int i = 0; i < D2; ++i)
    		dX2fv.addEntry(i, dX2[i]);
    	return dX2fv;
    }
    
    private FeatureVector getdX2L(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpU2 = lfd.wpU2, wpV2 = lfd.wpV2, wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	float[] dX2L = new float[DL];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int gp = actDeps[head];
    		if (gp == -1)
    			continue;
    		int dir = head > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		int plab = actLabs[head];
    		int plab2 = predLabs[head];
    		if (lab == lab2 && plab == plab2) continue;
    		float dotu2 = wpU2[gp][k];
    		float dotv2 = wpV2[head][k];
    		float dotw2 = wpW2[mod][k];
    		float val = dotu2 * dotv2 * dotw2 * (Y2L[lab][k] + Y2L[dir*T+lab][k]);
    		float val2 = dotu2 * dotv2 * dotw2 * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]);
    		dX2L[plab] += val;
    		dX2L[pdir*T+plab] += val;
    		dX2L[plab2] -= val2;
    		dX2L[pdir*T+plab2] -= val2;
    	}
    	
    	FeatureVector dX2Lfv = new FeatureVector(DL);
    	for (int i = 0; i < DL; ++i)
    		dX2Lfv.addEntry(i, dX2L[i]);
    	return dX2Lfv;
    }
    
    private FeatureVector getdY2(int k, LocalFeatureData lfd, int[] actDeps, int[] predDeps) {
    	float[][] wpU2 = lfd.wpU2, wpV2 = lfd.wpV2, wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	float[] dY2 = new float[D2];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int head2 = predDeps[mod];
    		int gp = actDeps[head];
    		int gp2 = predDeps[head2];
    		if (head == head2 && gp == gp2) continue;
    		int dir = head > mod ? 1 : 2;
    		int dir2 = head2 > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int pdir2 = gp2 > head2 ? 1 : 2;
    		if (gp != -1) {
    			float val = wpU2[gp][k] * wpV2[head][k] * wpW2[mod][k] * (X2[0][k] + X2[pdir][k]);
    			dY2[0] += val;
        		dY2[dir] += val;
    		}
    		if (gp2 != -1) {
    			float val2 = wpU2[gp2][k] * wpV2[head2][k] * wpW2[mod][k] * (X2[0][k] + X2[pdir2][k]);
    			dY2[0] -= val2;
        		dY2[dir2] -= val2;
    		}
    	}
    	
    	FeatureVector dY2fv = new FeatureVector(D2);
    	for (int i = 0; i < D2; ++i)
    		dY2fv.addEntry(i, dY2[i]);
    	return dY2fv;
    }
    
    private FeatureVector getdY2L(int k, LocalFeatureData lfd, int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) {
    	float[][] wpU2 = lfd.wpU2, wpV2 = lfd.wpV2, wpW2 = lfd.wpW2;
    	FeatureVector[] wordFvs = lfd.wordFvs;
    	int L = wordFvs.length;
    	float[] dY2L = new float[DL];
    	for (int mod = 1; mod < L; ++mod) {
    		int head  = actDeps[mod];
    		int gp = actDeps[head];
    		if (gp == -1)
    			continue;
    		int dir = head > mod ? 1 : 2;
    		int pdir = gp > head ? 1 : 2;
    		int lab  = actLabs[mod];
    		int lab2 = predLabs[mod];
    		int plab = actLabs[head];
    		int plab2 = predLabs[head];
    		if (lab == lab2 && plab == plab2) continue;
    		float dotu2 = wpU2[gp][k];
    		float dotv2 = wpV2[head][k];
    		float dotw2 = wpW2[mod][k];
    		float val = dotu2 * dotv2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]);
    		float val2 = dotu2 * dotv2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]);
    		dY2L[lab] += val;
    		dY2L[dir*T+lab] += val;
    		dY2L[lab2] -= val2;
    		dY2L[dir*T+lab2] -= val2;
    	}
    	
    	FeatureVector dY2Lfv = new FeatureVector(DL);
    	for (int i = 0; i < DL; ++i)
    		dY2Lfv.addEntry(i, dY2L[i]);
    	return dY2Lfv;
    }
    
	public float getHammingDis(int[] actDeps, int[] actLabs,
			int[] predDeps, int[] predLabs) 
	{
		float dis = 0;
		for (int i = 1; i < actDeps.length; ++i)
			if (actDeps[i] != predDeps[i]) dis += 1;
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
}
