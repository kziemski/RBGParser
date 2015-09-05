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
	
	public boolean useGP;
	public float C, gammaL;
	public int sizeL;
	public int rank, rank2;
	public int N, T, DL;
	
	public float[] paramsL;
	public transient float[] totalL;
	public float[][] U, V, WL;
	public float[][] U2, V2, W2, X2L, Y2L;
	public transient float[][] totalU, totalV, totalWL;
	public transient float[][] totalU2, totalV2, totalW2, totalX2L, totalY2L;
	
	public transient FeatureVector[] dU, dV, dWL;
	public transient FeatureVector[] dU2, dV2, dW2, dX2L, dY2L;
	
	public Parameters(DependencyPipe pipe, Options options) 
	{
		N = pipe.synFactory.numWordFeats;
		T = pipe.types.length;
		DL = T * 3;
        useGP = options.useGP;
		C = (float) options.C;
		gammaL = (float) options.gammaLabel;
		rank = options.R;
		rank2 = options.R2;
        
		sizeL = pipe.synFactory.numLabeledArcFeats+1;
		paramsL = new float[sizeL];
		totalL = new float[sizeL];
		
//		U = new float[rank][N];		
//		V = new float[rank][N];
//		WL = new float[rank][DL];
//		totalU = new float[rank][N];
//		totalV = new float[rank][N];
//		totalWL = new float[rank][DL];
		
		U = new float[N][rank];		
		V = new float[N][rank];
		WL = new float[DL][rank];
		totalU = new float[N][rank];
		totalV = new float[N][rank];
		totalWL = new float[DL][rank];
		dU = new FeatureVector[rank];
		dV = new FeatureVector[rank];
		dWL = new FeatureVector[rank];
		
		if (useGP) {
//			U2 = new float[rank2][N];
//			V2 = new float[rank2][N];
//			W2 = new float[rank2][N];
//			X2L = new float[rank2][DL];
//			Y2L = new float[rank2][DL];
//			totalU2 = new float[rank2][N];
//			totalV2 = new float[rank2][N];
//			totalW2 = new float[rank2][N];
//			totalX2L = new float[rank2][DL];
//			totalY2L = new float[rank2][DL];
			U2 = new float[N][rank2];
			V2 = new float[N][rank2];
			W2 = new float[N][rank2];
			X2L = new float[DL][rank2];
			Y2L = new float[DL][rank2];
			totalU2 = new float[N][rank2];
			totalV2 = new float[N][rank2];
			totalW2 = new float[N][rank2];
			totalX2L = new float[DL][rank2];
			totalY2L = new float[DL][rank2];
			dU2 = new FeatureVector[rank2];
			dV2 = new FeatureVector[rank2];
			dW2 = new FeatureVector[rank2];
			dX2L = new FeatureVector[rank2];
			dY2L = new FeatureVector[rank2];
		}
	}
	
	private void copyValue(float[][] target, float[][] source)
	{
		int n = source.length;
		for (int i = 0; i < n; ++i)
			target[i] = source[i].clone();
	}
	
	public void assignTotal()
	{
		copyValue(totalU, U);
		copyValue(totalV, V);
		copyValue(totalWL, WL);
		if (useGP) {
			copyValue(totalU2, U2);
			copyValue(totalV2, V2);
			copyValue(totalW2, W2);
			copyValue(totalX2L, X2L);
			copyValue(totalY2L, Y2L);
		}
		
//		for (int i = 0; i < rank; ++i) {
//			totalU[i] = U[i].clone();
//			totalV[i] = V[i].clone();
//			totalWL[i] = WL[i].clone();
//		}
//		if (useGP) {
//			for (int i = 0; i < rank2; ++i) {
//				totalU2[i] = U2[i].clone();
//				totalV2[i] = V2[i].clone();
//				totalW2[i] = W2[i].clone();
//				totalX2L[i] = X2L[i].clone();
//				totalY2L[i] = Y2L[i].clone();
//			}
//		}
	}
	
	private void assignColumn(float[][] mat, int col, float[] values)
	{
		int rsize = values.length;
		for (int id = 0, tot=values.length; id < tot; ++id)
			mat[id][col] = values[id];
	}
	
	public void randomlyInit() 
	{
		
		for (int i = 0; i < rank; ++i) {
			//U[i] = Utils.getRandomUnitVector(N);
			//V[i] = Utils.getRandomUnitVector(N);
			//WL[i] = Utils.getRandomUnitVector(DL);
			assignColumn(U, i, Utils.getRandomUnitVector(N));
			assignColumn(V, i, Utils.getRandomUnitVector(N));
			assignColumn(WL, i, Utils.getRandomUnitVector(DL));
		}
		if (useGP) {
			for (int i = 0; i < rank2; ++i) {
//				U2[i] = Utils.getRandomUnitVector(N);
//				V2[i] = Utils.getRandomUnitVector(N);
//				W2[i] = Utils.getRandomUnitVector(N);
//				X2L[i] = Utils.getRandomUnitVector(DL);
//				Y2L[i] = Utils.getRandomUnitVector(DL);
				assignColumn(U2, i, Utils.getRandomUnitVector(N));
				assignColumn(V2, i, Utils.getRandomUnitVector(N));
				assignColumn(W2, i, Utils.getRandomUnitVector(N));
				assignColumn(X2L, i, Utils.getRandomUnitVector(DL));
				assignColumn(Y2L, i, Utils.getRandomUnitVector(DL));
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
		averageTheta(paramsL, totalL, T, c);
		
		averageTensor(U, totalU, T, c);
		averageTensor(V, totalV, T, c);
		averageTensor(WL, totalWL, T, c);
		
		if (useGP) {
			averageTensor(U2, totalU2, T, c);
			averageTensor(V2, totalV2, T, c);
			averageTensor(W2, totalW2, T, c);
			averageTensor(X2L, totalX2L, T, c);
			averageTensor(Y2L, totalY2L, T, c);
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
		clearMat(WL);
		clearMat(totalU);
		clearMat(totalV);
		clearMat(totalWL);
		if (useGP) {
			clearMat(U2);
			clearMat(V2);
			clearMat(W2);
			clearMat(X2L);
			clearMat(Y2L);
			clearMat(totalU2);
			clearMat(totalV2);
			clearMat(totalW2);
			clearMat(totalX2L);
			clearMat(totalY2L);
		}
//		for (int i = 0; i < rank; ++i) {
//			Arrays.fill(U[i], 0);
//			Arrays.fill(V[i], 0);
//			Arrays.fill(WL[i], 0);
//			Arrays.fill(totalU[i], 0);
//			Arrays.fill(totalV[i], 0);
//			Arrays.fill(totalWL[i], 0);
//		}
//		if (useGP) {
//			for (int i = 0; i < rank2; ++i) {
//				Arrays.fill(U2[i], 0);
//				Arrays.fill(totalU2[i], 0);
//				Arrays.fill(V2[i], 0);
//				Arrays.fill(totalV2[i], 0);
//				Arrays.fill(W2[i], 0);
//				Arrays.fill(totalW2[i], 0);
//				Arrays.fill(X2L[i], 0);
//				Arrays.fill(totalX2L[i], 0);
//				Arrays.fill(Y2L[i], 0);
//				Arrays.fill(totalY2L[i], 0);
//			}
//		}
	}
	
	public void clearTheta() 
	{
		Arrays.fill(paramsL, 0);
		Arrays.fill(totalL, 0);
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
		printStat(WL, "WL");

		if (useGP) {
			printStat(U2, "U2");
			printStat(V2, "V2");
			printStat(W2, "W2");
			printStat(X2L, "X2L");
			printStat(Y2L, "Y2L");
		}
	}
	
	private void projectMat(float[][] mat, FeatureVector fv, float[] proj)
	{
		int rank = proj.length;
		for (int id = 0, n = fv.size(); id < n; ++id) {
			int i = fv.x(id);
			for (int j = 0; j < rank; ++j)
				proj[j] += mat[i][j];
		}
	}
	
	public void projectU(FeatureVector fv, float[] proj) 
	{
		projectMat(U, fv, proj);
		//for (int r = 0; r < rank; ++r) 
		//	proj[r] = fv.dotProduct(U[r]);
	}
	
	public void projectV(FeatureVector fv, float[] proj) 
	{
		projectMat(V, fv, proj);
		//for (int r = 0; r < rank; ++r) 
		//	proj[r] = fv.dotProduct(V[r]);
	}
	
	public void projectU2(FeatureVector fv, float[] proj)
	{
		projectMat(U2, fv, proj);
		//for (int r = 0; r < rank2; ++r) 
		//	proj[r] = fv.dotProduct(U2[r]);
	}
	
	public void projectV2(FeatureVector fv, float[] proj)
	{
		projectMat(V2, fv, proj);
		//for (int r = 0; r < rank2; ++r) 
		//	proj[r] = fv.dotProduct(V2[r]);
	}
	
	public void projectW2(FeatureVector fv, float[] proj)
	{
		projectMat(W2, fv, proj);
		//for (int r = 0; r < rank2; ++r) 
		//	proj[r] = fv.dotProduct(W2[r]);
	}
	
	public float dotProductL(FeatureVector fv)
	{
		return fv.dotProduct(paramsL);
	}
	
	public float dotProductL(float[] proju, float[] projv, int lab, int dir)
	{
		float sum = 0;
		for (int r = 0; r < rank; ++r)
			//sum += proju[r] * projv[r] * (WL[r][lab] + WL[r][dir*T+lab]);
			sum += proju[r] * projv[r] * (WL[lab][r] + WL[dir*T+lab][r]);
		return sum;
	}
	
	public float dotProduct2L(float[] proju, float[] projv, float[] projw,
			int plab, int lab, int pdir, int dir)
	{
		float sum = 0;
		for (int r = 0; r < rank2; ++r)
			//sum += proju[r] * projv[r] * projw[r] * (X2L[r][plab] + X2L[r][pdir*T+plab])
			//		* (Y2L[r][lab] + Y2L[r][dir*T+lab]);
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
		//int n = a.length;
		int n = da.length;
		for (int k = 0; k < n; ++k) {
    		FeatureVector dak = da[k];
    		if (dak == null)
    			continue;
    		for (int i = 0, K = dak.size(); i < K; ++i) {
    			int x = dak.x(i);
    			float z = dak.value(i);
    			//a[k][x] += coeff * z;
    			//totala[k][x] += coeff2 * z;
    			a[x][k] += coeff * z;
    			totala[x][k] += coeff2 * z;
    		}
    	}
	}
	
	public float updateLabel(DependencyInstance gold, int[] predDeps, int[] predLabs,
			LocalFeatureData lfd, int updCnt)
	{
    	int[] actDeps = gold.heads;
    	int[] actLabs = gold.deplbids;
    	
    	float Fi = getLabelDis(actDeps, actLabs, predDeps, predLabs);
        	
    	FeatureVector dtl = lfd.getLabeledFeatureDifference(gold, predDeps, predLabs);
    	float loss = - dtl.dotProduct(paramsL)*gammaL + Fi;
        float l2norm = dtl.Squaredl2NormUnsafe() * gammaL * gammaL;
    	
        // update U
    	for (int k = 0; k < rank; ++k) {        		
    		FeatureVector dUk = getdUL(k, lfd, actDeps, actLabs, predDeps, predLabs);
        	l2norm += dUk.Squaredl2NormUnsafe() * (1-gammaL) * (1-gammaL);            	
        	//loss -= dUk.dotProduct(U[k]) * (1-gammaL);
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
	        	//loss -= dU2k.dotProduct(U2[k]) * (1-gammaL);
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
    		//dU.addEntries(wordFvs[head], dotv * (WL[k][lab] + WL[k][dir*T+lab])
    		//							 - dotv * (WL[k][lab2] + WL[k][dir*T+lab2]));
    		dU.addEntries(wordFvs[head], dotv * (WL[lab][k] + WL[dir*T+lab][k])
    									 - dotv * (WL[lab2][k] + WL[dir*T+lab2][k]));
    	}
    	return dU;
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
    		//dV.addEntries(wordFvs[mod], dotu  * (WL[k][lab] + WL[k][dir*T+lab])
    		//							- dotu * (WL[k][lab2] + WL[k][dir*T+lab2]));    		
    		dV.addEntries(wordFvs[mod], dotu  * (WL[lab][k] + WL[dir*T+lab][k])
					- dotu * (WL[lab2][k] + WL[dir*T+lab2][k]));    
    	}
    	return dV;
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
    		//dU2.addEntries(wordFvs[gp], dotv2 * dotw2 * (X2L[k][plab] + X2L[k][pdir*T+plab]) * (Y2L[k][lab] + Y2L[k][dir*T+lab])
    		//						  - dotv2 * dotw2 * (X2L[k][plab2] + X2L[k][pdir*T+plab2]) * (Y2L[k][lab2] + Y2L[k][dir*T+lab2]));
    		dU2.addEntries(wordFvs[gp], dotv2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
					  - dotv2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
    	}
    	return dU2;
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
    		//dV2.addEntries(wordFvs[head], dotu2 * dotw2 * (X2L[k][plab] + X2L[k][pdir*T+plab]) * (Y2L[k][lab] + Y2L[k][dir*T+lab])
    		//						    - dotu2 * dotw2 * (X2L[k][plab2] + X2L[k][pdir*T+plab2]) * (Y2L[k][lab2] + Y2L[k][dir*T+lab2]));
    		dV2.addEntries(wordFvs[head], dotu2 * dotw2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
				    - dotu2 * dotw2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
    	}
    	return dV2;
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
    		//dW2.addEntries(wordFvs[mod], dotu2 * dotv2 * (X2L[k][plab] + X2L[k][pdir*T+plab]) * (Y2L[k][lab] + Y2L[k][dir*T+lab])
    		//						  - dotu2 * dotv2 * (X2L[k][plab2] + X2L[k][pdir*T+plab2]) * (Y2L[k][lab2] + Y2L[k][dir*T+lab2]));
    		dW2.addEntries(wordFvs[mod], dotu2 * dotv2 * (X2L[plab][k] + X2L[pdir*T+plab][k]) * (Y2L[lab][k] + Y2L[dir*T+lab][k])
					  - dotu2 * dotv2 * (X2L[plab2][k] + X2L[pdir*T+plab2][k]) * (Y2L[lab2][k] + Y2L[dir*T+lab2][k]));
    	}
    	return dW2;
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
    		//float val = dotu2 * dotv2 * dotw2 * (Y2L[k][lab] + Y2L[k][dir*T+lab]);
    		//float val2 = dotu2 * dotv2 * dotw2 * (Y2L[k][lab2] + Y2L[k][dir*T+lab2]);
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
    		//float val = dotu2 * dotv2 * dotw2 * (X2L[k][plab] + X2L[k][pdir*T+plab]);
    		//float val2 = dotu2 * dotv2 * dotw2 * (X2L[k][plab2] + X2L[k][pdir*T+plab2]);
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
