package parser;

import java.util.ArrayList;
import java.util.Comparator;

import utils.Utils;

public class SecondLowRankParam {
	public int N, D;
	public ArrayList<MatrixEntry> list;
	
	public SecondLowRankParam(Parameters parameters) 
	{
		N = parameters.N;
		D = parameters.D2;
		list = new ArrayList<MatrixEntry>();
	}
	
	public void add(int x, int y, int z, int r, double v)
	{
		list.add(new MatrixEntry(x, y, z, r, v));
	}
	
	public void decompose(Parameters params)
	{
		int maxRank = params.U2.length;
		
		int MAXITER=1000;
		double eps = 1e-6;
		for (int i = 0; i < maxRank; ++i) {
			double[] u = new double[N];
			double[] v = Utils.getRandomNormVector(N, 1);
			double[] x = Utils.getRandomNormVector(N, 1);
			double[] w = Utils.getRandomNormVector(D, 1);
			
			int iter = 0;
			double norm = 0.0, lastnorm = Double.POSITIVE_INFINITY;
			for (iter = 0; iter < MAXITER; ++iter) {
				
				// u = <T,-,v,w,x>
				for (int j = 0; j < N; ++j)
					u[j] = 0;
				for (MatrixEntry e : list) {
					u[e.x] += e.value * v[e.y] * w[e.z] * x[e.r];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(v, params.V2[j]) 
							   * Utils.dot(w, params.W2[j])
							   * Utils.dot(x, params.X2[j]);
					for (int k = 0; k < N; ++k)
						u[k] -= dot*params.U2[j][k];
				}
				Utils.normalize(u);
				
				// v = <T,u,-,w,x>
				for (int j = 0; j < N; ++j)
					v[j] = 0;
				for (MatrixEntry e : list) {
					v[e.y] += e.value * u[e.x] * w[e.z] * x[e.r];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(u, params.U2[j]) 
							   * Utils.dot(w, params.W2[j])
							   * Utils.dot(x, params.X2[j]);
					for (int k = 0; k < N; ++k)
						v[k] -= dot*params.V2[j][k];
				}
				Utils.normalize(v);
				
				// x = <T,u,v,w,->
				for (int j = 0; j < N; ++j)
					x[j] = 0;
				for (MatrixEntry e : list) {
					x[e.r] += e.value * u[e.x] * v[e.y] * w[e.z];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(u, params.U2[j]) 
							   * Utils.dot(v, params.V2[j])
							   * Utils.dot(w, params.W2[j]);
					for (int k = 0; k < N; ++k)
						x[k] -= dot*params.X2[j][k];
				}
				Utils.normalize(x);
				
				// w = <T,u,v,-,x>
				for (int j = 0; j < D; ++j)
					w[j] = 0;
				for (MatrixEntry e : list) {
					w[e.z] += e.value * u[e.x] * v[e.y] * x[e.r];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(u, params.U2[j]) 
							   * Utils.dot(v, params.V2[j])
							   * Utils.dot(x, params.X2[j]);
					for (int k = 0; k < D; ++k)
						w[k] -= dot*params.W2[j][k];
				}
							
				norm = Math.sqrt(Utils.squaredSum(w));
				if (lastnorm != Double.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
					break;
				lastnorm = norm;
			}
			if (iter >= MAXITER) {
				System.out.printf("\tWARNING: Power method didn't converge." +
						"R=%d sigma=%f%n", i, norm);
			}
			if (Math.abs(norm) <= eps) {
				System.out.printf("\tWARNING: Power method has nearly-zero sigma. R=%d%n",i);
			}
			System.out.printf("\t%.2f", norm);
			params.U2[i] = u;
			params.V2[i] = v;
			params.W2[i] = w;
			params.X2[i] = x;
		}
		System.out.println();
		
		for (int i = 0; i < maxRank; ++i) {
			params.totalU2[i] = params.U2[i].clone();
			params.totalV2[i] = params.V2[i].clone();
			params.totalW2[i] = params.W2[i].clone();
			params.totalX2[i] = params.X2[i].clone();
		}
		
		
		// check
//		double normT = 0, normL = 0, dp = 0;
//		
//		for (MatrixEntry e : list) {
//			normL += e.value * e.value;
//			
//			double s = 0;
//			for (int i = 0; i < maxRank; ++i)
//				s += params.U2[i][e.x] * params.V2[i][e.y] * params.W2[i][e.z] * params.X2[i][e.r];
//			dp += e.value * s;
//		}
//		
//		for (int i = 0; i < maxRank; ++i)
//			for (int j = 0; j < maxRank; ++j) {
//				double su = 0;
//				for (int k = 0; k < N; ++k)
//					su += params.U2[i][k] * params.U2[j][k];
//				double sv = 0;
//				for (int k = 0; k < N; ++k)
//					sv += params.V2[i][k] * params.V2[j][k];
//				double sw = 0;
//				for (int k = 0; k < D; ++k)
//					sw += params.W2[i][k] * params.W2[j][k];
//				double sx = 0;
//				for (int k = 0; k < N; ++k)
//					sx += params.X2[i][k] * params.X2[j][k];
//				normT += su * sv * sw * sx;
//			}
//		double diff = normL - 2*dp + normT;
//		
//		System.out.println(normL + "\t" + diff + "\t" + diff/normL);
	}
}

class MatrixEntry
{
	public int x, y, z, r;
	public double value;
	
	public MatrixEntry(int _x, int _y, int _z, int _r, double _value)
	{
		x = _x;
		y = _y;
		z = _z;
		r = _r;
		value = _value;
	}
}

class MatrixEntryComparator implements Comparator<MatrixEntry>
{

	@Override
	public int compare(MatrixEntry o1, MatrixEntry o2) {
		if (o1.y != o2.y)
			return o1.y - o2.y;
		else
			return o1.x - o2.x;
	}
	
}