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
	
	public void decompose(double[][] U, double[][] V, double[][] W, double[][] X)
	{
		int maxRank = U.length;
		
		int MAXITER=1000;
		double eps = 1e-6;
		for (int i = 0; i < maxRank; ++i) {
			double[] u = new double[N];
			double[] v = Utils.getRandomNormVector(N, 1);
			double[] w = Utils.getRandomNormVector(D, 1);
			double[] r = Utils.getRandomNormVector(N, 1);
			
			int iter = 0;
			double norm = 0.0, lastnorm = Double.POSITIVE_INFINITY;
			for (iter = 0; iter < MAXITER; ++iter) {
				
				// u = <T,-,v,w,r>
				for (int j = 0; j < N; ++j)
					u[j] = 0;
				for (MatrixEntry e : list) {
					u[e.x] += e.value * v[e.y] * w[e.z] * r[e.r];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(v, V[j])
							   * Utils.dot(w, W[j])
							   * Utils.dot(r, X[j]);
					for (int k = 0; k < N; ++k)
						u[k] -= dot*U[j][k];
				}
				Utils.normalize(u);
				
				// v = <T,u,-,w,r>
				for (int j = 0; j < N; ++j)
					v[j] = 0;
				for (MatrixEntry e : list) {
					v[e.y] += e.value * u[e.x] * w[e.z] * r[e.r];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(u, U[j]) 
							   * Utils.dot(w, W[j])
							   * Utils.dot(r, X[j]);
					for (int k = 0; k < N; ++k)
						v[k] -= dot*V[j][k];
				}
				Utils.normalize(v);
				
				// w = <T,u,v,-,r>
				for (int j = 0; j < D; ++j)
					w[j] = 0;
				for (MatrixEntry e : list) {
					w[e.z] += e.value * u[e.x] * v[e.y] * r[e.r];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(u, U[j]) 
							   * Utils.dot(v, V[j])
							   * Utils.dot(r, X[j]);
					for (int k = 0; k < D; ++k)
						w[k] -= dot*W[j][k];
				}
				Utils.normalize(w);
				
				// r = <T,u,v,w,->
				for (int j = 0; j < N; ++j)
					r[j] = 0;
				for (MatrixEntry e : list) {
					r[e.r] += e.value * u[e.x] * v[e.y] * w[e.z];
				}
				for (int j = 0; j < i; ++j) {
					double dot = Utils.dot(u, U[j]) 
							   * Utils.dot(v, V[j])
							   * Utils.dot(w, W[j]);
					for (int k = 0; k < N; ++k)
						r[k] -= dot*X[j][k];
				}			
				norm = Math.sqrt(Utils.squaredSum(r));
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
			U[i] = u;
			V[i] = v;
			W[i] = w;
			X[i] = r;
		}
		System.out.println();
		
		
		// check
		double normT = 0, normL = 0, dp = 0;
		
		for (MatrixEntry e : list) {
			normL += e.value * e.value;
			
			double s = 0;
			for (int i = 0; i < maxRank; ++i)
				s += U[i][e.x] * V[i][e.y] * W[i][e.z] * X[i][e.r];
			dp += e.value * s;
		}
		
		for (int i = 0; i < maxRank; ++i)
			for (int j = 0; j < maxRank; ++j) {
				double su = 0;
				for (int k = 0; k < N; ++k)
					su += U[i][k] * U[j][k];
				double sv = 0;
				for (int k = 0; k < N; ++k)
					sv += V[i][k] * V[j][k];
				double sw = 0;
				for (int k = 0; k < D; ++k)
					sw += W[i][k] * W[j][k];
				double sx = 0;
				for (int k = 0; k < N; ++k)
					sx += X[i][k] * X[j][k];
				normT += su * sv * sw * sx;
			}
		double diff = normL - 2*dp + normT;
		
		System.out.println(normL + "\t" + diff + "\t" + diff/normL);
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