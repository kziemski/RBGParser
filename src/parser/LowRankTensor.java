package parser;

import java.util.ArrayList;
import java.util.Comparator;

import utils.Utils;

public class LowRankTensor {
	public int dim, rank;
	public int[] N;
	public ArrayList<MatEntry> list;
	
	public LowRankTensor(int[] _N, int _rank) 
	{
		N = _N.clone();
		dim = N.length;
		rank = _rank;
		list = new ArrayList<MatEntry>();
	}
	
	public void add(int[] x, float val)
	{
		list.add(new MatEntry(x, val));
	}
	
	public void decompose(ArrayList<float[][]> param)
	{	
		int MAXITER=1000;
		float eps = 1e-6f;
		for (int i = 0; i < rank; ++i) {
			ArrayList<float[]> a = new ArrayList<float[]>();
			for (int k = 0; k < dim; ++k) {
				a.add(Utils.getRandomUnitVector(N[k]));
			}
			
			int iter = 0;
			float norm = 0.0f, lastnorm = Float.POSITIVE_INFINITY;
			for (iter = 0; iter < MAXITER; ++iter) {
				for (int k = 0; k < dim; ++k) {
					float[] b = a.get(k);
					for (int j = 0; j < N[k]; ++j)
						b[j] = 0;
					for (MatEntry e : list) {
						float s = e.val;
						for (int l = 0; l < dim; ++l)
							if (l != k)
								s *= a.get(l)[e.x[l]];
						b[e.x[k]] += s;
					}
					for (int j = 0; j < i; ++j) {
						float dot = 1;
						for (int l = 0; l < dim; ++l)
							if (l != k)
								dot *= Utils.dot(a.get(l), param.get(l)[j]);
						for (int p = 0; p < N[k]; ++p)
							b[p] -= dot*param.get(k)[j][p];
					}
					if (k < dim-1)
						Utils.normalize(b);
					else norm = (float)Math.sqrt(Utils.squaredSum(b));
				}
				if (lastnorm != Float.POSITIVE_INFINITY && Math.abs(norm-lastnorm) < eps)
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
			for (int k = 0; k < dim; ++k)
				param.get(k)[i] = a.get(k);
		}
		System.out.println();
	}
}

class MatEntry
{
	public int[] x;
	public float val;
	
	public MatEntry(int[] _x, float _val)
	{
		x = _x.clone();
		val = _val;
	}
}