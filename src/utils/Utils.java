package utils;

import java.util.Random;

public final class Utils {
	
	public static Random rnd = new Random(System.currentTimeMillis());
	
	public static void Assert(boolean assertion) 
	{
		if (!assertion) {
			(new Exception()).printStackTrace();
			System.exit(1);
		}
	}
		
	public static int log2(long x) 
	{
		long y = 1;
		int i = 0;
		while (y < x) {
			y = y << 1;
			++i;
		}
		return i;
	}
	
	public static double logSumExp(double x, double y) 
	{
		if (x == Double.NEGATIVE_INFINITY && x == y)
			return Double.NEGATIVE_INFINITY;
		else if (x < y)
			return y + Math.log1p(Math.exp(x-y));
		else 
			return x + Math.log1p(Math.exp(y-x));
	}
	
	public static double[] getRandomNormVector(int length, double norm) 
	{
		double[] vec = new double[length];
		for (int i = 0; i < length; ++i)
			vec[i] = rnd.nextDouble() - 0.5;
		normalize(vec, norm);
		return vec;
	}
	
	public static double[] getRandomRangeVector(int length, double s) 
	{
		double[] vec = new double[length];
		for (int i = 0; i < length; ++i) {
			vec[i] = rnd.nextDouble() - 0.5;
			vec[i] *= s/0.5;
		}
		return vec;
	}
	
	public static double squaredSum(float[] vec) 
	{
		double sum = 0;
		for (int i = 0, N = vec.length; i < N; ++i)
			sum += vec[i] * vec[i];
		return sum;
	}
	
	public static double squaredSum(double[] vec) 
	{
		double sum = 0;
		for (int i = 0, N = vec.length; i < N; ++i)
			sum += vec[i] * vec[i];
		return sum;
	}
	
	public static void normalize(double[] vec) 
	{
		normalize(vec, 1);
	}
	
	public static void normalize(double[] vec, double norm) 
	{
		double coeff = Math.sqrt(norm / squaredSum(vec));
		for (int i = 0, N = vec.length; i < N; ++i)
			vec[i] *= coeff;
	}
	
	public static double max(float[] vec) 
	{
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			max = Math.max(max, vec[i]);
		return max;
	}
	
	public static double min(float[] vec) 
	{
		double min = Double.POSITIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			min = Math.min(min, vec[i]);
		return min;
	}
	
	public static double max(double[] vec) 
	{
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			max = Math.max(max, vec[i]);
		return max;
	}
	
	public static double min(double[] vec) 
	{
		double min = Double.POSITIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			min = Math.min(min, vec[i]);
		return min;
	}

	public static double dot(double[] u, double[] v)
	{
		double dot = 0.0;
		for (int i = 0, N = u.length; i < N; ++i)
			dot += u[i]*v[i];
		return dot;
	}
	
}
