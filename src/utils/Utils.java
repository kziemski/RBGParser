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
	
	public static float logSumExp(float x, float y) 
	{
		if (x == Float.NEGATIVE_INFINITY && x == y)
			return Float.NEGATIVE_INFINITY;
		else if (x < y)
			return y + (float)Math.log1p(Math.exp(x-y));
		else 
			return x + (float)Math.log1p(Math.exp(y-x));
	}
	
	public static float[] getRandomUnitVector(int length) 
	{
		float[] vec = new float[length];
		float sum = 0;
		for (int i = 0; i < length; ++i) {
			vec[i] = rnd.nextFloat() - 0.5f;
			sum += vec[i] * vec[i];
		}
		float invSqrt = (float)(1.0 / Math.sqrt(sum));
		for (int i = 0; i < length; ++i) 
			vec[i] *= invSqrt;
		return vec;
	}
	
	public static float squaredSum(float[] vec) 
	{
		float sum = 0;
		for (int i = 0, N = vec.length; i < N; ++i)
			sum += vec[i] * vec[i];
		return sum;
	}
	
	public static void normalize(float[] vec) 
	{
		float coeff = (float)(1.0 / Math.sqrt(squaredSum(vec)));
		for (int i = 0, N = vec.length; i < N; ++i)
			vec[i] *= coeff;
	}
	
	public static float max(float[] vec) 
	{
		float max = Float.NEGATIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			max = Math.max(max, vec[i]);
		return max;
	}
	
	public static float min(float[] vec) 
	{
		float min = Float.POSITIVE_INFINITY;
		for (int i = 0, N = vec.length; i < N; ++i)
			min = Math.min(min, vec[i]);
		return min;
	}

	public static float dot(float[] u, float[] v)
	{
		float dot = 0.0f;
		for (int i = 0, N = u.length; i < N; ++i)
			dot += u[i]*v[i];
		return dot;
	}
	
	public static int getBinnedDistance(int x) {
    	int y = x > 0 ? x : -x;
    	int dis = 0;
    	if (y > 10)
    		dis = 7;
    	else if (y > 5)
    		dis = 6;
    	else dis = y;
    	return x > 0 ? dis : dis + 7;
    }
}
