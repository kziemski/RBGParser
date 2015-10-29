package utils;

public class SVD {

    public static native float powerMethod(int[] x, int[] y, float[] z,
                float[] u, float[] v);
    
    public static native int lowRankSvd(float[] At, float[] Bt,
    	int n, int m, int r, float[] S, float[] Ut, float[] Vt);
    
    public static native int svd(float[] A, int n, int m, 
    		float[] S, float[] Ut, float[] Vt);
    
    public static native int svd(int n, int m, int r, int[] x, int [] y, float[] z,
    		float[] S, float[] Ut, float[] Vt);
    
    static {
        System.loadLibrary("SVDImp");
    }
    
//	public static float epsilon = 1e-8;
//	public static int maxNumIters = 1000;
//	public static Random rnd = new Random(System.currentTimeMillis());
//	
//	public static float runIterations(SparseMatrix M, float[] u, float[] v) {
//        
//        System.out.println("N=" + v.length);
//        System.out.println("M=" + M.size);
//
//		int N = v.length, iter;
//		float sigma = 0, prevSigma = -1;
//		
//		for (int i = 0; i < N; ++i) v[i] = rnd.nextDouble() - 0.5;
//		norm(v);
//				
//		for (iter = 1; iter <= maxNumIters; ++iter) {
//			// u = Mv
//			for (int i = 0; i < N; ++i) u[i] = 0;
//			for (MatrixEntry e = M.element; e != null; e = e.next) 
//				u[e.x] += e.value * v[e.y];
//			sigma = norm(u);
//			
//			if (prevSigma != -1 && Math.abs(sigma - prevSigma) < epsilon) {
//				break;
//			} else if (iter % 10000 == 1) System.out.println("\tIter " + iter + " " + sigma);
//			prevSigma = sigma;
//			
//			// v = M^Tu
//			for (int i = 0; i < N; ++i) v[i] = 0;
//			for (MatrixEntry e = M.element; e != null; e = e.next) 
//				v[e.y] += e.value * u[e.x];
//			norm(v);
//		}
//		System.out.println("\tIter " + iter + " " + sigma);
//		assert(iter <= maxNumIters);
//		return sigma;
//	}
//	
//	private static float norm(float[] x) {
//		float s = 0;		
//		for (int N = x.length, i = 0; i < N; ++i)
//			s += x[i]*x[i];
//		s = Math.sqrt(s);
//		for (int N = x.length, i = 0; i < N; ++i)
//			x[i] /= s;
//		return s;
//	}
}
