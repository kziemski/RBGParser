package parser;

public class DependencyArcList {
	public int n;
	public int[] st, edges;
	
	public DependencyArcList(int n)
	{
		this.n = n;
		st = new int[n];
		edges = new int[n];
	}
	
	public DependencyArcList(int[] heads)
	{
		n = heads.length;
		st = new int[n];
		edges = new int[n];
		constructDepTreeArcList(heads);
	}
	
	public int startIndex(int i)
	{
		return st[i];
	}
	
	public int endIndex(int i) 
	{
		return (i >= n-1) ? n-1 : st[i+1];
	}
	
	public int get(int i)
	{
		return edges[i];
	}
	
	public void constructDepTreeArcList(int[] heads) 
	{
		
		for (int i = 0; i < n; ++i)
			st[i] = 0;
		
		for (int i = 1; i < n; ++i) {
			int j = heads[i];
			++st[j];
		}
				
		for (int i = 1; i < n; ++i)
			st[i] += st[i-1];
		
		//Utils.Assert(st[n-1] == n-1);
		
		for (int i = n-1; i > 0; --i) {
			int j = heads[i];
			--st[j];
			edges[st[j]] = i;
		}
	}
}
