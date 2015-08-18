package utils;

public class ScoreCollector implements Collector {
	
	float[] weights;
	public double score;
	
	public ScoreCollector(float[] w) {
		weights = w;
		score = 0;
	}
	
	@Override
	public void addEntry(int x) {
		score += weights[x];
	}

	@Override
	public void addEntry(int x, double va) {
		score += weights[x]*va;
	}
	
}
