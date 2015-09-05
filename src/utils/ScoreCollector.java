package utils;

public class ScoreCollector implements Collector {
	
	float[] weights;
	public float score;
	
	public ScoreCollector(float[] w) {
		weights = w;
		score = 0;
	}
	
	@Override
	public void addEntry(int x) {
		score += weights[x];
	}

	@Override
	public void addEntry(int x, float va) {
		score += weights[x]*va;
	}
	
}
