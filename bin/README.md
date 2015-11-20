
### RBGParser v1.1

This version improves parsing speed using the hash kernel (see [4]) and by optimizing the code. We also improved the unlabeled attachment score (UAS) slightly and labeled attachment score (LAS) significantly. 
  * feature index lookup: use hash kernel (i.e. ignoring collisions) instead of a look-up table
  * dependency labels: now use a complete set of first-order features; will consider adding rich features later
  * online update method: a slightly modified version
  * optimized feature cache at code level
  * now can prune low-frequent labels, words, etc.

=========

#### About and Contact

This project is developed at Natural Language Processing group in MIT. It contains a Java implementation of a syntactic dependency parser with tensor decomposition and greedy decoding, described in [1,2,3].

This project is implemented by Tao Lei (taolei [at] csail.mit.edu) and Yuan Zhang (yuanzh [at] csail.mit.edu).

=========

#### Usage

##### 1. Compilation

To compile the project, first do a "make" in directory lib/SVDLIBC to compile the [SVD library](http://tedlab.mit.edu/~dr/SVDLIBC/). Next, make sure you have Java JDK installed on your machine and find the directory path of Java JNI include files. The directory should contains header files *jni.h* and *jni_md.h*. Take a look or directly use the shell script *make.sh* to compile the rest of the Java code. You have to replace the "jni_path" variable in *make.sh* with the correct JNI include path. Also, create a "bin" directory in the project directory before running *make.sh* script. 


<br> 

##### 2. Data Format

The data format of this parser is the one used in CoNLL-X shared task, which describes a collection of annotated sentences (and the corresponding gold dependency structures). See more details of the format at [here](http://ilk.uvt.nl/conll/#dataformat) and [here](https://code.google.com/p/clearparser/wiki/DataFormat#CoNLL-X_format_%28conll%29). We use annotated <b>non-projective</b> dependency trees provided in the data.


<br>

##### 3. Example Usage

###### 3.1 Basic Usage

Take a look at *run.sh* as an example of running the parser. You could also run the parser as follows. The first thing is to add the RBGParser directory to the library path such that the parser can find the compiled jni library for SVD tensor intialization. Assuming the directory is "/path/to/rbg", this can be done by:
```
export LD_LIBRARY_PATH="/path/to/rbg:${LD_LIBRARY_PATH}"
```

After this, we can run the parser:
```
java -classpath "bin:lib/trove.jar" -Xmx32000m parser.DependencyParser \
  model-file:example.model \
  train train-file:example.train \
  test test-file:example.test \
  output-file:example.out
```

This will train a parser from the training data *example.train*, save the dependency model to the file *example.model*, evaluate this parser on the test data *example.test* and output dependency predictions to the file *example.out*.


###### 3.2 More Options

The parser will train a 3rd-order parser by default. To train a 1st-order (arc-based) model, run the parser like this:
```
java -classpath "bin:lib/trove.jar" -Xmx32000m parser.DependencyParser \
  model-file:example.model \
  train train-file:example.train \
  test test-file:example.test \
  model:basic
```
The argument ``model:MODEL-TYPE'' specifies the model type (basic: 1st-order features, standard: 3rd-order features and full: high-order global features).

There are many other possible running options. Here is a more complicated example:
```
java -classpath "bin:lib/trove.jar" -Xmx32000m parser.DependencyParser \
  model-file:example.model \
  train train-file:example.train \
  test test-file:example.test \
  output-file:example.out \
  model:standard  C:1.0  iters:5  pruning:false \
  R:20 gamma:0.3 thread:4 converge-test:50
```
This will run a standard model with regularization *C=1.0*, number of training iteration *iters=5*, rank of the tensor *R=20*, number of threads in parallel *thread=4*, weight of the tensor component *gamma=0.3*, the number of adaptive hill-climbing restarts during testing *converge-test=50*, and no dependency arc pruning *pruning=false*. You may take a look at RBGParser/src/parser/Options.java to see a full list of possible options.


###### 3.3 Using Word Embeddings

To add unsupervised word embeddings (word vectors) as auxiliary features to the parser. Use option "word-vector:WORD-VECTOR-FILE":
```
java -classpath "bin:lib/trove.jar" -Xmx32000m parser.DependencyParser \
  model-file:example.model \
  train train-file:example.train \
  test test-file:example.test \
  model:basic \
  word-vector:example.embeddings
```
The input file *example.embeddings* should be a text file specifying the real-value vectors of different words. Each line of the file should starts with the word, followed by a list of real numbers representing the vector of this word. For example:
```
this 0.01 0.2 -0.05 0.8 0.12
and 0.13 -0.1 0.12 0.07 0.03
to 0.11 0.01 0.15 0.08 0.23
*UNKNOWN* 0.04 -0.14 0.03 0.04 0
...
...
```
There may be a special word \*UNKNOWN\* used for OOV (out-of-vocabulary) word. Each line should contain the same number of real numbers. 

======

#### References

[1] Tao Lei, Yu Xin, Yuan Zhang, Regina Barzilay and Tommi Jaakkola. Low-Rank Tensors for Scoring Dependency Structures.  ACL 2014. [PDF](http://people.csail.mit.edu/taolei/papers/acl2014.pdf)

[2] Yuan Zhang, Tao Lei, Regina Barzilay, Tommi Jaakkola and Amir Globerson. Steps to Excellence: Simple Inference with Refined Scoring of Dependency Trees.  ACL 2014. [PDF](http://people.csail.mit.edu/yuanzh/papers/acl2014.pdf)

[3] Yuan Zhang\*, Tao Lei\*, Regina Barzilay and Tommi Jaakkola. Greed is Good if Randomized: New Inference for Dependency Parsing. EMNLP 2014. [PDF](http://people.csail.mit.edu/taolei/papers/emnlp2014.pdf)

[4] Bernd Bohnet. Very High Accuracy and Fast Dependency Parsing is not a Contradiction. The 23rd International Conference on Computational Linguistics. COLING 2010. [PDF](http://anthology.aclweb.org/C/C10/C10-1011.pdf)
