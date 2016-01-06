

### Labeler



This project is used to assign labels to an (unlabeled) dependency tree by a distributional representation learning technique for scoring and a dynamic programming algorithm for labeling. 



=========



#### Usage



##### 1. Compiling



Make sure you have Java JDK installed on your machine, then run the following command to compile the Java code (please create a "bin" directory in the project directory before running it):

```

javac -d bin -sourcepath src -classpath "lib/trove.jar" src/parser/DependencyParser.java

```





<br> 



##### 2. Data Format



We support CoNLL-09 (default) and CoNLL-06 (by specifying the argument "format:CONLL-06") data format.





<br>



##### 3. Example Usage



###### 3.1 Basic Usage



Run the labeler by command:

```

java -classpath "bin:lib/trove.jar" -Xmx20000m parser.DependencyParser \

  model-file:example.model \

  train train-file:example.train \

  test test-file:example.test \

  pred-file:example.pred \

  output-file:example.out

```

This will train the labeler from the training data *example.train*, save the labeling model to the file *example.model*, assign labels to (unlabeled) dependency trees in *example.pred* and evaluate against the test data *example.test* (*example.pred* and *example.test* should match except for the "HEAD" and "DEPREL" column). Labeling results are output to the file *example.out*.





###### 3.2 More Options



There are many other possible running options. Here is a more complicated example:

```

java -classpath "bin:lib/trove.jar" -Xmx20000m parser.DependencyParser \

  model-file:example.model \

  train train-file:example.train \

  test test-file:example.test \

  pred-file:example.pred \

  output-file:example.out\

  model:second  C:1.0  iters:5 \

  R:100 R2:50 gammaLabel:0.3

```

This will run a 2nd-order model with regularization *C=1.0*, number of training iterations *iters=5*, rank of the first-order tensor *R=100* and second-order tensor *R2=50*, and weight of the traditional features in the scoring function *gammaLabel=0.3* (note that when traditional features incorporated, i.e. *gammaLabel*>0, the labeler will be significantly slowed down).



You may take a look at labeler/src/parser/Options.java to see the full list of possible options.
