#!/bin/sh

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:."

args=$1
runid=$2
shift
shift

	java -classpath "bin:lib/trove.jar" -Xmx20000m parser.DependencyParser model-file:runs/$args.$runid.model train train-file:data/$args.train.lab unimap-file:unimap/$args.uni.map test test-file:data/$args.test.lab $@ | tee runs/$args.$runid.log


