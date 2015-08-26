#!/bin/sh

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:."

type="lab"
args=$1
runid=$2
shift
shift
	java -classpath "bin:lib/trove.jar" -Xmx20000m parser.DependencyParser model-file:runs/$args.model.$type.$runid train train-file:data/$args.train.$type unimap-file:unimap/$args.uni.map test test-file:data/$args.test.$type pred-file:data/$args.test.$type $@ | tee runs/$args.$type.$runid.log


