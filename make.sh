#!/bin/sh

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:."

javac -d bin -sourcepath src -classpath "lib/trove.jar" src/parser/DependencyParser.java



