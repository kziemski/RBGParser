#!/bin/sh

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:."

JNI_PATH="jni_include"

javac -d bin -sourcepath src -classpath "lib/trove.jar" src/parser/DependencyParser.java



