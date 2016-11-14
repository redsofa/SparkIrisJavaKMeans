#!/bin/bash

spark-submit --class ca.redsofa.jobs.KMeansApp \
--master local[*] \
./target/SparkIrisJavaKMeans-1.0-SNAPSHOT.jar \
data/iris.data