#!/bin/bash
if [ $1 -eq 1 ]
then
  echo Activating CyLag profiling
  sed -i -r "s/# cython: profile=False/# cython: profile=True/g" cylag/*.pyx
  sed -i -r "s/# cython: profile=False/# cython: profile=True/g" cylag/*/*.pyx
  sed -i -r "s/# cython: profile=False/# cython: profile=True/g" cylag/*/*/*.pyx
else
  echo De-activating CyLag profiling
  sed -i -r "s/# cython: profile=True/# cython: profile=False/g" cylag/*.pyx
  sed -i -r "s/# cython: profile=True/# cython: profile=False/g" cylag/*/*.pyx
  sed -i -r "s/# cython: profile=True/# cython: profile=False/g" cylag/*/*/*.pyx
fi
