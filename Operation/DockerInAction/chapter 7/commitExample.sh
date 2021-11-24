#!/bin/bash

docker container run --name hw_container \
  ubuntu:latest \
  touch /HelloWorld

docker container commit hw_container hw_image

docker container rm -vf hw_container

docker container run --rm \
  hw_image \
  ls -l /HelloWorld