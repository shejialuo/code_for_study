#!/bin/bash

docker run --rm \
  --network none \
  alpine:3.8 ip -o addr