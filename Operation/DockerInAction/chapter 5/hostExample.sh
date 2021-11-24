#!/bin/bash

docker run --rm \
  --network host \
  alpine:3.8 ip -o addr