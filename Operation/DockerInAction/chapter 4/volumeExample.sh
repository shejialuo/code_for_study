#!/bin/bash

docker volume create \
  --driver local
  --label example=cassandra \
  cass-shared

docker run -d \
  --volume cass-shared:/var/lib/cassandra/data \
  --name cass1 \
  cassandra:2.2

docker run -it --rm \
  --link cass1:cass \
  cassandra:2.2 cqlsh cass

docker run -d \
  --volume cass-shared:/var/lib/cassandra/data \
  --name cass2 \
  cassandra:2.2

docker run -it --rm \
  --link cass2:cass \
  cassandra:2.2 \
  cqlsh cass
