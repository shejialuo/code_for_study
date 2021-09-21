FROM gcc:latest

RUN mkdir /usr/src/microservices

COPY HALoadBalancer /usr/src/microservices

WORKDIR /usr/src/microservices

CMD ["./HALoadBalancer"]