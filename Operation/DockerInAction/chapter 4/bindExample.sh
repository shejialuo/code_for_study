#! /bin/bash

touch ~/example.log

cat >~/example.conf << EOF
server {
  listen 80;
  server_name localhost;
  access_log /var/log/nginx/custom.host.access.log main;
  location / {
    root /usr/share/nignx/html;
    index index.html index.htm;
  }
}
EOF

CONF_SRC=~/example.conf
CONF_DST=/etc/nginx/conf.d/default.conf
LOG_SRC=~/example.log
LOG_DST=/var/log/nginx/custom.host.access.log

docker run -d --name diaweb \
  --mount type=bind,src="$CONF_SRC",dst="$CONF_DST",readonly=true \
  --mount type=bind,src="$LOG_SRC",dst="$LOG_DST" \
  -p 80:80 \
  nginx:latest
