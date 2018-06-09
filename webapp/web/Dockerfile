FROM alpine
MAINTAINER Alex Egg <eggie5@gmail.com>

# basic flask environment
RUN apk add --no-cache bash git nginx uwsgi uwsgi-python py2-pip postgresql-dev build-base \
	&& pip2 install --upgrade pip \
	&& pip2 install flask
RUN apk --no-cache add musl-dev python-dev py-numpy linux-headers g++
RUN pip2 install psycopg2 Flask-SQLAlchemy
# application folder
ENV APP_DIR /app

# app dir
RUN mkdir $APP_DIR
WORKDIR $APP_DIR


ADD . $APP_DIR/

COPY entrypoint.sh /entrypoint.sh
# exectute start up script
ENTRYPOINT ["/entrypoint.sh"]