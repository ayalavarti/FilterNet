FROM ubuntu:16.04
FROM python:3.6

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

COPY . /app

ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

EXPOSE 8080
CMD ["gunicorn", "--bind", ":8080", "app:app"]
