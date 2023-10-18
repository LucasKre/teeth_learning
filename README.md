# Teeht Learning

## Run it
Setup your local environment using docker
1. Build docker container with `docker build -t teeth_learning .`
2. Run docker container with `docker run --name teeth_learning -d -v "%cd%":/app/ --gpus all -p 8888:8888  -e JUPYTER_TOKEN=password teeth_learning`

