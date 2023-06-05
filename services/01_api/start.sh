docker build . -t enet-api
docker run --rm -p 8000:8000 --env-file .env --name enet-api-1 -d enet-api