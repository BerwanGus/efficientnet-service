docker build . -t enet-streamlit
docker run --rm -p 8001:8001 --name enet-streamlit-1 -d enet-streamlit