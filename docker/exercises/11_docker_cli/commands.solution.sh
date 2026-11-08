# Exercise 11 - The Docker CLI commands you use every day  [cli]
# Solution for: commands.sh

# 1
docker ps
docker ps -a
# 2
docker images
# 3
docker build -t myapp:1.0 .
# 4
docker run -d --name web -p 8080:8000 myapp:1.0
# 5
docker logs -f web
# 6
docker exec -it web sh
# 7
docker exec web python --version
# 8
docker stats --no-stream
# 9
docker stop web && docker rm web        # or: docker rm -f web
# 10
docker tag myapp:1.0 yourname/myapp:1.0
docker push yourname/myapp:1.0
# 11
docker history myapp:1.0
# 12
docker system prune                     # add -a to also remove unused images, --volumes for volumes
# 13
docker run --rm -it python:3.12-slim python
# 14
docker cp web:/app/main.py .
