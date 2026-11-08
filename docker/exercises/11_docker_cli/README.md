# 11 - The Docker CLI commands you use every day

```text
Write the command for each task (one per line, keep the numbered comments). Then run them for real.
 1. list running containers, then ALL containers (including stopped)
 2. list local images
 3. build the image in the current folder and tag it myapp:1.0
 4. run myapp:1.0 in the background, named web, mapping host port 8080 to container port 8000
 5. follow the logs of `web`
 6. open an interactive shell inside `web`
 7. run a one-off command inside `web`: python --version
 8. show CPU / memory usage of running containers (one snapshot)
 9. stop and remove `web`
10. tag myapp:1.0 as yourname/myapp:1.0 and push it to Docker Hub (after docker login)
11. show the layers of an image and their sizes
12. clean up: remove stopped containers, unused networks, dangling images and build cache
13. run a throwaway Python 3.12 REPL in a container that is deleted on exit
14. copy a file from the container `web` (/app/main.py) to the current folder
```

## Files


- `commands.blank.sh` (you write) / `commands.solution.sh`

## Check (run in this folder)

```bash
# compare your commands.blank.sh with commands.solution.sh, then try each one on a real container
# (exercise 03's image is a good one to practice on: docker run -d --name web -p 8080:8000 gym-03)
```
