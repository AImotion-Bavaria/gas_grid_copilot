# Tip for handling screen
You can use screen
```
sudo apt-get install screen

screen 

./run-my-script
```
Ctrl-A then D to get out of your screen
From there you will be able to close out your ssh terminal. Come back later and run
```
screen -ls

screen -r $screen_running
```
The screen running is usually the first 5 digits you see after you've listed all the screens. You can see if you're script is still running or if you've added logging you can see where in the process you are.