#!/bin/bash

#Optionally run the docker container, and run the client right here in the tmux

gsam_server=$1

#If any arg
if [ -n "$gsam_server" ]; then
    SESSION_NAME="joe_session"

    #Run the docker container
    tmux new-session -d -s $SESSION_NAME
    tmux rename-window -t $SESSION_NAME:0 'gsam server'
    tmux send-keys -t $SESSION_NAME:0 'python ~/catkin_ws/src/segment3d/run_container_gsam.py' C-m

    sleep 5

    tmux new-window -t $SESSION_NAME:1 -n 'client'
    tmux send-keys -t $SESSION_NAME:1 'python ~/catkin_ws/src/segment3d/src/gsam_service.py' C-m

    tmux attach-session -t $SESSION_NAME
else
    python ~/catkin_ws/src/segment3d/src/gsam_service.py
fi


