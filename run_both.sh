#!/bin/bash

langsam=$1
rosnode=$2

# Name of the tmux session
SESSION_NAME="joe_session"

# Start a new tmux session in detached mode
tmux new-session -d -s $SESSION_NAME

if [ "$langsam" == "yes" ]; then
    tmux rename-window -t $SESSION_NAME:0 'langsam'
    tmux send-keys -t $SESSION_NAME:0 'conda activate langsam' C-m
    tmux send-keys -t $SESSION_NAME:0 'python /home/user/workspace/src/perception/langsam_pipe.py' C-m
fi

sleep 5

if [ "$rosnode" == "yes" ]; then
    tmux new-window -t $SESSION_NAME:1 -n 'rosnode'
    tmux send-keys -t $SESSION_NAME:1 'conda activate ovir3d' C-m
    tmux send-keys -t $SESSION_NAME:1 'python /home/user/workspace/src/perception/sam_service_rospy.py' C-m
fi

tmux attach-session -t $SESSION_NAME
