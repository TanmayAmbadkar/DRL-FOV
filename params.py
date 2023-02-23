params = {
    "images_dir": "dataset", #dataset directory
    "Environment": {
        "n_frames": 2, # number of frames as part of state
        "video_frames": 200, # Total frames in 1 episode, i.e. video length
        "frame_width": 160, # FOV width (smaller than video width)
        "frame_height": 90,# FOV height (smaller than video height)
    },
    "Agent": {
        "n_frames": 2, # number of frames as part of state
        "gamma": 0.3, # discount factor
        "epsilon": 1.0, # exploration parameter
        "epsilon_decay": 0.995, # exploration decay factor (epsilon = epsilon * epsilon_decay) 
        "epsilon_final": 0.01, # final exploration parameter
        "lr": 0.001, # learning rate for DQN
    },
    
    "save_path": "saved_agents/agent2frame.model",
    "episodes": 1,
    "train": True, # True: train and test model, False: test only (use pretrained weights)
}
