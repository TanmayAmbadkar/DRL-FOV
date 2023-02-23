params = {
    "images_dir": "360_saliency_dataset_2018eccv/output",
    "Environment": {
        "n_frames": 2,
        "video_frames": 200,
        "frame_width": 160,
        "frame_height": 90,
    },
    "Agent": {
        "n_frames": 2,
        "gamma": 0.3,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_final": 0.01,
        "lr": 0.001,
        "episodes": 100, 
        "save_path": "saved_agents/agent2frame.model"},
    "train": True,
}
