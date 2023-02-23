import numpy as np

class Environment():

    def __init__(self, saliency_frames, colored_frames, n_frames, video_frames, frame_width=128, frame_height=72):
        '''
            Parameters:
            saliency_frames (np.array): Saliency frames from dataloader.
            colored_frames (np.array): Color frames from dataloader.  
            n_frames (int): number of input frames to agent. 
            video_frames (int): number of frames in a single episode (video length).
            total_frames (int): number of frames in a single episode (video length).
            frame_width (int): width of FOV
            frame_height (int): height of FOV
        '''
        self.saliency_frames = saliency_frames
        self.colored_frames = colored_frames
        self.total_frames = len(saliency_frames)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_frames = video_frames
        self.n_frames = n_frames
        self.start_frame = 0
        self.current_frame = self.n_frames
        self.current_x = 480
        self.current_y = 120
        
    def reward(self, last_observation, current_observation, last_x, last_y):
        
        try:
            return np.sum(current_observation[:,self.current_y-self.frame_height//2:self.current_y+self.frame_height//2,self.current_x-self.frame_width//2:self.current_x+self.frame_width//2]) - (np.sum(last_observation[:,last_y-self.frame_height//2:last_y+self.frame_height//2,last_x-self.frame_width//2:last_x+self.frame_width//2]))
        except:
            return 0
    
    def step(self, action):
        
        last_x = self.current_x
        last_y = self.current_y
        if action == 0: #UP
            self.current_y = max(self.current_y-20, 72)
        
        if action == 1: #NorthEast
            self.current_x = (self.current_x-240+14)%480 + 240
            self.current_y = max(self.current_y-14, 72)
        
        if action == 2: #East
            self.current_x = (self.current_x-240+14)%480 + 240
        
        if action == 3: #SouthEast
            self.current_x = (self.current_x-240+14)%480 + 240
            self.current_y = min(self.current_y+14, 240-72)
        
        if action == 4: #South
            self.current_y = min(self.current_y+20, 240-72)
        
        if action == 5: #SouthWest
            self.current_x = (self.current_x-240-14)%480 + 240
            self.current_y = min(self.current_y+14, 240-72)
        
        if action == 6: #West
            self.current_x = (self.current_x-240-20)%480 + 240
        
        if action == 7: #NorthWest
            self.current_x = (self.current_x-240-14)%480 + 240
            self.current_y = max(self.current_y-14, 72)
        
        
        last_observation = self.saliency_frames[self.current_frame-self.n_frames:self.current_frame,:]
        current_observation = self.saliency_frames[self.current_frame:self.current_frame+self.n_frames,:]
        self.current_frame += self.n_frames
        done = self.current_frame - self.start_frame == self.video_frames
        reward = self.reward(self.sigmoid(last_observation), self.sigmoid(current_observation), last_x, last_y)
        
        if not done:
            next_observation = self.saliency_frames[self.current_frame:self.current_frame+self.n_frames,:,240:720]
            next_observation = self.sigmoid(next_observation)
        else:
            next_observation = None
        
        return next_observation, reward, done
            
    def reset(self):
        
        self.start_frame = np.random.randint(self.total_frames-self.video_frames-self.n_frames-1)
        self.current_frame = self.start_frame + self.n_frames
        self.current_x = 480
        self.current_y = 120
        return self.sigmoid(self.saliency_frames[self.start_frame:self.start_frame+self.n_frames,:,240:720])
    
    def render(self):
        
        current_observation = self.colored_frames[self.current_frame-self.n_frames:self.current_frame,:]
        return current_observation[:,self.current_y-self.frame_height//2:self.current_y+self.frame_height//2,self.current_x-self.frame_width//2:self.current_x+self.frame_width//2]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    