# DeepMind Atari agent
![Playing Atari breakout](breakout_gameplay.gif)


Simple Atari DQN network (requires 8GB available memory, tensorflow-gpu, open ai gym)

Trains 10,000,000 timesteps on Atari Breakout using OpenAI Gym and TensorFlow-GPU. Trains in ~12 hours on Nvidia M4000 GPU.

The network trains every 1000 timesteps on 8000 previous experiences. (experience is: state, action, reward, bool_game_over, next_state)

where each state is 4 frames of 80x80 pixels in black-and-white. 

+1 reward for each brick hit, -1 reward for each life lost. bool_game_over is set to true each time a life is lost.


For questions how to get running or more details, email me elidavis [at] me.com
