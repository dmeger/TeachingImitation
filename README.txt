README

Instruction:
The expert data for hopper is in the hopper data folder
To install the necessary packages, first cd in to this directory

#install gym if you don't have it
cd gym
pip3 install -e .

#install pybullet-gym
cd pybullet-gym
pip3 install -e .

#installing library for generating gifs
pip3 install imageio

The starter code takes input several command line argument
To use you can run something like, python3 run_pybullet_gym.py --lr=0.0001 --batch_size=100 --env_id="HopperPyBulletEnv-v0"
Note a few arguments
    --eval_mode="human" -> will show gui
    --eval_mode="rgb_array" -> will not show gui 
    Both will save gifs of the eval run. 

    --eval_random=1 -> Evaluation will sample action completely randomly
    --eval_random=0 -> Evaluation will sample action according to your policy defined by get_action

    To differentiate runs, you can also use --custom_id 
    This will add the custom_id to the name of checkpoint, gif and log directory 

Finally, keep the max_episode_length as 1000, that is the standard value for it. 
If you find any bug, message me on email xiru.zhu@mail.mcgill or slack