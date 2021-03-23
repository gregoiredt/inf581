import gym
from gym import spaces
import pybullet as p
import time
import pybullet_data
import numpy as np
from IPython.display import clear_output


# Initial parameters, those can be modified in this file if one want to try other type of initial behaviors.
TIME_LIM = 240*3
X_LIM, Y_LIM, Z_LIM = 1, 1, 3
INITIAL_KEN_POS, INITIAL_KEN_OR = [0,0,0.7], [0,0,0]
INITIAL_DAMA_POS, INITIAL_DAMA_OR = [0,0,0.3], [0, 3.14/2, 0]


class KendamaEnv(gym.Env):
  """Custom Environment that follows gym interface to model the simulation of a 3D Kendama"""

  def __init__(self, render=True):
    '''
    Initialisation of the environment, no arguments are required. Initialisation parameters can only be modified in the .py file.
    '''
    super(KendamaEnv, self).__init__()
    self.dt = 240.0

    # 2 (force, torque) * 3D
    self.action_space = spaces.Box(low=np.array([-1,-1,-1,-1,-1,-1]), high=np.array([1,1,1,1,1,1]))
    # 2 objects * 3 (pos, vit, acc) * 3D
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(7,3), dtype=np.float32)

    
    # Physics rendering
    if(render):
      self.physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    else:
      self.physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    
    # Gravity initialisation
    p.setGravity(0,0,-10)

    # Downloading Ken and setting constraints
    cubeStartPos = INITIAL_KEN_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_KEN_OR)
    self.ken = p.loadURDF("./URDF/kendama_hand/kendama_hand.urdf",cubeStartPos, cubeStartOrientation, 
                    flags=p.URDF_USE_INERTIA_FROM_FILE)        
    self.ken_constraint = p.createConstraint(self.ken, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0,0,0], cubeStartPos)

    # Downloading dama
    cubeStartPos = INITIAL_DAMA_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_DAMA_OR)
    self.dama = p.loadURDF("./URDF/kendama_ball/kendama_ball.urdf",cubeStartPos, cubeStartOrientation, 
                    flags=p.URDF_USE_INERTIA_FROM_FILE)

    # Activtating constraints and center, the center model the joint bewteen the string and the ken.
    self.center = p.loadURDF("./URDF/cube/cube.urdf",INITIAL_KEN_POS+np.array([0,0,0.1]),flags=p.URDF_USE_INERTIA_FROM_FILE)
    p.createConstraint(self.ken, -1, self.center, -1, p.JOINT_POINT2POINT,jointAxis=[1,0,0],parentFramePosition=[0,0,0],childFramePosition=[0,0,0])
    self.pulling = False
    self.pulling2 = False
    self.wasHigher = False

    self.targetKenPos = np.array(INITIAL_KEN_POS)
    self.targetKenAng = np.array(INITIAL_KEN_OR)
    # Activate Pulling if functionning
    if(self.pulling):
      posAttacheDama, angleDama = p.getBasePositionAndOrientation(self.dama)
      posAttacheDama = np.array(posAttacheDama) + np.matmul(np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(angleDama),[3,3])), np.array([-0.03,0,0]))
      posCenterCube, angleCenterCube = p.getBasePositionAndOrientation(self.center)
      posCenterCube = np.array(posCenterCube)
      vec = posAttacheDama - posCenterCube
      vec = np.matmul(vec, np.reshape(p.getMatrixFromQuaternion(angleCenterCube),[3,3])) # vec dans le référentiel du cube

      self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,[1,1,1], vec/np.linalg.norm(vec)*0.5, [-0.03,0,0])
      p.changeConstraint(self.link,maxForce=10)
    
    # Initialisation
    self.time = 0
    self.killNextTime = False
    self.list_reward = []
    self.hasTouched = False
    self.hasBeenDetached = False
    
    # Configuration of the visualisation
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(1.18,40.2,-25.6,[0,0,0.82])


  def out_of_time(self):
    '''
    Checks if the simulation has finished, given the 
    '''
    return self.time > TIME_LIM
  
  # check spatial limits
  def out_of_box(self):
    '''
    Checks if the ken or the dama are outside from a certain Bbox
    '''
    kx, ky, kz = np.absolute(p.getBasePositionAndOrientation(self.ken)[0])
    dx, dy, dz = np.absolute(p.getBasePositionAndOrientation(self.dama)[0])


    if ( kx > X_LIM or dx > X_LIM):
      return True
    if ( ky > Y_LIM or dy > Y_LIM):
      return True
    if(p.getBasePositionAndOrientation(self.ken)[0][2] < 0 or p.getBasePositionAndOrientation(self.ken)[0][2] > 2):
      return True

    return False

  def step(self, action):
    '''
    Execute one step of the simulation from the action given, and calculate the corresponding rewards and observations.
    '''

    # Takes last reward into account while ending the simulation
    if(self.killNextTime):
      # If this function is executed, self.obs is defined.
      return self.last_obs, 0, True, {}
   
    # Resize action to prevent impossible behaviour
    action = np.array(action)/240.0*5.0

    #Updating params
    self.time += 1
    self.newPull = False
    
    # Simulating the physics :
    self._simulate_physics()

    # Action of the kendama, changing the constraint :
    prvPos, prvAngle = self.targetKenPos, self.targetKenAng
    prvPos = prvPos + np.array(action[:3])
    prvAngle = prvAngle + np.array(action[3:])
    n_orn = p.getQuaternionFromEuler(prvAngle)
    self.targetKenPos = prvPos
    self.targetKenAng = prvAngle
    p.changeConstraint(self.ken_constraint, prvPos, jointChildFrameOrientation=n_orn, maxForce=30)
    
    # Make pybullet simulation work
    p.stepSimulation()

    # Get observations from the action result
    # Get ken informations
    kenVel,kenVelRad = p.getBaseVelocity(self.ken)
    kenVel, kenVelRad = np.array(kenVel), np.array(kenVelRad)
    kenPos,kenAngle = p.getBasePositionAndOrientation(self.ken)
    kenPos, kenAngle = np.array(kenPos), np.array(kenAngle)
    kenAngle = p.getEulerFromQuaternion(kenAngle)
    kenAngle = np.array(kenAngle)

    # Get dama observations
    damaVel,damaVelRad = p.getBaseVelocity(self.dama)
    damaVel, damaVelRad = np.array(damaVel), np.array(damaVelRad)
    damaPos,damaAngle = p.getBasePositionAndOrientation(self.dama)
    damaAngle = p.getEulerFromQuaternion(damaAngle)
    damaPos, damaAngle = np.array(damaPos), np.array(damaAngle)
    
    # Observation :
    observation = np.array([damaPos, kenPos, damaVel, damaAngle ,kenAngle, damaVelRad ,kenVelRad])
    observation = self.normalizeObs(observation)
    reward, done = self.get_reward(damaPos, kenPos, damaVel, kenVel, damaAngle, kenAngle, damaVelRad, kenVelRad,action)
    
    self.last_obs= observation
    
    return observation, reward, done, {}
  
  def normalizeObs(self, obs):
    '''
    Renormalize the observation in order for it to be in the box required by the algorithm
    '''
    obs[0] = np.clip(obs[0] + np.array([0,0,-1]),-1,1)
    return obs

  def reset(self):
    '''
    Reset the whole environment. Some randomness is added to the initial position
    '''
    # Ken initial position
    cubeStartPos = INITIAL_KEN_POS
    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_KEN_OR)
    p.resetBasePositionAndOrientation(self.ken, cubeStartPos, cubeStartOrientation)

    # Dama initial position is set randomly
    cubeStartPos = np.random.normal(np.array(INITIAL_DAMA_POS), scale = 0.01, size=3)

    cubeStartOrientation = p.getQuaternionFromEuler(INITIAL_DAMA_OR)
    p.resetBasePositionAndOrientation(self.dama, cubeStartPos, cubeStartOrientation)
    p.resetBasePositionAndOrientation(self.center, INITIAL_KEN_POS + np.array([0,0,0.1]), p.getQuaternionFromEuler([0,0,0]))
    n_orn = p.getQuaternionFromEuler(INITIAL_KEN_OR)
    p.changeConstraint(self.ken_constraint, INITIAL_KEN_POS, jointChildFrameOrientation=n_orn, maxForce=30)
    
    # Reinitialisation of the parameters
    self.targetKenAng = np.array(INITIAL_KEN_OR)
    self.targetKenPos = np.array(INITIAL_KEN_POS)
    self.killNextTime = False
    self.time = 0
    # Get observations from the action result

    # Get ken informations
    kenVel,kenVelRad = p.getBaseVelocity(self.ken)
    kenVel, kenVelRad = np.array(kenVel), np.array(kenVelRad)
    kenPos,kenAngle = p.getBasePositionAndOrientation(self.ken)
    kenPos, kenAngle = np.array(kenPos), np.array(kenAngle)
    kenAngle = p.getEulerFromQuaternion(kenAngle)
    kenAngle = np.array(kenAngle)

    # Get dama observations
    damaVel,damaVelRad = p.getBaseVelocity(self.dama)
    damaVel, damaVelRad = np.array(damaVel), np.array(damaVelRad)
    damaPos,damaAngle = p.getBasePositionAndOrientation(self.dama)
    damaAngle = p.getEulerFromQuaternion(damaAngle)
    damaPos, damaAngle = np.array(damaPos), np.array(damaAngle)

    # Initial observation
    observation = np.array([damaPos, kenPos, damaVel, damaAngle ,kenAngle, damaVelRad ,kenVelRad])
    observation = self.normalizeObs(observation)

    self.wasHigher = False
    self.hasTouched = False
    self.hasBeenDetached = False

    return observation  # reward, done, info can't be included
  
  
  def close (self):
    p.disconnect()
  
  def _simulate_physics(self):
    '''
    Simulate all the physics from the simulation
    '''
    # Check tension
    if(self.pulling):
        force = p.getConstraintState(self.link)
    else :
        force = [0,0,0]
    
    # We work the dynamics of the system
    posAttacheDama, angleDama = p.getBasePositionAndOrientation(self.dama)
    posAttacheDama = np.array(posAttacheDama)  + np.matmul(np.array([-0.03,0,0]),np.linalg.inv(np.reshape(p.getMatrixFromQuaternion(angleDama),[3,3])))
    posCenterCube, angleCenterCube = p.getBasePositionAndOrientation(self.center)
    posCenterCube = np.array(posCenterCube)

    vec = posAttacheDama - posCenterCube
    vec = np.matmul(vec,np.reshape(p.getMatrixFromQuaternion(angleCenterCube),[3,3])) # vecteur dans le référentiel du cube

    
    dirFil = posAttacheDama - posCenterCube
    tension = np.dot(dirFil/np.linalg.norm(dirFil),np.array(force))
    # What happens when there is no more tension
    if( self.pulling and tension > 0): #np.linalg.norm(vec)<0.5
        self.pulling = False
        self.hasBeenDetached = True
        p.removeConstraint(self.link)
    
    # What happens when tension comes back
    elif(np.linalg.norm(vec) > 0.5 and not self.pulling):
        self.pulling = True
        self.pulling2 = True
        self.newPull = True
        self.link = p.createConstraint(self.center, -1, self.dama, -1, p.JOINT_POINT2POINT,[1,1,1], vec/np.linalg.norm(vec)*0.5, np.array([-0.03,0,0]))
        p.changeConstraint(self.link,maxForce=100)
    
    # Friction when there is tension
    if self.pulling : 
        friction = 0.1
        proj_vitesse = np.dot(dirFil/np.linalg.norm(dirFil), np.array(p.getBaseVelocity(self.dama)[0])) * dirFil/np.linalg.norm(dirFil)
        friction_force = - friction * (proj_vitesse - np.array(p.getBaseVelocity(self.ken)[0]))
        p.applyExternalForce(objectUniqueId=self.dama, linkIndex=-1,
                         forceObj=friction_force, posObj=np.array(p.getBasePositionAndOrientation(self.dama)[0]), flags=p.WORLD_FRAME)
    
    # Friction to dissipate energy
    friction = 0.03
    friction_force = - friction * np.array(p.getBaseVelocity(self.dama)[0])
    p.applyExternalForce(objectUniqueId=self.dama, linkIndex=-1,
                      forceObj=friction_force, posObj=np.array(p.getBasePositionAndOrientation(self.dama)[0]), flags=p.WORLD_FRAME) 

    # Friction on the angular rotation of the dama
    p.changeDynamics(self.dama, -1, angularDamping=1)


  def iscolliding(self):
    '''
    Determine if there is a collision between the kendama and the ball
    '''
    localOrientation = np.array([0,0,1])
    r = np.reshape(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.ken)[1]),[3,3])
    vect_ken_orientation = np.matmul(localOrientation, np.linalg.inv(r))

    pdama = np.array(p.getBasePositionAndOrientation(self.dama)[0]).copy()
    pken = np.array(p.getBasePositionAndOrientation(self.ken)[0])
    pken += vect_ken_orientation / np.linalg.norm(vect_ken_orientation) * 0.077 # Spike middle position
    return np.linalg.norm(pdama-pken) < 0.005 # Threshold has been determined after some tests

  def get_reward(self, damaPos, kenPos, damaVel, kenVel, damaAngle, kenAngle, damaVelRad, kenVelRad,action):
    '''
    Compute the rewards given the observartion. Every agr must be an array !
    '''

    done = False
    reward = 0

    # Dama is under the ken
    if damaPos[2] < kenPos[2]:
      if(self.wasHigher):
        return 0, True
      reward += 0.1*min(damaPos[2], 1)


    # Dama is above the ken
    else :
      self.wasHigher=True
      if(len(p.getContactPoints(bodyA=self.ken, bodyB=self.dama)) > 0 ):
        reward += 0.2
      
      # Computation of the useful direction vectors
      localOrientation = np.array([0,0,1])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(kenAngle)),[3,3])
      vect_ken_orientation = np.matmul(localOrientation, np.linalg.inv(r))
      spike_ken = vect_ken_orientation / np.linalg.norm(vect_ken_orientation) * 0.077 # Spike middle position
            
      localOrientation = np.array([1,0,0])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(damaAngle)),[3,3])
      vect_dama_orientation = np.matmul(localOrientation, np.linalg.inv(r))
      
      localOrientation = np.array([0,0,1])
      r = np.reshape(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(kenAngle)),[3,3])
      vect_ken_orientation = np.matmul(localOrientation, np.linalg.inv(r))

      # Rewards :
      reward += 0.2 * np.exp(-15.0*np.linalg.norm(damaPos - kenPos - spike_ken)**2)
      reward += 10 * np.exp(-500.0*np.linalg.norm(damaPos - kenPos - spike_ken)**2)
      reward += 0.2 * np.exp(-7*(np.dot(vect_ken_orientation, vect_dama_orientation)+1)**2)
      reward += 40 * np.exp(-20*(np.dot(vect_ken_orientation, vect_dama_orientation)+1)**2) *  np.exp(-100.0*np.linalg.norm(damaPos - kenPos - spike_ken)**2)
      reward += 0.2*np.exp(-10*(np.dot(vect_ken_orientation, damaVel/np.linalg.norm(damaVel))+1)**2) # exp( - (u*v +1)**2 )
      

    reward += 0.3*np.exp(- 3.0* np.linalg.norm(kenPos - np.array(INITIAL_KEN_POS))**2)
    reward += 0.3*np.exp(- 3.0* np.linalg.norm(kenAngle - np.array(INITIAL_KEN_OR))**2)
    reward += 0.3*np.exp(- 3.0* np.linalg.norm(damaPos[:2] - np.array(INITIAL_DAMA_POS[:2]))**2)
    reward += 0.3*np.exp(- 3.0* np.linalg.norm(damaAngle - np.array(INITIAL_DAMA_OR))**2)
    reward += 4*np.exp(- 10.0* np.linalg.norm(damaAngle - np.array(INITIAL_DAMA_OR))**2)
    reward += 0.3*np.exp(- np.linalg.norm(damaVelRad)**2/(2*3.14*3.14))

    if damaPos[2] < kenPos[2]:
      reward /= 3

    if self.iscolliding():
      # The dama is catched
      reward = 5000.0
      done = False
      self.killNextTime = True
      print("IT'S A CATCH !")

    if self.out_of_box():
      self.killNextTime = True
      done = False
      reward = -1000

    if self.out_of_time():
      done = False
      self.killNextTime = True
      reward = -1000

    return reward, done

  def evaluate(self,action,reward,vmaxd,vavgd,dmin,davg,wavgd,catch,time,box):
    '''
    This function lets the user analyse the performance of the trained agent
    '''

    obs, rewards, dones, info = self.step(action)
    reward.append(rewards)
    damaPos, kenPos, damaVel, _ , _ , damaVelRad , _ = obs

    vd = np.linalg.norm(damaVel)
    if vd > vmaxd[0]:
      vmaxd[0] = vd
    vavgd.append(vd)

    d = np.linalg.norm(kenPos - damaPos)
    if d < dmin[0]:
      dmin[0] = d
    davg.append(d)

    wavgd.append(np.linalg.norm(damaVelRad))

    if self.iscolliding():
      catch[0] += 1

    if self.out_of_time():
      time[0] += 1

    if self.out_of_box():
      box[0] += 1

    return obs, rewards, dones, info
