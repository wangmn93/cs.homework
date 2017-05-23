import numpy
import matplotlib.pyplot as plt
import math
gamma= 0.99
alpha = 0.1
epsilon = 0.0
lamda = 0.5

X_UPPER = 0.5
X_LOWER =-1.2
X_DISCRETE= 20.0

Y_UPPER = 0.07
Y_LOWER = -0.07
Y_DISCRETE= 20.0

N_OF_ACTIONS= 3
Q = numpy.zeros(shape=(int(X_DISCRETE),int(Y_DISCRETE),N_OF_ACTIONS))
E = numpy.zeros(shape=(int(X_DISCRETE),int(Y_DISCRETE),N_OF_ACTIONS))
MAX_STEP_EPISODE=10**5
EPISODES=100
timesteps=[0]*EPISODES
start = [0,0]

REPEAT=10

def repeat_learning():
	Average = [0]*EPISODES
	Q = numpy.zeros(shape=(int(X_DISCRETE),int(Y_DISCRETE),N_OF_ACTIONS))

	for x in range(REPEAT):
		Q = numpy.zeros(shape=(int(X_DISCRETE),int(Y_DISCRETE),N_OF_ACTIONS))
		ALL_timesteps = do_learning(x)
		print "finished trial "+ str(x)
		for j in range(EPISODES):
			Average[j]+= ALL_timesteps[j]
	#Averaging and showing
	for j in range(EPISODES):
		Average[j]/=REPEAT
	#show Averaging
	
	x = numpy.arange(EPISODES)
	plt.plot(x,Average,label = "Average")
	plt.ylabel('Time Steps for '+str(X_DISCRETE)+" dicretization")
	plt.xlabel('Episodes')
	plt.legend()
	plt.show()
	

def do_learning(n):
	timesteps=[0]*EPISODES
	for i in range(EPISODES):
		timesteps[i] = do_episode()
		if(timesteps[i]>=MAX_STEP_EPISODE):
			print "Exceed no of timesteps :: ",
		print "finished Episode "+ str(n)+"."+str(i)+" in "+str(timesteps[i])+" steps"
	#show result of learning
	x = numpy.arange(EPISODES)
	plt.plot(x,timesteps,label = "Trial: "+str(n+1))
	plt.ylabel('Time Steps')
	plt.xlabel('Episodes')	
	return timesteps

def do_episode():
	steps = 0
	s = start
	E = numpy.zeros(shape=(int(X_DISCRETE),int(Y_DISCRETE),N_OF_ACTIONS))

	a = epsilonGreedy(s)
	while s[0]<0.5 and steps<MAX_STEP_EPISODE:
		#learn
		x = getDisceretePosition(s[0])
		y = getDiscereteVelocity(s[1])
		#print "step "+str(steps)+ ", a = "+str(a)+"current position ",s[0],y		
		[r,sprime] = update_state(s,a)
		adash = epsilonGreedy(sprime)
		astar = findmax(sprime)
		delta = r+ getq(sprime,astar)-getq(s,a)
		E[x,y,a] = 1
		#print E[x,y,a]
		for i in range(len(Q)):
			for j in range(len(Q[0])):
				for k in range(len(Q[0,0])):
					
					Q[i,j,k]+= alpha*delta*E[i,j,k]
					if adash == astar:
						E[i,j,k] = gamma * lamda* E[i,j,k]
					else:
						E[i,j,k] = 0
		#print Q
		s = sprime
		a = adash
		#print steps
		steps+=1

	return steps
def getq(s,a):
	return Q[getDisceretePosition(s[0]),getDiscereteVelocity(s[1]),a]
def random(current_pos):	
	random_action = numpy.random.choice(numpy.arange(-1, 2), p =  getprobability(current_pos))
	return random_action + 1 

def epsilonGreedy(current_pos):
	max_action =  findmax(current_pos)	
	random_action = numpy.random.choice(numpy.arange(-1, 2), p =  getprobability(current_pos))
	action = numpy.random.choice([max_action,random_action+1],p=[1-epsilon,epsilon])	
	#if action == max_action:
	#	print "max action"
	#elif action == random_action :
	#	print "random action"

	return action
def findmax(current_pos):
	max_action=-1
	max_q=-100000000
	x = getDisceretePosition(current_pos [0])
	y = getDiscereteVelocity(current_pos[1])
	#print current_pos,x,y
	normalvalue = -100000000
	for i in range(-1, 2):
		if Q[x,y,i+1] > max_q:
			max_action = i
			max_q = Q[x,y,i+1]
		else:
			normalvalue= Q[x,y,i+1]
	if max_q == normalvalue:		
		max_action = numpy.random.choice(numpy.arange(-1, 2), p =  getprobability(current_pos))			

	return max_action + 1
	
def getprobability(current_pos):
		p=[1]*N_OF_ACTIONS
		for i in range(N_OF_ACTIONS):
			p[i]= 1.0/(N_OF_ACTIONS)
		return p

def update_state(current_state,action):
	x = current_state[0]
	y=  current_state[1]
	a = action -1
	x = x + y
	y = y + (0.001*a)-(0.0025 *math.cos(3*current_state[0]))

	if (x >= 0.5):
		return [0,[x,y]]
	elif (x<=-1.2):
		return [-1,[0,y]]	
	else:
		return [-1,[x,y]]


def getDisceretePosition(pos):
	disp= 0-X_LOWER;	
	return int(math.floor(((pos+disp)*(X_DISCRETE-1))/(X_UPPER+disp)))

def getDiscereteVelocity(pos):
	disp= 0-Y_LOWER;
	return int(math.floor(((pos+disp)*(Y_DISCRETE-1))/(Y_UPPER+disp)))

repeat_learning()
