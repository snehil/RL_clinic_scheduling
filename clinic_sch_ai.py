import math 
import collections
import numpy as np
from IPython.core.debugger import Tracer

# Global Settings
verbose = False

class Schedule:
    class Slot:
        def __init__(self, slot, state, personId, isNoShow = False):
            self.hr         = slot.split(':')[0]  # Time slot hour hand
            self.mins       = slot.split(':')[1]  # Time slot mins hand
            self.state      = state               # 1 = Occupied/Unavailable; 0 = Unoccupied/Available
            self.personId   = personId            # Person ID to whom the slot is assigned to (applies only if state=1)
            self.isNoShow   = isNoShow            # Is this person predicted to no-show       (applies only if state=1)
            
        def __repr__(self):
            return '{0}: {1}: {2}: {3}; \n'.format(self.hr + ':' + self.mins, self.state, self.personId, self.isNoShow)
        
    class Day:
        def __init__(self, title, time_slots):
            self.title      = title               # Day title (e.g. Monday or Day 1, etc..)
            self.time_slots = time_slots          # Lists that holds 'Slot' objects or time-slots
            
        def __repr__(self):
            return '{0}: {1};'.format(self.title, self.time_slots)
        
    def __init__(self, slotCount=336):
        self.dayTitles     = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5', 'Day 6', 'Day 7']
        self.times         = [ '09:00', '09:10', '09:20', '09:30', '09:40', '09:50',
                               '10:00', '10:10', '10:20', '10:30', '10:40', '10:50',
                               '11:00', '11:10', '11:20', '11:30', '11:40', '11:50',
                               '12:00', '12:10', '12:20', '12:30', '12:40', '12:50',
                               '01:00', '01:10', '01:20', '01:30', '01:40', '01:50',
                               '02:00', '02:10', '02:20', '02:30', '02:40', '02:50',
                               '03:00', '03:10', '03:20', '03:30', '03:40', '03:50',
                               '04:00', '04:10', '04:20', '04:30', '04:40', '04:50',
                               '05:00', '05:10', '05:20', '05:30', '05:40', '05:50',
                               '06:00', '06:10', '06:20', '06:30', '06:40', '06:50'
                              ]
        
        self.overtime_hrs  = [ '05:00', '05:10', '05:20', '05:30', '05:40', '05:50',
                               '06:00', '06:10', '06:20', '06:30', '06:40', '06:50'
                             ]
        
        self.days          = []  # List that hold 'Day' Objects
        
        # Define lookup tables
        self.consultLookup = {0:'10', 1:'20', 2:'30', 3:'40', 4:'50', 5:'60'}
        self.hrLookup      = {0:'09', 1:'10', 2:'11', 3:'12', 4:'01', 5:'02', 6:'03', 7:'04'}
        self.minsLookup    = {0:'00', 1:'10', 2:'20', 3:'30', 4:'40', 5:'50'}
        
        # Create empty schedules for all available days
        for dayTitle in self.dayTitles:
            self.days.append(self.Day(dayTitle, self.generateEmptySlots()))
        
        self.updateState()
            
    def reset(self):
        if verbose:
            print("Resetting the Schedule..")
            
        self.days = []
    
        # Create empty schedules for all available days
        for dayTitle in self.dayTitles:
            self.days.append(self.Day(dayTitle, self.generateEmptySlots()))
            
        self.updateState()
            
    def generateEmptySlots(self):
        time_slots = []
        
        for time in self.times:
            time_slots.append(self.Slot(time, 0, ''))
        
        return time_slots
    
    def display(self):
        if verbose:
            print("Displaying The Clinic Schedule..")        
        
        for day in self.days:
            print(day)
            print("\n")

    def updateState(self):
        if verbose:
            print("Updating the Schedule States..")
            
        states = []
        
        for day in self.days:
            for slot in day.time_slots:
                states.append(slot.state)
                
        self.states = states
        
    def getState(self, predicted_consult_time, predictedIsNoShow = False, normalize = False):
        '''
            Returns a 30 element long list with the following details:
                1. isNoShow     (predicted)
                2. consult_time (predicted)
                3. for Each day (7 days):
                    - Number of deciles where patients will show up
                    - Number of deciles where patient will not show up
                    - Number of overtime deciles 
                    - Max. consecutive free deciles 
        '''
        if verbose:
            print("Getting the Schedule States..")

        state = []
        
        state.append(1 if predictedIsNoShow else 0)
        state.append(predicted_consult_time)
        
        dayStates = []
        
        for dayIdx in range(len(self.days)): 
            show_count, no_show_count, overtime_count, free_count = self.getDayState(dayIdx)
#             dayStates.append(show_count)
            dayStates.append(no_show_count)
            dayStates.append(overtime_count)
            dayStates.append(free_count)
            
        def normalize(rawList):   
            rawList = [float(i) for i in rawList]
            maxx    = max(rawList)
            minn    = min(rawList)

            return [round((x - minn)/(maxx - minn), 1) for x in rawList]
    
        if normalize:
            state.extend(normalize(dayStates))
        else:
            state.extend(dayStates)
            
        return state
            
    def isValidDayForSlot(self, day, consult_mins):
        if verbose:
            print("Checking Validity of time slot- day: {0}, consult_mins:{1}".format(day, consult_mins))
        
        if day<0 or day>6:
            if(verbose==True):
                print("[Time-slot invalid] day is out of bounds, Day: {0}, consult time:{1}".format(day, consult_mins))
            return False
        
        if consult_mins not in self.consultLookup.values():
            if verbose:
                print("[Time-slot invalid] consult_mins is out of bounds, Day: {0}, consult time:{1}".format(day, consult_mins))
            return False
                
        if not self.isTimeSlotAvailable(day, consult_mins):
            if verbose:
                print("[Time-slot no Available] Day: {0}, consult time:{1}".format(day, consult_mins))
            return False
        
        return True
    
    def isFull(self, consult_mins):
        '''
            The schedule is full if the patient cannot be scheduled on any of the days
        '''
        if verbose:
            print("Checking if the schedule is full..")
        
        for each_day in range(0, len(self.days)):
            isScheduleFull = not self.isTimeSlotAvailable(each_day, consult_mins)
            
            if verbose:
                print("[isScheduleFull] " + str(isScheduleFull))
            
            return isScheduleFull
        
    def isTimeSlotAvailable(self, day, slot_duration):            
        if verbose:
            print("[Reward] Checking if Time-slot is free for day: {0}, slot_duration:{1}".format(day, slot_duration))

        time_slots_for_the_day    = iter(self.days[day].time_slots)
        free_slots_to_find        = int(slot_duration)/10
        consecutive_free_slot_cnt = 0

        # Detect if 'n' consucutive free slots exist in the day's schedule
        for _ts in range(len(self.days[day].time_slots)):
            time_slot = time_slots_for_the_day.next()

            if time_slot.state == 0 and int(slot_duration) == 10:
                return True

            if time_slot.state == 0 and int(slot_duration) > 10:
                consecutive_free_slot_cnt += 1

                if consecutive_free_slot_cnt >= int(slot_duration)/10:
                    if verbose:
                        print("[isTimeSlotAvailable] Day: {0} has {1} mins free".format(day, consecutive_free_slot_cnt * 10))
                    
                    return True

            else:
                consecutive_free_slot_cnt = 0
                
        return False
    
    def bookSlotForTheDay(self, day, personId, consult_time, isNoShow = False):
        if verbose:
            print("[Scheduler] Booking slot for day: {0}; consult_time: {1} mins".format(day, consult_time))
            
        # Check if the day has a free time-slot
        if self.isValidDayForSlot(day, consult_time):
            # Book the time-slot
            time_slots_for_the_day = iter(self.days[day].time_slots)
            slots_booked           = 0
            
            for _ts in range(len(self.days[day].time_slots)):
                time_slot = time_slots_for_the_day.next()
                                
                # Find the first free slot to start booking (assuming slots are always filled from top to bottom)
                if time_slot.state == 0 and (slots_booked * 10 < int(consult_time)):
                    # Book the current slot
                    time_slot.state    = 1
                    time_slot.personId = personId
                    time_slot.isNoShow = isNoShow
                    slots_booked       += 1
                    
                    if verbose:
                        print("[Scheduler] {0} min Slots Booked on day: {1}".format(consult_time, day))
        
            if slots_booked > 0:
                return True
            
        return False
    
    def displayAllDayState(self):
        for dayIdx in range(len(self.days)):
            if verbose:
                print("Day {0} state: {1}".format(dayIdx, self.getDayState(dayIdx)))
    
    def getDayState(self, day):    
        time_slots = self.days[day].time_slots
        
#         def minMaxNormalize(a,b,c,d):
            
            
        
        patient_show_up_decile_count = 0
        patient_no_show_decile_count = 0
        overtime_decile_count        = 0 
        max_consucutive_free_deciles = 0
        
        for time_slot in time_slots:
            if time_slot.state == 0:
                max_consucutive_free_deciles += 1
            else:
                
                if not time_slot.isNoShow:              # Count patients that will show up
                    patient_show_up_decile_count += 1
                else:                                   # Count no-shows
                    patient_no_show_decile_count += 1
                
                # Count overtime decile slots
                if (time_slot.hr + ":" + time_slot.mins) in self.overtime_hrs:
                    overtime_decile_count += 1
            
        return patient_show_up_decile_count, patient_no_show_decile_count, overtime_decile_count, max_consucutive_free_deciles
            
    def getExpectedProfitForTheDay(self, day):
        if verbose:
            print("[Schedluer] Getting the expected profit for the day:{0}".format(day))
            
        expected_revenue = 0
        overtime_cost    = 0
            
        show_count, no_show_count, overtime_count, free_count = self.getDayState(day)
        
        expected_revenue = expected_revenue + (100 * show_count)
        overtime_cost    = overtime_cost    + (200 * overtime_count)
        
        return expected_revenue - overtime_cost
        
    def getExpectedProfit(self):
        if verbose:
            print("[Scheduler] Calculating Expected profit..")
            
        expected_revenue = 0
        overtime_cost    = 0
        
        for dayIdx in range(len(self.days)):

            show_count, no_show_count, overtime_count, free_count = self.getDayState(dayIdx)
            
            expected_revenue = expected_revenue + (100 * show_count)
            overtime_cost    = overtime_cost    + (200 * overtime_count)
                
        return expected_revenue - overtime_cost
    
    def bookAllSlotsWithDummyData(self):
        if verbose == True:
            print("Filling all slots with dummy data..")
            
        for day in range(len(self.days)):
            for _ts in self.days[day].time_slots:
                self.bookSlotForTheDay(day, 'Test_Patient', '10', isNoShow = False)
 
    def getReward(self, action_day, consult_time, old_score, new_score):
        if verbose:
            print("[Rewards] Calculating..")
            
        # +100 if total expected profit for 7 days increases due to addition of patient 
        if int(new_score) >= int(old_score): 
            return 100
        
        # -100 if the total profit decreases for the 7 days 
        if int(new_score) < int(old_score): 
            return -100
        
        # -50  if a slot is predicted/alloted on Nth day even if a free slot exists on any of the prior days 
        def isTimeSlotAvailable(day, slot_duration):            
            if(verbose==True):
                print("[Reward] Checking if Time-slot is free for day: {0}, slot_duration:{1}".format(day, slot_duration))
                
            time_slots_for_the_day    = iter(self.days[day].time_slots)
            free_slots_to_find        = int(slot_duration)/10
            consecutive_free_slot_cnt = 0
            
            # Detect if 'n' consucutive free slots exist in the day's schedule
            for _ts in range(len(self.days[day].time_slots)):
                time_slot = time_slots_for_the_day.next()
                
                if time_slot.state == 0 and int(slot_duration) == 10:
                    return True
                
                if time_slot.state == 0 and int(slot_duration) > 10:
                    consecutive_free_slot_cnt += 1
                    
                    if consecutive_free_slot_cnt > int(slot_duration)/10:
                        if(verbose==True):
                            print("[Reward] -50 since day: {0} has {1} consecutive mins free".format(day, consecutive_free_slot_cnt * 10))
                        return True
                
                else:
                    consecutive_free_slot_cnt = 0
                            
            return False
        
#         if int(action_day) > 0:
#             for each_day in range(len(sch.days) - 1):
#                 isSlotAvailable = isTimeSlotAvailable(each_day, consult_time)

#                 if verbose == True:
#                     print("[isSlotAvailable] " + str(isSlotAvailable))

#                 if isSlotAvailable == True:
#                     return -50
    
    
    #####################################
        
class Patient:
    def __init__(self, personId, isNoShow, consult_time):
        self.personId     = personId
        self.isNoShow     = isNoShow
        self.consult_time = consult_time
        
class SchedulingAgent:
    def __init__(self, schedule):
        if verbose == True:
            print("[Agent] Initializing...")
            
        self.schedule = schedule
        
        from keras.models      import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers  import RMSprop, SGD

        model = Sequential()
#         model.add(Dense(438, init='lecun_uniform', input_shape=(30,)))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.7)) 

#         model.add(Dense(2154, init='lecun_uniform'))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.2))
        # Layer 1
        model.add(Dense(164, init='lecun_uniform', input_shape=(23,)))
        model.add(Activation('relu'))
#         model.add(Dropout(0.2)) 

#         # Layer 2
#         model.add(Dense(150, init='lecun_uniform'))
#         model.add(Activation('relu'))
# #         model.add(Dropout(0.2))
        
#         # Layer 3
#         model.add(Dense(164, init='lecun_uniform', input_shape=(30,)))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.8)) 

#         # Layer 4
#         model.add(Dense(164, init='lecun_uniform'))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.8))
        
#         # Layer 3
#         model.add(Dense(164, init='lecun_uniform', input_shape=(30,)))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.8)) 
        
#         # Layer 4
#         model.add(Dense(164, init='lecun_uniform'))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.8))
        
#         # Layer 3
#         model.add(Dense(164, init='lecun_uniform', input_shape=(30,)))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.8)) 

#         # Layer 4
#         model.add(Dense(150, init='lecun_uniform'))
#         model.add(Activation('relu'))
#         model.add(Dropout(0.8))

        model.add(Dense(7, init='lecun_uniform'))
        model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

        self.rms = RMSprop(lr=1e-3)
        self.sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=self.rms)
        
        self.model    = model
        
    def getRandomPatient(self):
        import random

        consultLookup = {0:'10', 1:'20', 2:'30', 3:'40', 4:'50', 5:'60'}
        
        personId      = 'Test Patient'
        isNoShow      = False
        consult_time  = consultLookup.get(random.choice(consultLookup.keys()))
        
        return Patient(personId, isNoShow, consult_time)
    
    def createNetworkInput(self, consultTime, isNoShow, schedule):   
        return np.asarray(schedule.getState(consultTime, isNoShow)).reshape(1, 23)
                
    def train(self, epochs = 1000, epsilon = 1.0, gamma = 0.9):
        if verbose == True:
            print("[Agent] Training agent; epochs = {0}".format(str(epochs)))
            
        from IPython.display import clear_output
        import random
        
        schedule = self.schedule
        
        explorationCount  = 0
        exploitationCount = 0
        patient_counter   = 0
        
        for cur_epoch in range(epochs):
            # Init state
            schedule.reset()
            randomPatient          = self.getRandomPatient()
            randomPatient.personId = "Patient#_" + str(patient_counter + 1)
            state                  = self.createNetworkInput(randomPatient.consult_time, randomPatient.isNoShow, schedule)
            episodeInProgress      = True 
                        
            while(episodeInProgress):                      
                # Forward pass to get rewards for all actions given S from our Q network
                qval = self.model.predict(state, batch_size=1)
                    
                # Choose an action based on epsilon to strike a balance between exploration & exploitation
                if (random.random() < epsilon): 
                    action            = np.random.randint(0, len(schedule.days)) # Choose random action
                    explorationCount += 1
                    if verbose == True:
                        print("[Agent] Random action chosen: " + str(action))
                else:
                    action             = (np.argmax(qval))       # Choose best action from Q(s,a) values
                    exploitationCount += 1
                    
                    if verbose == True:
                        print("[Agent] Best action from Q(s,a) chosen: " + str(action))
                                        
                # Get old expected profit
                old_score = schedule.getExpectedProfit()
                    
                # Take action, observe new state S'                
                isSlotBooked = schedule.bookSlotForTheDay(action, randomPatient.personId, randomPatient.consult_time, randomPatient.isNoShow)
                
                new_state = self.createNetworkInput(randomPatient.consult_time, randomPatient.isNoShow, schedule)
                
                # print state (for debugging)
#                 print(schedule.displayAllDayState())
#                 print(schedule.display())

                # Get new expected profit
                new_score = schedule.getExpectedProfit()
                
                # Observe reward
                reward = schedule.getReward(action, randomPatient.consult_time, old_score, new_score)
                
                #Get max_Q(S',a)
                newQ = self.model.predict(new_state, batch_size=1)
                maxQ = np.max(newQ)

                y = np.zeros((1, len(schedule.days)))
                y[:] = qval[:]  # Array broadcasting
                
                isEndOfEpisode = schedule.isFull(randomPatient.consult_time) or not isSlotBooked
                
                # Compute the parameter update for the Neural-Network function approximator Q(s,a)
                if not isEndOfEpisode: #non-terminal state
                    update = (reward + (gamma * maxQ))
                    print("End of Episode Update: " + str(update))
                else:                #terminal state
                    update = reward
                    print("In-progress Episode Update: " + str(update))

                y[0][action] = update #target output
            
                print
                print("************** EPOCH#: %s ***************" % (cur_epoch,))
                
                self.model.fit(state, y, batch_size=1, nb_epoch=1, verbose=1)
                state = new_state
                
                if isEndOfEpisode:
                    episodeInProgress = not isEndOfEpisode
                    
                clear_output(wait=True)
                                    
            # Decay epsilon to strike a balance between exploration v/s exploitation
            if epsilon > 0.1:
                epsilon -= (float(1)/float(epochs))

        print("Exploration rate  = " +str(float(explorationCount)/float(epochs)))
        print("Exploitation rate = " +str(float(exploitationCount)/float(epochs)))
        return schedule

    def trainWithExperienceReplay(self, epochs = 3000, epsilon = 1.0, gamma = 0.975, batchSize=1000):
        self.model.compile(loss='mse', optimizer=self.rms)     #reset weights of neural network
        
        if verbose == True:
            print("[Agent] Training agent with Experience Replay; epochs = {0}".format(str(epochs)))
            
        from IPython.display import clear_output
        import random
        
        schedule = self.schedule  
        
        explorationCount  = 0
        exploitationCount = 0

        buffer            = 2 * batchSize
        replay            = []             #stores tuples of (S, A, R, S')
        h                 = 0

        episode_num       = 0
        
        for cur_epoch in range(epochs):
            # Init state
            schedule.reset()
            randomPatient          = self.getRandomPatient()
            randomPatient.personId = "Patient#_" + str(cur_epoch)
            state                  = self.createNetworkInput(randomPatient.consult_time, randomPatient.isNoShow, schedule)

            scheduleFull           = False 
            episodeInProgress      = True 
                     
            while(episodeInProgress):  
                isEndOfEpisode = False
                
                # Forward pass to get rewards for all actions given S from our Q network
                qval = self.model.predict(state, batch_size=1)
                
                # Choose an action based on epsilon to strike a balance between exploration & exploitation
                if (random.random() < epsilon): 
                    action            = np.random.randint(0,7) # Choose random action
                    explorationCount += 1
                    if verbose == True:
                        print("[Agent] Random action chosen: " + str(action))
                else:
                    action             = (np.argmax(qval))       # Choose best action from Q(s,a) values
                    exploitationCount += 1
                    
                    if verbose == True:
                        print("[Agent] Best action from Q(s,a) chosen: " + str(action))
                    
                # Get old expected profit
                old_score = schedule.getExpectedProfit()
                
                # Take action, observe new state S'                
                isSlotBooked = schedule.bookSlotForTheDay(action, randomPatient.personId, randomPatient.consult_time, randomPatient.isNoShow)
                
                new_state = self.createNetworkInput(randomPatient.consult_time, randomPatient.isNoShow, schedule)
                
                # Get new expected profit
                new_score = schedule.getExpectedProfit()
                
                # Observe reward
                reward = schedule.getReward(action, randomPatient.consult_time, old_score, new_score)

                # Experience Replay Stage
                if (len(replay) < buffer):  #if buffer not filled, add to it
                    replay.append((state, action, reward, new_state))
                else: 
                    # If buffer is full, overwrite old values
                    if (h < (buffer-1)):
                        h += 1
                    else:
                        h = 0
                        
                    replay[h] = (state, action, reward, new_state)
                    
                    # Randomly sample our experience replay memory
                    minibatch = random.sample(replay, batchSize)
                    X_train = []
                    y_train = []
                    
                    for memory in minibatch:
                        
                        # Get max_Q(S',a)
                        old_state, action, reward, new_state = memory
                        
                        old_qval = self.model.predict(old_state, batch_size=1)
                        
                        newQ     = self.model.predict(new_state, batch_size=1)
                        maxQ     = np.max(newQ)
                        
                        y        = np.zeros((1,7))
                        y[:]     = old_qval[:]
                        
                        isEndOfEpisode = schedule.isFull(randomPatient.consult_time) or not isSlotBooked

                        # Compute the parameter update for the Neural-Network function approximator Q(s,a)
                        if not isEndOfEpisode: #non-terminal state
                            update = (reward + (gamma * maxQ))
#                             print("End of Episode Update: " + str(update))
                        else:                #terminal state
                            update = reward
#                             print("In-progress Episode Update: " + str(update))
                    
                        y[0][action] = update # target output
                        
                        X_train.append(old_state.reshape(23,))
                        y_train.append(y.reshape(7,))

                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    
                    print
                    print("************** EPOCH#: %s ***************" % (cur_epoch + 1,))
                
                    self.model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=1)
                    state = new_state
                                    
                if isEndOfEpisode:
                    episodeInProgress = not isEndOfEpisode
                    
                clear_output(wait=True)
                                    
            # Decay epsilon to strike a balance between exploration v/s exploitation
            if epsilon > 0.1:
                epsilon -= (float(1)/float(epochs))
        
        return schedule
		
		
# How to execute
# sch = Schedule()
# sa = SchedulingAgent(sch)
# sa.train(epochs = 10)
# sa.trainWithExperienceReplay(epochs = 100)
# sch.display()
                    