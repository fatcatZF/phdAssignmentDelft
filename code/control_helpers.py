
import os
import sys
import math

path_tools = os.path.join(os.environ.get("SUMO_HOME"), 'tools')
if path_tools in sys.path:
    #print("Found in home")
    pass
else:
    sys.path.append(path_tools)

import traci

def get_control(density, previous_target_flow, time_val, control_alg, verbose=False):

    # Call passed alinea function
    target_flow, log_prob, state = control_alg(density, previous_target_flow)

    # Calc prop
    known_sat = 1800
    green_time_prop = target_flow/known_sat
    trigger = True
    critical_density = 35

        
    #if density < 0.25 * critical_density :
    #    trigger = False

    # Evaluation will take place every 60 seconds and new cycle will be setup.
    cycle_no = math.ceil(30 * green_time_prop)
    
    # Cycle no is times yellow and times green (fixed 1 sec). 
    r_time = int((60 - cycle_no - cycle_no)/(cycle_no+1e-6))
    # Set min r_time
    if r_time < 1:
        r_time = 1
    if verbose:
        print(f"------------------------")
        print(f"Time: {time_val}")
        print(f"    Control Algorithm is active: {trigger}.") 
        print(f"    The number of green cycles in 60s: {cycle_no}")
        print(f"    The target flow is set to {target_flow}.")
    

    
    return r_time, trigger, target_flow, log_prob, state
    
def control_run(control_function, verbose):
    # Init var
    step = 0
    start_cycle = 0
    r_time = 0
    current_r_time = 0
    trigger = False
    veh_count_up = 0
    veh_count_down = 0
    speed_up_avg = 0
    speed_down_avg = 0
    previous_target_flow = 1800
    speed_mul_counts = []
    saved_log_probs = [] #store actions when training with deep Q learning
    states = []
    densities = []
    sumoCMD = ["sumo", "-n", "files_e3/onramp_network.net.xml", "-r", "files_e3/gen_ab.xml,files_e3/gen_ob.xml", "-a", "files_e3/output_req.xml,files_e3/add_loop.xml", "--fcd-output", "files_e3/output/fcd_out.xml",  "--lateral-resolution", "2.5"]

    # Start Simulation
    traci.start(sumoCMD)
    occ = [[],[]]
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        time_val = traci.simulation.getTime()

        # Get lane data
        speed_up = traci.inductionloop.getLastStepMeanSpeed("e1")
        speed_down = traci.inductionloop.getLastStepMeanSpeed("e0")
        v_number_up = traci.inductionloop.getLastStepVehicleNumber("e1")
        v_number_down = traci.inductionloop.getLastStepVehicleNumber("e0")
        veh_count_up = veh_count_up + v_number_up
        veh_count_down = veh_count_down + v_number_down
        
        if speed_up >= 0:
            speed_up_avg = (speed_up_avg + speed_up) / 2
        
        if speed_down >= 0:
            speed_down_avg = (speed_down_avg + speed_down) / 2

        # Get light data
        light_state = traci.trafficlight.getRedYellowGreenState("J2")
        # check new cycle
        

        if time_val % 60 == 0:
            start_cycle = time_val

            avg_speed = sum([speed_up_avg, speed_down_avg]) * 3.6 / 2
            avg_veh_count_hours = sum([veh_count_up * 60, veh_count_down * 60]) / 2
            density = avg_veh_count_hours / avg_speed
            avg_speed_mul_count = avg_speed*(sum([veh_count_up, veh_count_down]) / 2)
            speed_mul_counts.append(avg_speed_mul_count) #reward of the last action
            r_time, trigger, previous_target_flow, log_prob, state = get_control(density, previous_target_flow, time_val, control_function, verbose)
                                                                         
            saved_log_probs.append(log_prob)
            states.append(state)
            densities.append(density)

            if verbose:
                print("Density: {:.4f}".format(density))
                print("Average speed: {:.4f}".format(avg_speed))
                print("Average vehicle per hour: {:.4f}".format(avg_veh_count_hours))
                print("Average speed mul count: {:.4f}".format(avg_speed_mul_count))
            
            # Init for next round
            veh_count_up = 0
            veh_count_down = 0
        
        # Check if alinea activated and control pattern needs to be used.
        if trigger:
            if light_state == 'G':
                traci.trafficlight.setRedYellowGreenState("J2", "y")
            elif light_state == 'y':
                traci.trafficlight.setRedYellowGreenState("J2", "r")
                # Increase red time count
                current_r_time += 1
            elif (light_state == 'r') & (current_r_time >= r_time):
                # if exceeded r_time set back to green and reset variable
                traci.trafficlight.setRedYellowGreenState("J2", "G")
                current_r_time = 0
            elif light_state == 'r':
                current_r_time += 1
        
        # If alinea deactivated set light to green
        else:
            if light_state != 'G':
                traci.trafficlight.setRedYellowGreenState("J2", "G")

        step += 1

    traci.close()
    return densities, saved_log_probs, states
    

def run(control_function, verbose=False):    
    densities, saved_log_probs, states = control_run(control_function, verbose)
    return densities, saved_log_probs, states
