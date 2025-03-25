# Figure - Simulation of batch reactor with yeast
#          with functions added to facilitate explorative simulation work
#
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2022-11-21 - Constraint-based approach
# 2022-11-24 - FMU-explore 0.9.6a now with more option for simu() when doing evaluation etc - should simplify?
# 2023-02-24 - Consolidate FMU-explore to 0.9.6 and means parCheck and par() udpate and simu() with opts as arg
# 2023-05-26 - Added plotType TimeSeries2 for the Modelica paper
# 2023-05-29 - Update to FMU-explore 0.9.7 
# 2023-05-31 - Adjustments for simplifications of the model and included qO2 for logging as well
# 2023-05-31 - Adjusted for FMPy
# 2023-05-31 - Quick fix for OM FMU wtih small negative ethanol conc
# 2023-05-31 - Adjusted to from importlib.meetadata import version
# 2023-09-12 - Updated to FMU-explore 0.9.8 and introduced process diagram
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as img
import zipfile  

from fmpy import simulate_fmu
from fmpy import read_model_description
import fmpy as fmpy

from itertools import cycle
from importlib.metadata import version  

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
global fmu_model, model
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   flag_vendor = 'JM'
   flag_type = 'CS'
   fmu_model ='BPL_YEAST_COB_Batch_windows_jm_cs.fmu'        
   model_description = read_model_description(fmu_model)  
elif platform.system() == 'Linux':
#   flag_vendor = input('Linux - run FMU from JModelica (JM) or OpenModelica (OM)?')  
#   flag_type = input('Linux - run FMU-CS (CS) or ME (ME)?')  
#   print()   
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-comiled OpenModelica 1.21.0') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_YEAST_COB.Batch_linux_om_cs.fmu'    
         model = load_fmu(fmu_model, log_level=0) 
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_YEAST_COB.Batch_linux_om_me.fmu'    
      model_description = read_model_description(fmu_model) 
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = {'NCP': 500}
   opts_fast = {'NCP': 100}
elif flag_type in ['ME', 'me']:
   opts_std = {'NCP': 500}
   opts_fast = {'NCP': 100}
else:    
   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   constants = [v for v in model_description.modelVariables if v.causality == 'local'] 
   MSL_usage = [x[1] for x in [(constants[k].name, constants[k].start) for k in range(len(constants))] if 'MSL.usage' in x[0]][0]   
   MSL_version = [x[1] for x in [(constants[k].name, constants[k].start) for k in range(len(constants))] if 'MSL.version' in x[0]][0]
   BPL_version = [x[1] for x in [(constants[k].name, constants[k].start) for k in range(len(constants))] if 'BPL.version' in x[0]][0] 
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '3.2.3 - used components: RealInput, RealOutput' 
   MSL_version = '3.2.3'
   BPL_version = 'Bioprocess Library version 2.1.1' 
else:    
   print('There is no FMU for this platform')


# Simulation time
global simulationTime; simulationTime = 12.0
global prevFinalTime; prevFinalTime = 0

# Provide process diagram on disk
fmu_process_diagram ='BPL_GUI_YEAST_COB_Batch_process_diagram_om.png'

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture']

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateDict, parDict, diagrams, newplot(), describe()
#------------------------------------------------------------------------------------------------------------------
   
# Create stateDict that later will be used to store final state and used for initialization in 'cont':
global stateDict; stateDict =  {}
stateDict = {variable.derivative.name:None for variable in model_description.modelVariables \
                                            if variable.derivative is not None}
stateDict.update(timeDiscreteStates) 

global stateDictInitial; stateDictInitial = {}
for key in stateDict.keys():
    if not key[-1] == ']':
         if key[-3:] == 'I.y':
            stateDictInitial[key] = key[:-10]+'I_0'
         elif key[-3:] == 'D.x':
            stateDictInitial[key] = key[:-10]+'D_0'
         else:
            stateDictInitial[key] = key+'_0'
    elif key[-3] == '[':
        stateDictInitial[key] = key[:-3]+'_0'+key[-3:]
    elif key[-4] == '[':
        stateDictInitial[key] = key[:-4]+'_0'+key[-4:]
    elif key[-5] == '[':
        stateDictInitial[key] = key[:-5]+'_0'+key[-5:] 
    else:
        print('The state vector has more than 1000 states')
        break

global stateDictInitialLoc; stateDictInitialLoc = {}
for value in stateDictInitial.values():
    stateDictInitialLoc[value] = value

# Create dictionaries parDict and parLocation
global parDict; parDict = {}
parDict['V_0'] = 4.5
parDict['VX_0'] = 1.0
parDict['VG_0'] = 10.0
parDict['VE_0'] = 0.0

parDict['mum'] = 0
parDict['qGr'] = 0
parDict['qEr'] = 0
parDict['qO2'] = 0

global parLocation; parLocation = {}
parLocation['V_0'] = 'bioreactor.V_0'
parLocation['VX_0'] = 'bioreactor.m_0[1]'
parLocation['VG_0'] = 'bioreactor.m_0[2]'
parLocation['VE_0'] = 'bioreactor.m_0[3]'

parLocation['mum'] = 'bioreactor.culture.mum'
parLocation['qGr'] = 'bioreactor.culture.qGr'
parLocation['qEr'] = 'bioreactor.culture.qEr'
parLocation['qO2'] = 'bioreactor.culture.qO2'

# Extra only for describe()
global key_variables; key_variables = []
parLocation['mu'] = 'bioreactor.culture.mu'; key_variables.append(parLocation['mu'])

# Parameter value check - especially for hysteresis to avoid runtime error
global parCheck; parCheck = []
parCheck.append("parDict['V_0'] > 0")
parCheck.append("parDict['VX_0'] >= 0")
parCheck.append("parDict['VG_0'] >= 0")

# Create list of diagrams to be plotted by simu()
global diagrams
diagrams = []

# Define standard plots
def newplot(title='Batch cultivation', plotType='TimeSeries'):
   """ Standard plot window 
       title = '' """
    
   # Reset pens
   setLines()
   
   # Subplots
   global ax1, ax2, ax3, ax11, ax12, ax21, ax22

   # Plot diagram 
   if plotType == 'TimeSeries':
         
      # Transfer of argument to global variable       
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('G [g/L]')
    
      ax2.grid()
      ax2.set_ylabel('E [1/h]')
      
      ax3.grid()
      ax3.set_ylabel('X [g/L]')      
      ax3.set_xlabel('Time [h]') 
      
      diagrams.clear()
      diagrams.append("ax1.plot(sim_res['time'],sim_res['bioreactor.c[2]'], label='G', color='b', linestyle=linetype)") 
      diagrams.append("ax2.plot(sim_res['time'],sim_res['bioreactor.c[3]'], label='E', color='b', linestyle=linetype)") 
      diagrams.append("ax3.plot(sim_res['time'],sim_res['bioreactor.c[1]'], label='X', color='b', linestyle=linetype)") 
      
      
   elif plotType == 'TimeSeries2':
         
      # Transfer of argument to global variable       
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('X [g/L]')
    
      ax2.grid()
      ax2.set_ylabel('mu [1/h]')
      
      ax3.grid()
      ax3.set_ylabel('G, E [g/L]')      
      ax3.set_xlabel('Time [h]') 
      
      diagrams.clear()
      diagrams.append("ax1.plot(sim_res['time'],sim_res['bioreactor.c[1]'], label='X', color='r', linestyle=linetype)") 
      diagrams.append("ax2.plot(sim_res['time'],sim_res['bioreactor.culture.mu'], label='mu', color='r', linestyle=linetype)")     
      diagrams.append("ax3.plot(sim_res['time'],sim_res['bioreactor.c[2]'], label='G', color='b', linestyle=linetype)") 
      diagrams.append("ax3.plot(sim_res['time'],sim_res['bioreactor.c[3]'], label='E', color='g', linestyle=linetype)") 
      diagrams.append("ax3.legend(['G','E'])")
 
   elif plotType == 'Extended':
         
      # Transfer of argument to global variable       
      plt.figure()
      ax11 = plt.subplot(2,2,1)
      ax12 = plt.subplot(2,2,2)
      ax21 = plt.subplot(2,2,3)
      ax22 = plt.subplot(2,2,4)
    
      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('G and E [g/L]')
    
      ax21.grid()
      ax21.set_ylabel('X [g/L]')
      ax21.set_xlabel('Time [h]')
      
      ax12.grid()
      ax12.set_ylabel('mu [1/h]')  
      
      ax22.grid()
      ax22.set_ylabel('qO2 [mole/(g*h)]')  
      ax22.set_xlabel('Time [h]')     
      
      diagrams.clear()
      diagrams.append("ax11.plot(sim_res['time'],sim_res['bioreactor.c[2]'], label='G', color='b', linestyle='-')") 
      diagrams.append("ax11.plot(sim_res['time'],sim_res['bioreactor.c[3]'], label='E', color='r', linestyle='-')") 
      diagrams.append("ax21.plot(sim_res['time'],sim_res['bioreactor.c[1]'], label='X', color='b', linestyle='-')")       
      diagrams.append("ax12.plot(sim_res['time'],sim_res['bioreactor.culture.mu'], label='mu', color='b', linestyle='-')")
      
      diagrams.append("ax22.plot(sim_res['time'],sim_res['bioreactor.culture.qO2'], 'b-')")
      
   elif diagram == 'PhasePlane':
      plt.figure()
      ax1 = plt.subplot(1,1,1)
                
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('G')
      ax1.set_xlabel('X')
                   
   else:
      print("Plot window type not correct") 

# Define and extend describe for the current application
def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""
        
   if name == 'culture':
      print('Saccharomyces cerevisae - default parameters for strain H1022')        
     
   elif name in ['broth', 'liquidphase', 'media']:
      X = model.get('liquidphase.X')[0]; 
      X_description = model.get_variable_description('liquidphase.X'); 
      X_mw = model.get('liquidphase.mw[1]')[0]
      
      G = model.get('liquidphase.G')[0]; 
      G_description = model.get_variable_description('liquidphase.G'); 
      G_mw = model.get('liquidphase.mw[2]')[0]
      
      E = model.get('liquidphase.E')[0]; 
      E_description = model.get_variable_description('liquidphase.E'); 
      E_mw = model.get('liquidphase.mw[3]')[0]

      print('Reactor broth substances included in the model')
      print()
      print(X_description, '  index = ', X, 'molecular weight = ', X_mw, 'Da')
      print(G_description, 'index = ', G, 'molecular weight = ', G_mw, 'Da')
      print(E_description, 'index = ', E, 'molecular weight = ', E_mw, 'Da')
     
   elif name in ['parts']:
      describe_parts(component_list_minimum)

   elif name in ['MSL']:
      describe_MSL()

   else:
      describe_general(name, decimals)
      
#------------------------------------------------------------------------------------------------------------------
#  General code 
FMU_explore = 'FMU-explore for FMPy version 0.9.8'
#------------------------------------------------------------------------------------------------------------------

# Define function par() for parameter update
def par(parDict=parDict, parCheck=parCheck, parLocation=parLocation, *x, **x_kwarg):
   """ Set parameter values if available in the predefined dictionaryt parDict. """
   x_kwarg.update(*x)
   x_temp = {}
   for key in x_kwarg.keys():
      if key in parDict.keys():
         x_temp.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an accessible parameter - check the spelling')
   parDict.update(x_temp)
   
   parErrors = [requirement for requirement in parCheck if not(eval(requirement))]
   if not parErrors == []:
      print('Error - the following requirements do not hold:')
      for index, item in enumerate(parErrors): print(item)

# Define function init() for initial values update
def init(parDict=parDict, *x, **x_kwarg):
   """ Set initial values and the name should contain string '_0' to be accepted.
       The function can handle general parameter string location names if entered as a dictionary. """
   x_kwarg.update(*x)
   x_init={}
   for key in x_kwarg.keys():
      if '_0' in key: 
         x_init.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an initial value, use par() instead - check the spelling')
   parDict.update(x_init)

# Define fuctions similar to pyfmi model.get(), model.get_variable_descirption(), model.get_variable_unit()
def model_get(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get() but returns just a value and not a list"""
   par_var = model_description.modelVariables
   for k in range(len(par_var)):
      if par_var[k].name == parLoc:
         try:
            if par_var[k].name in start_values.keys():
                  value = start_values[par_var[k].name]
            elif par_var[k].variability in ['constant']:        
                  value = float(par_var[k].start)                          
            elif par_var[k].variability in ['fixed', 'continuous']:
               try:
                  value = sim_res[par_var[k].name][-1]
               except (AttributeError, ValueError):
                  value = None
                  print('Variable not logged')
            else:
               value = None
         except NameError:
            print('Error: Information available after first simulation')
            value = None
   return value

def model_get_variable_description(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get_variable_description() but returns just a value and not a list"""
   par_var = model_description.modelVariables
#   value = [x[1] for x in [(par_var[k].name, par_var[k].description) for k in range(len(par_var))] if parLoc in x[0]]
   value = [x.description for x in par_var if parLoc in x.name]   
   return value[0]
   
def model_get_variable_unit(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get_variable_unit() but returns just a value and not a list"""
   par_var = model_description.modelVariables
#   value = [x[1] for x in [(par_var[k].name, par_var[k].unit) for k in range(len(par_var))] if parLoc in x[0]]
   value = [x.unit for x in par_var if parLoc in x.name]
   return value[0]
      
# Define function disp() for display of initial values and parameters
def disp(name='', decimals=3, mode='short'):
   """ Display intial values and parameters in the model that include "name" and is in parLocation list.
       Note, it does not take the value from the dictionary par but from the model. """
   
   def dict_reverser(d):
      seen = set()
      return {v: k for k, v in d.items() if v not in seen or seen.add(v)}
   
   if mode in ['short']:
      k = 0
      for Location in [parLocation[k] for k in parDict.keys()]:
         if name in Location:
            if type(model_get(Location)) != np.bool_:
               print(dict_reverser(parLocation)[Location] , ':', np.round(model_get(Location),decimals))
            else:
               print(dict_reverser(parLocation)[Location] , ':', model_get(Location))               
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model_get(Location)) != np.bool_:
                  print(parName,':', np.round(model_get(parLocation[parName]),decimals))
               else: 
                  print(parName,':', model_get(parLocation[parName])[0])

   if mode in ['long','location']:
      k = 0
      for Location in [parLocation[k] for k in parDict.keys()]:
         if name in Location:
            if type(model_get(Location)) != np.bool_:       
               print(Location,':', dict_reverser(parLocation)[Location] , ':', np.round(model_get(Location),decimals))
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model_get(Location)) != np.bool_:
                  print(parLocation[parName], ':', dict_reverser(parLocation)[Location], ':', parName,':', 
                     np.round(model_get(parLocation[parName]),decimals))

# Line types
def setLines(lines=['-','--',':','-.']):
   """Set list of linetypes used in plots"""
   global linecycler
   linecycler = cycle(lines)

# Show plots from sim_res, just that
def show(diagrams=diagrams):
   """Show diagrams chosen by newplot()"""
   # Plot pen
   linetype = next(linecycler)    
   # Plot diagrams 
   for command in diagrams: eval(command)

# Define simulation
def simu(simulationTime=simulationTime, mode='Initial', options=opts_std, diagrams=diagrams):
   """Model loaded and given intial values and parameter before, and plot window also setup before."""   
   
   # Global variables
   global sim_res, prevFinalTime, stateDict, stateDictInitial, stateDictInitialLoc, start_values
   
   # Simulation flag
   simulationDone = False
   
   # Internal help function to extract variables to be stored
   def extract_variables(diagrams):
       output = []
       variables = [v for v in model_description.modelVariables if v.causality == 'local']
       for j in range(len(diagrams)):
           for k in range(len(variables)):
               if variables[k].name in diagrams[j]:
                   output.append(variables[k].name)
       return output

   # Run simulation
   if mode in ['Initial', 'initial', 'init']: 
      
      start_values = {parLocation[k]:parDict[k] for k in parDict.keys()}
      
      # Simulate
      sim_res = simulate_fmu(
         filename = fmu_model,
         validate = False,
         start_time = 0,
         stop_time = simulationTime,
         output_interval = simulationTime/options['NCP'],
         record_events = True,
         start_values = start_values,
         fmi_call_logger = None,
         output = list(set(extract_variables(diagrams) + list(stateDict.keys()) + key_variables))
      )
      
      simulationDone = True
      
   elif mode in ['Continued', 'continued', 'cont']:
      
      if prevFinalTime == 0: 
         print("Error: Simulation is first done with default mode = init'")
         
      else:         
         # Update parDictMod and create parLocationMod
         parDictRed = parDict.copy()
         parLocationRed = parLocation.copy()
         for key in parDict.keys():
            if parLocation[key] in stateDictInitial.values(): 
               del parDictRed[key]  
               del parLocationRed[key]
         parLocationMod = dict(list(parLocationRed.items()) + list(stateDictInitialLoc.items()))
   
         # Create parDictMod and parLocationMod
         parDictMod = dict(list(parDictRed.items()) + 
            [(stateDictInitial[key], stateDict[key]) for key in stateDict.keys()])      

         start_values = {parLocationMod[k]:parDictMod[k] for k in parDictMod.keys()}
  
         # Simulate
         sim_res = simulate_fmu(
            filename = fmu_model,
            validate = False,
            start_time = prevFinalTime,
            stop_time = prevFinalTime + simulationTime,
            output_interval = simulationTime/options['NCP'],
            record_events = True,
            start_values = start_values,
            fmi_call_logger = None,
            output = list(set(extract_variables(diagrams) + list(stateDict.keys()) + key_variables))
         )
      
         simulationDone = True
   else:
      
      print("Error: Simulation mode not correct")

   if simulationDone:
      
      # Plot diagrams from simulation
      linetype = next(linecycler)    
      for command in diagrams: eval(command)
   
      # Store final state values in stateDict:        
      for key in stateDict.keys(): stateDict[key] = max(model_get(key), 0.0)  # Quick fix for OM FMU  
         
      # Store time from where simulation will start next time
      prevFinalTime = sim_res['time'][-1]
      
   else:
      print('Error: No simulation done')
            
# Describe model parts of the combined system
def describe_parts(component_list=[]):
   """List all parts of the model""" 
       
   def model_component(variable_name):
      i = 0
      name = ''
      finished = False
      if not variable_name[0] == '_':
         while not finished:
            name = name + variable_name[i]
            if i == len(variable_name)-1:
                finished = True 
            elif variable_name[i+1] in ['.', '(']: 
                finished = True
            else: 
                i=i+1
      if name in ['der', 'temp_1', 'temp_2', 'temp_3', 'temp_4', 'temp_5', 'temp_6', 'temp_7']: name = ''
      return name
    
#   variables = list(model.get_model_variables().keys())
   variables = [v.name for v in model_description.modelVariables]
        
   for i in range(len(variables)):
      component = model_component(variables[i])
      if (component not in component_list) \
      & (component not in ['','BPL', 'Customer', 'today[1]', 'today[2]', 'today[3]', 'temp_2', 'temp_3']):
         component_list.append(component)
      
   print(sorted(component_list, key=str.casefold))

# Describe MSL   
def describe_MSL(flag_vendor=flag_vendor):
   """List MSL version and components used"""
   print('MSL:', MSL_usage)
 
# Describe parameters and variables in the Modelica code
def describe_general(name, decimals):
  
   if name == 'time':
      description = 'Time'
      unit = 'h'
      print(description,'[',unit,']')
      
   elif name in parLocation.keys():
      description = model_get_variable_description(parLocation[name])
      value = model_get(parLocation[name])
      try:
         unit = model_get_variable_unit(parLocation[name])
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)            
      else:
        print(description, ':', np.round(value, decimals), '[',unit,']')
                  
   else:
      description = model_get_variable_description(name)
      value = model_get(name)
      try:
         unit = model_get_variable_unit(name)
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)     
      else:
         print(description, ':', np.round(value, decimals), '[',unit,']')

# Plot process diagram
def process_diagram(fmu_model=fmu_model, fmu_process_diagram=fmu_process_diagram):   
   try:
       processDiagram = zipfile.ZipFile(fmu_model, 'r').open('documentation/processDiagram.png')
   except KeyError:
       print('No processDiagram.png file in the FMU, but try the file on disk.')
       processDiagram = fmu_process_diagram
   try:
       plt.imshow(img.imread(processDiagram))
       plt.axis('off')
       plt.show()
   except FileNotFoundError:
       print('And no such file on disk either')
         
# Describe framework
def BPL_info():
   print()
   print('Model for the process has been setup. Key commands:')
   print(' - par()       - change of parameters and initial values')
   print(' - init()      - change initial values only')
   print(' - simu()      - simulate and plot')
   print(' - newplot()   - make a new plot')
   print(' - show()      - show plot from previous simulation')
   print(' - disp()      - display parameters and initial values from the last simulation')
   print(' - describe()  - describe culture, broth, parameters, variables with values/units')
   print()
   print('Note that both disp() and describe() takes values from the last simulation')
   print('and the command process_diagram() brings up the main configuration')
   print()
   print('Brief information about a command by help(), eg help(simu)') 
   print('Key system information is listed with the command system_info()')

def system_info():
   """Print system information"""
#   FMU_type = model.__class__.__name__
   constants = [v for v in model_description.modelVariables if v.causality == 'local']
   
   print()
   print('System information')
   print(' -OS:', platform.system())
   print(' -Python:', platform.python_version())
   try:
       scipy_ver = scipy.__version__
       print(' -Scipy:',scipy_ver)
   except NameError:
       print(' -Scipy: not installed in the notebook')
   print(' -FMPy:', version('fmpy'))
   print(' -FMU by:', read_model_description(fmu_model).generationTool)
   print(' -FMI:', read_model_description(fmu_model).fmiVersion)
   if model_description.modelExchange is None:
      print(' -Type: CS')
   else:
      print(' -Type: ME')
   print(' -Name:', read_model_description(fmu_model).modelName)
   print(' -Generated:', read_model_description(fmu_model).generationDateAndTime)
   print(' -MSL:', MSL_version)    
   print(' -Description:', BPL_version)   
   print(' -Interaction:', FMU_explore)
   
#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

BPL_info()