#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This version is in testing...It is not suitible for external use. 
This version seeks to develop the 'Phase Fraction' capability
"""

"""
This script allows the user to generate csv files containing ternary datasets, comprising of the phases present at a 
temperature(k) and elemental ratios (Mass%) ternary point. It can be used for metallic ternary systems, or for 
'psuedo-ternaries' (E.g. V-4Cr-4Ti;C;N). It is designed to be used in conjunction with a second script;
Anisothermal Plotter.

This code was written by Sam Yeates(The University of Sheffield) as part of his PhD, supervised by Prof. Russell Goodall 
and Prof. Katerina Christofidou. Assistance in ThermoCalc handling was given by Carl-Magnus Lancelot(ThermoCalc Software AB)  
This version of the script was finalised on 11/06/2025.

DISCLAIMER - This code does not carry out any CALPHAD or Thermodynamic modelling. It purely uses TC_Python to generate
phase data based on ThermoCalc calculations and databases. Therefore data outputted from this code is only ever as accurate
as ThermoCalc. Users should ensure they  validate any results to the accuracy/detail their work requires. 
This tool is designed first and foremost as a way to visualise systems to allow for first-pass alloy design or for 
scientific communication purposes. It is not designed to act as a precise or accurate design tool.

"""
import os
import numpy as np
from tc_python import *
import time
import concurrent.futures
import pandas as pd
Start = time.perf_counter()

#%% User Inputs - Only edit code in this section

"""This section is where you set up your Thermo-Calc license parameters. The below are set up by the author for MacOS,
    and are based on the standard install locations for TC and MacOS.
"""
os.environ['TC24A_HOME'] = '/Applications/Thermo-Calc-2025a.app/Contents/Resources'
os.environ['TC25A_HOME'] = '/Applications/Thermo-Calc-2025a.app/Contents/Resources'
os.environ['LSHOST'] = 'mse-lic1.shef.ac.uk'


"""This section is where you set up your (psuedo)ternary calculation. Open for set-up instructions

    First, select a suitible databse, and check which database is most suitible for your elemental range.
    WARNING - This code will make no checks for this.

    Second, create your temperature range and step size in Kelvin. In the authors experience, a Tres of 25k gives a good starting visual
    WARNING - ThermoCalc is anecdotatly poor below 600K, so be wary of setting your temp range below this.
    
    Third, set up your axis. Due to the way ThermoCalc works, there must always be a 'balance' element. This is to prevent the equation 
    becoming overdefined due to insufficient degrees of freedom. Depending on what you are wishing to investigate, there are two ways to 
    approach this.

    
    1) For Metallic True Ternary (E.g. V-Cr-Ti)
        Axis One defines your 'balance' element. 
        A1Es should take a list with a single str input of the desired element, e.g. A1Es = ["V"]
        A1EComps should take a list with a single value of 1.0, e.g. A1EComps = [1.0]
        
        Axis Two and Three are where you define your other two elements. You also define your compositional range and resolution here.
        For a typical Metallic True Ternary, you would expect the range to be 0-100wt% for each element, and an axis composition of 1.0
        Therefore 
        
        A2Es = ["Cr"]
        A2EComps = [1.0]
        A2min takes a float equal to the min wt%, e.g. A2min = 0.0
        A2max takes a float equal to the max. wt%, e.g. = 100.00
        A2res takes a float equal to the wt% step desired, e.g. A2res = 1.0 
            ;The author has recommend not going higher than 5wt%  for fast runs, or lower than 1wt%  for higher resolution.
            Should the user have signficant time, computational power, or need, this code can go to finer resolution.
    
        For this current software version, axis 2 and 3 must have matching ranges and resolutions. For a Metallic, this can either mean:
            
        A3Es = ["Ti"]
        A3EComps = [1.00] 
        A3min = 0.000
        A3max = 100.00
        A3res = 1.0
            
        or 
        
        A3Es = ["Ti"]
        A3EComps = A2EComps
        A3min = A2min
        A3max = A2max
        A3res = A2res
"""
"""

    2) For Psuedo Ternaries (E.g. V4Cr4Ti-CN-O)
        This approach is designed to allow analysis of impurity elements on metallic ternaries, or probe higher compositional spaces,
        such as commonly probed for HEAs/CCAs. However it requires care in strucutre as ThermoCalc must be fed the elements in the 
        correct way. 
        
        There are now several sets of compositions, one of each 'corner' of the ternary space. For example, if we take V4Cr4Ti-O-N we have the ratio of
            92wt% V : 4wt% Cr : 4wt% Ti
            50wt% C : 50wt% N
            100wt% O
        
        This refers to the idea of the metallic ternary V-4Cr-4Ti, with impurities where C and N are always equal, and O is considered seperately. 
        For the above statement to be true, the ratios shown above must be fixed. We do this using the AXEComps list. 
        We can then further vary the ammount of Axis Two and Axis Three's components through use of the AXmin,max and res, as we would for a metallic ternary. The
        simplest way to conceptualise this is that Axis One is your 'base' and you are adding in Axis Two and Three in varying ammounts. This is how this
        code is set up to run, by varying the concetration of Axis Two and Three Components, while holding the relative ratios of each axis constituent constant.
        
        WARNING - For this version of the script, Axis One and Two must vary in the same amount
        
        As an example in setting this up, let us consider the V4Cr4Ti-CN-O psuedo ternary, where we probe the 0-0.1wt% inclusion of CN and O, at a resolution
        of 0.001wt% (I.e. we will have 100 points of each axis). As above, our components are: 
        
            1) V-4Cr-4Ti
            2) C-N
            3) O
        at component ratios of:
            
            1) 92wt% V : 4wt% Cr : 4wt% Ti
            2) 50wt% C : 50wt% N
            3) 100wt% O            
        at axis ratios of:
            
            1) 99.8wt%-100wt%
            2) 0-0.1wt%
            3) 0-0.1wt%
        
        
        Axis One defines your 'balance' component. 
        A1Es should take a list of  strings of the desired elements, e.g. A1Es = ["V","Cr","Ti"]
        A1EComps should take a list of floats of the desired component mass fractions, e.g. A1EComps = [0.92,0.04,0.04]
        
        Axis Two and Three are where you define your other two components. You also define your axis range and resolution here.
        
        A2Es = ["C","N"]
        A2EComps = [0.5,0.5]
        A2min takes a float equal to the min wt%, e.g. A2min = 0.0
        A2max takes a float equal to the max. wt%, e.g. = 0.1
        A2res takes a float equal to the wt% step desired, e.g. A2res = 0.001 
             (The author has recommend not going less than 20 steps (5.0wt% for a 100wt range,0.05wt% for a 1wt% range,0.005wt% 
             for a 0.1wt% range)for fast runs, or more than 100 steps (1.0wt% for a 100wt range,0.01wt% for a 1wt% range,0.001wt% 
             for a 0.1wt% range)for higher resolution. Should the user have signficant time, computational power, or need, this 
             code can go to finer resolution.)
    
        For this current software version, axis 2 and 3 must have matching ranges and resolutions. For a Metallic, this can either mean:
            
        A3Es = ["O"]
        A3EComps = [1.00] 
        A3min = 0.000
        A3max = 0.100
        A3res = 0.001
        
        or 
        
        A3Es = ["O"]
        A3EComps = A2EComps
        A3min = A2min
        A3max = A2max
        A3res = A2res
"""

#Set Up the database
Database = "SSOL8"

#Define if you wish to produce Phase Fraction Data
GeneratePhaseFraction = True #SHould be true. If false will only return stable phases

#Temperature Inputs in K
Tmin = 1000
Tmax = 1400 
Tres = 10

#Pressue Inputs in Pa
Pressure = 100000 #Set to desired pressure. For default atomos, set to 100000


#Set up your axis
MassFraction = False #If true, calculaiton runs in Mass Faction compositions, if False, calculation runs in Mole Faction compositions
#Set up Axis 1
A1Es = ["Ta","V","Ti","W","Cr"]
A1EComps = [0.30,0.30,0.30,0.05,0.05] #Form of [Element,Element,Element,...]


#Set up Axis 2
A2Es = ["C"]
A2EComps = [1.00] #Handles psuedo ternary composition ratios. For Metallic Ternary, set to [1.0]
A2max = 0.02 #Defines upper wt% range of axis. For Metallic Ternaries, set to 100.0
A2res = 0.0002
#Set up Axis 3
A3Es = ["N"]
A3EComps = [1.00] #Handles psuedo ternary composition ratios. For Metallic Ternary, set to [1.0]
A3max =0.02 #For Metallic Ternaries, set to 100.0
A3res = 0.0002

#%% End of User Inputs - DO NOT EDIT CODE BEYOND THIS POINT

# Define the two min/max lists for axis 2 & 3 for filename later
A2min = 0.0
A3min = 0.0
Mins = [A2min, A3min]
Maxs = [A2max, A3max]

A2min = A2min/100.00
A2max = A2max/100.00
A2res = A2res/100.00

A3min = A3min/100.00
A3max = A3max/100.00
A3res = A3res/100.00


#Build Axis Definitions and Names
AxisDef = [A1Es,A2Es,A3Es]
AxisComps = [A1EComps, A2EComps, A3EComps]
AxisNames = []

if MassFraction == False:
    CT = "at%"
elif MassFraction:
    CT = "wt%"


ElementList = []
for Axis in AxisDef:
    for Element in Axis:
        if Element not in ElementList:
            ElementList.append(Element)


for Elements, Compositions in zip(AxisDef, AxisComps):
    if len(Elements) == 1:
        # single‐component axis - just the element symbol
        AxisNames.append(Elements[0])
    else:
        Parts = [Elements[0]]  \
            + [ f"{int(c*100.00)}{e}" 
                for e, c in zip(Elements[1:], Compositions[1:]) ]
        AxisNames.append("-".join(Parts))

# Define calculation as a function, because concurrent futures requires functions to parallellise
def RunIsothermCalculation(Parameters):
    #----  
    PhaseFractions = []
    PhaseCompositions = []
    ComponentFractions = []
    #----  
    Rows = []
    T = float(Parameters["T"])

    with TCPython() as start:
        BaseCalc = (
            start
            .set_cache_folder("./cache/")
            .select_database_and_elements(Database, ElementList)
            .get_system()
            .with_single_equilibrium_calculation()
            .enable_global_minimization()
            .set_condition(ThermodynamicQuantity.temperature(), T)
            .set_condition(ThermodynamicQuantity.pressure(), Pressure)
            )
        
        #results = {}
        for A2 in np.arange(A2min, A2max+A2res, A2res):
            for A3 in np.arange(A3min, A3max+A3res, A3res):
                A1 = 1.00000000-A2-A3
                if A1 >= 0:
                    
                    IsoCalc = BaseCalc
                    AxisRatio = [A1,A2,A3]
                    # now compute each element’s true fraction
                    TotalElementalComp = {}
                    for AxisFraction, Elements, Composition in zip(AxisRatio, AxisDef, AxisComps):
                        for Element, Composition in zip(Elements, Composition):
                            # if you want weight‐percent, multiply by 100 here
                            TotalElementalComp[Element] = round(AxisFraction * Composition, 12)
                   
                    if MassFraction:
                        for i, (Element, Fraction) in enumerate(TotalElementalComp.items()):
                            if i == 0:           # skip index 0
                                continue
                            Calc = (IsoCalc.
                                    set_condition(ThermodynamicQuantity.mass_fraction_of_a_component(Element),Fraction))
                            
                    elif MassFraction == False:
                        for i, (Element, Fraction) in enumerate(TotalElementalComp.items()):
                            if i == 0:           # skip index 0
                                continue
                            Calc = (IsoCalc.
                                    set_condition(ThermodynamicQuantity.mole_fraction_of_a_component(Element),Fraction))    
                    try:
                        
                        ResultObject = BaseCalc.calculate()
                        PointPhases = ResultObject.get_stable_phases()
                        
                        if GeneratePhaseFraction ==True:
                            for Phase in PointPhases:
                                PhaseFraction = ResultObject.get_value_of(f"NPM({Phase})")
                                PhaseFractions.append(PhaseFraction)
                                
                                for Element in ElementList:
                                    ComponentFraction = ResultObject.get_value_of(f"X({Phase},{Element})")
                                    if ComponentFraction < 0.0000:
                                        continue
                                    ComponentFractions.append(f"{Element}:{(ComponentFraction*100):0.2f}%")
                                    
                                PhaseCompositions.append("-".join(ComponentFractions))
                                ComponentFractions = []                        
                            
                 
                    except CalculationException:

                        print("\nThere were too many iterations to process")
                        PointPhases = "NaN"
                            

                            
                    #print(PhaseFractions)
                    #----
                    
                    Row = {
                        "Temp/K": T,}  
                    for Element, Fraction in TotalElementalComp.items():
                        Row[f"{Element}/Mass %"] = round(Fraction, 12)
                    # add the phases column
                    Row["Phases"] = "NaN" if PointPhases == "NaN" else ", ".join(PointPhases)
                    
                    #----  
                    if GeneratePhaseFraction ==True:
                            if PointPhases == "NaN":
                                Row["Phase Fractions"] = "NaN"
                                Row["Phase Compositions"] = "NaN"
                            else:
                                # convert each numeric fraction to a string first
                                Row["Phase Fractions"] = ", ".join(str(Fraction) for Fraction in PhaseFractions)
                                Row["Phase Compositions"] = ", ".join(PhaseCompositions)
                    #---- 
                    
                    Rows.append(Row)
                    PointPhases = []
                    PhaseFractions =[]
                    PhaseCompositions = []
                    
    return Rows



if __name__ == '__main__':


    Ts  = range(Tmin, Tmax+1, Tres)

    Parameters = [
        {'index': i, 'T': T}
        for i, T in enumerate(Ts)
    ]
    
    CPU_No = 10 #Number of cpu cores to use

    # This uses concurrent futures to map the different simulations onto multiple CPU cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=CPU_No) as Executor:
        results = Executor.map(RunIsothermCalculation,Parameters)
        
        """
        for res in zip(parameters, executor.map(run_isotherm_calculation, parameters)):
            params, calc_results = res
            # Now parameters and calculation results have been collected. We choose to create a results list
            # containing the temperature and the results for each simulation. This can be modified to instead
            # contain some other varying parameter
            results.append(calc_results)
        """


    # flatten the list of lists
    ResultsFlat = [row for sublist in results for row in sublist]

    # build  column list dynamically:
    #   first the temperature, then one column per element, then phases
    ElementColumn = [f"{Element}/Mass %" for Element in ElementList]
    if GeneratePhaseFraction == True:
        Columns  = ["Temp/K"] + ElementColumn + ["Phases"] + ["Phase Fractions"] + ["Phase Compositions"]
    else:
        Columns  = ["Temp/K"] + ElementColumn + ["Phases"]
        
    Results = pd.DataFrame(ResultsFlat, columns=Columns)
    print(Results)

    # Collect the three axis name‐parts, inserting ranges only when not 0–100
    NameParts = [AxisNames[0]]
    
    
    for i in (1, 2):
        label = AxisNames[i]
        Label = f"{label}"
        AxisMin, AxisMax = Mins[i-1], Maxs[i-1]
        # only include range if it isn't the full 0–100
        if not (AxisMin == 0 and AxisMax == 100):
            # strip any trailing zeros on floats, e.g. 0.100 -> 0.1
            StrippedAxisMin = str(AxisMin).rstrip('0').rstrip('.')  
            StrippedAxisMax = str(AxisMax).rstrip('0').rstrip('.')
            Label = f"{label}({StrippedAxisMin}-{StrippedAxisMax}{CT},{A2res})"
        NameParts.append(Label)

            
            
        
    
    #Add the temperature window
    Filename = "_".join(NameParts) + f"-{Tmin}K-{Tmax}K-{Tres}K step"
    
    if GeneratePhaseFraction == True:
        Filename = Filename + "_with_phase_fractions.csv"
    
    else:
        Filename = Filename + ".csv"
    print(Filename)
    
    AxisDefStr  = ";".join(",".join(Group) for Group in AxisDef)
    AxisNameStr = ";".join(AxisNames)
    
    if MassFraction:
        CompType = "Mass"
    elif MassFraction == False:
        CompType = "Atomic"
    
    
    with open(Filename, "w", newline="") as f:
        # front-matter must exactly match the loader’s keys:
        f.write(f"# axis_defs={AxisDefStr}\n")
        f.write(f"# axis_names={AxisNameStr}\n")
        f.write(f"# comp_type ={CompType}\n")
        
        # now write your real CSV (comma-separated by default)
        Results.to_csv(f, index=False)
    print(f"Wrote {Filename}")
    
    
    End = time.perf_counter()
    Time = End - Start
    print(f"Total runtime: {Time:.2f} seconds")
    
    
    
    
    
    
    