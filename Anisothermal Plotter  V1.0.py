#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script allows the user to input csv files containing ternary datasets, comprising of the phases present at a 
temperature(k) and elemental ratios (Mass%) ternary point and plot these into a 3d space. It is designed to be used 
in conjunction with a second script; Anisothermal Ternary Generator.

This code was written by Sam Yeates (The University of Sheffield) as part of his PhD, supervised by Prof. Russell Goodall 
and Prof. Katerina Christofidou. 
This version of the script was finalised on 28/04/2025.

DISCLAIMER - This code does not carry out any CALPHAD or Thermodynamic modelling. It purely uses TC_Python to extract
phase data. Therefore data outputted from this code is only ever as accurate as ThermoCalc. Users should ensure they 
validate any results to the accuracy/detail their work requires. 
This tool is designed first and foremost as a way to visualise systems to allow for first-pass alloy design or for 
scientific communication purposes. It is not designed to act as a precise or accurate design tool.

"""
import numpy as np
import csv
import plotly.graph_objects as pgo
import plotly.express as px
from scipy.ndimage import gaussian_filter
from skimage import measure
import plotly.io as pio
pio.renderers.default = 'browser'



#%% Select you file

#Hypothetical Datasets
#filename = 'Test Eutectic Valley.csv'
#filename = 'ternary_dataset_eutectic_line_1k.csv'

#Metallic Ternary Test Files
#filename = 'V_Cr_Ti-300K-2300K-25K step.csv'
#filename = 'V_Cr_Ti_400K_2300K-25K step.csv'
#filename = 'Al_Fe_Si_700K-1300K_25K step.csv'
#filename = 'Zn-Cu-Al-400K-1000K_25K step.csv'
#filename = 'Cu_Ni_Sb_700K-1800K_25K step.csv'
#filename = 'Al_Cu_Ni_700K-1800K_25K step.csv'
#filename = 'Ti_Al_V_700K_2300K-1wt%-25K step.csv'
#filename = 'Ti_Al_V_700K_2300K-5wt%-25K step.csv'
#filename = 'Nb_Ti_Ni_300K_1300K-25K step.csv'


#Metallic Partial Ternary Test Files
#filename = 'CuCr(0-2wt%)_Zr(0-2wt%)_600K_1400K-20K step.csv'
#filename = 'V_Cr(0-20wt%)_Ti(0-20wt%)_400K_600K_25K step.csv'
#filename = 'V_Cr(0-20wt%)_Ti(0-40wt%)_1000K_1800K-25K step.csv'
#filename = 'V_Cr(0-50wt%)_Ti(0-60wt%)_400K_500K-25K step.csv'

#'Interesting' Datasets

#Psuedo Ternary Test Files
#filename = 'Ta-18Ti-17V_W_Cr_700K_1200K-25K step.csv'
filename = 'Ta-18V-17Ti_W(0-10wt%,0.001)_Cr(0-10wt%,0.001)-600K-1075K-25K step.csv'
#filename = "V-4Cr-4Ti_Al_Zn_800K-900K_20K step.csv"
#filename  = 'V_C(0-0.001wt%)_N(0-0.001wt%)-600K-1075K-25K step.csv'

#V44:C:N
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_N(0-0.001wt%)-600K-700K-25K step.csv'
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_N(0-0.001wt%)-600K-1600K-25K step.csv'
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_N(0-0.001wt%)-275K-1075K-20K step.csv'
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_N(0-0.001wt%)_0.00001wt%-600K-800K-25K step.csv'
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_N(0-0.001wt%)- 0.000025wt% -600K-1075K-25K step.csv'

#V44:O:N
#filename = 'V-4Cr-4Ti_O(0-0.001wt%)_N(0-0.001wt%)-300K-1600K-25K step.csv'

#V44:C:O
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_O(0-0.001wt%)-300K-1600K-25K step.csv' #Initial 0.00005wt% run
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_O(0-0.001wt%)_0.00001wt%-400K-1050K-25K step.csv' #0.00001wt% hi res run
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_O(0-0.001wt%)_0.00005wt%_2TPa-400K-1050K-25K step' #Part of pressure effect test
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_O(0-0.001wt%)_0.00005wt%_2GPa-400K-1050K-25K step' #Part of pressure effect test
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_O(0-0.001wt%)_0.00005wt%_2MPa-400K-1050K-25K step' #Part of pressure effect test
#filename = 'V-4Cr-4Ti_C(0-0.001wt%)_O(0-0.001wt%)_0.00005wt%_100MPa-400K-1050K-25K step' #Part of pressure effect test
#%%
def PlotterFunction(axis1_list,axis2_list,axis3_list, PhaseList, TempList, AxisLabels,filename,TempRes=None, sigma=None):
    

    A1, A2, A3 = AxisLabels
    Corner_Labels = [A1,A2,A3]
    
    
    #------------------Rescale Ternary Data-----------------------
    # — Step 0: record the original data ranges for tick labels —
    
    OGMin1,OGMax1 = min(axis1_list), max(axis1_list)
    OGMin2,OGMax2 = min(axis2_list), max(axis2_list)
    OGMin3,OGMax3 = min(axis3_list), max(axis3_list)
      
   
    
    #Convert list to array for numpy processing 
    Axis1Array = np.array(axis1_list)
    Axis2Array = np.array(axis2_list)
    Axis3Array = np.array(axis3_list)
    
    
    #Scale the dataset
    ScaleAxis1Array = (Axis1Array-OGMin1)/(OGMax1-OGMin1)
    ScaleAxis2Array = (Axis2Array-OGMin2)/(OGMax2-OGMin2)
    ScaleAxis3Array = (Axis3Array-OGMin3)/(OGMax3-OGMin3)
    

    
    #Normalise/Close the scaled dataset    
    NormAxis2Array = ScaleAxis2Array/(ScaleAxis1Array+ScaleAxis2Array+ScaleAxis3Array)
    NormAxis3Array = ScaleAxis3Array/(ScaleAxis1Array+ScaleAxis2Array+ScaleAxis3Array)
    

    #Create equilateral 'Unit Length 1' triangle coordinates
    Vertex1Array = np.array([0.0, 0.0])
    Vertex2Array = np.array([1.0, 0.0])
    Vertex3Array = np.array([0.5, np.sqrt(3)/2.0])

    
        
    #------------------Create 'axis' parameters--------------------
        # 1) Define your triangle vertices in Cartesian coords
    CentroidArray = (Vertex1Array + Vertex2Array + Vertex3Array) / 3
    
    # 2) Parameters: number of ticks per edge, and their length
    NumTicks   = 9
    TickLength = 0.02
    CornerLabelOffet = 0.04
    LabelOffset = 3.0
    
    # 2) Compute unit “outward” direction for each corner
    CornerCoors = [Vertex1Array, Vertex2Array, Vertex3Array]
    CornerDirection = []
    for Corner in CornerCoors:
        Vec = Corner - CentroidArray
        CornerDirection.append(Vec / np.linalg.norm(Vec))


    # edges and their corresponding data‐ranges:
    # 2) Define your edges and which raw‐range they correspond to
    Edges = [
        ((Vertex1Array, Vertex2Array), (OGMin2, OGMax2)),  # edge v1→v2 spans axis3
        ((Vertex2Array, Vertex3Array), (OGMin3, OGMax3)),  # v2→v3 spans axis1
        ((Vertex3Array, Vertex1Array), (OGMin1, OGMax1)),  # v3→v1 spans axis2
    ]
    # 3) Build the base outline (edges + ticks) at z=0
    XSBase, YSBase, ZSBase = [], [], []
    
    for Start, End in [(Vertex1Array, Vertex2Array), (Vertex2Array, Vertex3Array), (Vertex3Array, Vertex1Array)]:
        XSBase += [Start[0], End[0], None]
        YSBase += [Start[1], End[1], None]
        ZSBase += [0, 0, None]
        
    
    #------------------Create Tick Labels-----------------------    

    TickTextX, TickTextY = [], []
    TickLabelString = []
        
    for (Start, End), (RawLow, RawHigh) in Edges:
        # equal points in normalized space
        Points   = np.linspace(Start, End, NumTicks + 2)[1:-1]
        Fractions = np.linspace(0, 1, NumTicks + 2)[1:-1]
    
        # choose decimals based on span
        Span = (RawHigh - RawLow)*100
        if Span > 50:
            Decimals = 0
        elif Span > 1:
            Decimals = 1
        elif Span > 0.1:
            Decimals = 2 
        elif Span > 0.01:
            Decimals = 3
        elif Span > 0.001:
            Decimals = 4
        else:
            Decimals = 5
        fmt = f"{{:.{Decimals}f}}"
    
        Tangent = (End - Start) / np.linalg.norm(End - Start)
        Normal  = np.array([-Tangent[1], Tangent[0]]) * -1  # inward
    
        for (x0, y0), Fraction in zip(Points, Fractions):
            # draw the tick line
            x1, y1 = x0, y0
            x2, y2 = x0 + Normal[0]*TickLength, y0 + Normal[1]*TickLength
            XSBase += [x1, x2, None]
            YSBase += [y1, y2, None]
            ZSBase += [0,  0,  None]
    
            # compute and format the raw‐value label
            RawVal = RawLow + Fraction * (RawHigh - RawLow)
            TickLabelString.append(fmt.format((RawVal*100)))
    
            # decide where to put that text
            TextX = x0 + Normal[0]*TickLength*LabelOffset
            TextY = y0 + Normal[1]*TickLength*LabelOffset
            TickTextX.append(TextX)
            TickTextY.append(TextY)
    
        
    
    #------------Convert Ternary Data to Cartesian Data------------

    # --- Prepare grid and label array ---
    UniqueA2 = np.unique(ScaleAxis2Array)
    UniqueA3 = np.unique(ScaleAxis3Array)
    NA2 = UniqueA2.size
    NA3 = UniqueA3.size
    if NA2==NA3:
        N=NA2
    else:
        print("Axis resolution are different. Selecting min. value")
        N = min(NA2, NA3)
   
    
    A1SArray = np.linspace(0, 1, N)
    A2SArray = np.linspace(0, 1, N)
    A1GridArray, A2GridArray = np.meshgrid(A1SArray, A2SArray, indexing='ij')

    mask = (A1GridArray + A2GridArray) > 1
    
    
    # Unique phases
    UniquePhases = sorted(set(PhaseList))
    NumUniquePhases = len(UniquePhases)
    PhaseToInt = {ph: i for i,ph in enumerate(UniquePhases)}
    
    
    SortedUniqueTemperature = sorted(set(TempList))
    # determine Tres if unset
    if TempRes is None and len(SortedUniqueTemperature) > 1:
        dTs = np.diff(SortedUniqueTemperature)
        if not np.allclose(dTs, dTs[0]):
            raise ValueError("Non-uniform T_list spacing; set Tres manually.")
        TempRes = float(dTs[0])
    TempRes = TempRes or 1.0
    TempMin, TempMax = SortedUniqueTemperature[0], SortedUniqueTemperature[-1]
    
    TempIndex = {T: k for k, T in enumerate(SortedUniqueTemperature)}
    LabelGrid = -1 * np.ones((N, N, len(SortedUniqueTemperature)), dtype=int)

    #Label Grid Values are defined here
    # vectorized index computation
    IIndex = np.rint(NormAxis2Array * (N-1)).astype(int)
    JIndex = np.rint(NormAxis3Array * (N-1)).astype(int)
    # clip to avoid rare float‐overshoot
    IIndex = np.clip(IIndex, 0, N-1)
    JIndex = np.clip(JIndex, 0, N-1)
    
    # fill in one go
    for ii, jj, Phase, T in zip(IIndex, JIndex, PhaseList, TempList):
        k = TempIndex[T]
        LabelGrid[ii, jj, k] = PhaseToInt[Phase]
                
            
            
    #------------------Set up Colour Scales-----------------------

    
    Palette   = px.colors.qualitative.Plotly
    PhaseColours = {ph: Palette[i % len(Palette)] for ph,i in PhaseToInt.items()}
    ColourScale = [[i/(NumUniquePhases-1), PhaseColours[UniquePhases[i]]] for i in range(NumUniquePhases)]
 
    
   
    #------------------Start Plotting-----------------------
    
    Fig = pgo.Figure()
   
    AxisTraceIndex =[]
    LabelTraceIndex =[]
    TickLabelTraceIndex = []

    #------------------Add 2D Slices-----------------------
    SliceTraceIndex = []
    # mask for outside triangle

    # add each temperature slice
    for k, T in enumerate(SortedUniqueTemperature):
        StartIndex = len(Fig.data)
        PhaseGrid = LabelGrid[:, :, k].astype(float)
        # compute 2D coords
        #TGridArray = 1 - A1GridArray - CGridArray
        
        XGridArray = (A1GridArray + 0.5*A2GridArray)
        YGridArray = (np.sqrt(3)/2)*A2GridArray
        Z = np.full_like(XGridArray, T)

        # mask out-of-triangle
        XGridArray[mask] = YGridArray[mask] = Z[mask] = np.nan
        
        PhaseGrid = np.where(PhaseGrid < 0, np.nan, PhaseGrid)    # any original –1 → NaN
        PhaseGrid[mask] = np.nan 
        
        MissingCells = np.isnan(PhaseGrid)
        XGridArray[MissingCells] = np.nan
        YGridArray[MissingCells] = np.nan
        Z[MissingCells] = np.nan
        
                
        Rows, Cols = XGridArray.shape
        CustomData = np.empty((Rows, Cols), dtype=object)
        
        # raw_loX, raw_hiX are your orig_min1/orig_max1, etc.
        Raw1 = 100*(OGMin3 + A1GridArray*(OGMax3-OGMin3))
        Raw2 = 100*(OGMin2 + A2GridArray*(OGMax2-OGMin2))
        Raw3 = 100*(OGMin1 + (1-A1GridArray-A2GridArray)*(OGMax1-OGMin1))
        
        
        for i in range(Rows):
            for j in range(Cols):
                if not mask[i, j] and (PhaseGrid[j, i] == PhaseGrid[j, i]):  # pg[j,i]==NaN fails
                    Index = int(PhaseGrid[j, i])   # in [0..nph-1]
                    Phase = UniquePhases[Index]
                    CustomData[i, j] = (Raw1[i, j],Raw2[i, j],Raw3[i, j], Phase, T)
                else:
                    CustomData[i, j] = None
                        
        Fig.add_trace(pgo.Surface(
            x=XGridArray,
            y=YGridArray,
            z=Z,
            surfacecolor=PhaseGrid,
            customdata=CustomData,
            colorscale=ColourScale,
            cmin=0, 
            cmax=NumUniquePhases-1, 
            showscale=False,
            hovertemplate=(    f"{A1}: %{{customdata[2]:.{Decimals}f}} wt%<br>"
            f"{A2}: %{{customdata[1]:.{Decimals}f}} wt%<br>"
            f"{A3}: %{{customdata[0]:.{Decimals}f}} wt%<br>"
            "Phase: %{customdata[3]}<br>"
            "Temp: %{customdata[4]:.0f} K<br>"
            "<extra></extra>"
            ),
            name=f"{int(T)} K",
            legendgroup=str(int(T)),
            opacity=0.3,
            visible = False,
            hoverinfo='skip',
            contours=dict(
                z=dict(
                    show=True,
                    start=0.5,
                    end=NumUniquePhases-1.5,
                    size=1,
                    color="black",
                    width=2
                )
            )
        ))
        
        SliceTraceIndex.append(StartIndex)
           
    
    
        ZS = [(z0 + T) if z0 is not None else None for z0 in ZSBase]
        outline_idx = len(Fig.data)
        Fig.add_trace(
            pgo.Scatter3d(
                x=XSBase,
                y=YSBase,
                z=ZS,
                mode='lines',
                hoverinfo='skip',
                line=dict(color='black', width=2),
                showlegend=False,
                visible=(k == 0)
            )
        )
        AxisTraceIndex.append(outline_idx)
        
    
        
        #Corner Label Trace
        # unpack into x, y, and use T for all z
        XSLabel = []
        YSLabel = []
        ZSLabel = []
        for Corner, Direction in zip(CornerCoors, CornerDirection):
            XSLabel.append(Corner[0] + Direction[0] * CornerLabelOffet)
            YSLabel.append(Corner[1] + Direction[1] * CornerLabelOffet)
            ZSLabel.append(T)
        
        # 3) add the corner‐labels at z=T
        LabelIndex = len(Fig.data)
        Fig.add_trace(
            pgo.Scatter3d(
                x=XSLabel,
                y=YSLabel,
                z=ZSLabel,
                mode='text',
                hoverinfo='skip',
                text=Corner_Labels,              # your 3 labels
                textposition='middle center',
                textfont=dict(color='black', size=12),
                showlegend=False,
                visible=(k == 0)
            )
        )
        LabelTraceIndex.append(LabelIndex)
        
        #Tick Value Label Trace
        text_idx = len(Fig.data)
        Fig.add_trace(pgo.Scatter3d(
            x=TickTextX,
            y=TickTextY,
            z=[T]*len(TickTextX),
            mode='text',
            hoverinfo='skip',
            text=TickLabelString,
            textposition='middle center',
            textfont=dict(color='black', size=10),
            showlegend=False,
            visible=(k==0)
        ))
        TickLabelTraceIndex.append(text_idx)
        
        
        
     #------------------Add 3D Volumes-----------------------
     
    # 1) plot the 3D mesh volumes (unchanged)
    
        """VArray = VGridArray*100            # shape (N,N)
        CArray = CGridArray*100
        UArray = (1.0 - VGridArray - CGridArray)*100
        
        rows, cols = XGridArray.shape
        cd = np.empty((rows, cols), dtype=object)
        
        for i in range(rows):
            for j in range(cols):
                if not mask[i, j]:
                    ph = unique_ph[int(pg[i, j])]
                    # pack the triple + phase + temp
                    cd[i, j] = (VArray[i, j], CArray[i, j], UArray[i, j], ph, T)
                else:
                    cd[i, j] = None"""
    
    
    VolumeTraceIndex =[]
    for Phase in UniquePhases:
        Index = PhaseToInt[Phase]
        mask = (LabelGrid==Index).astype(float)
        if sigma is not None:
            mask = gaussian_filter(mask, sigma=sigma)
        verts, faces, _, _ = measure.marching_cubes(
            np.pad(mask,1,mode='constant'), level=0.5
        )
        verts -= 1
        
        IIndex = verts[:,0].astype(int)
        JIndex = verts[:,1].astype(int)
        KIndex = verts[:,2].astype(int)
        
        
        f1 = A1SArray[IIndex]
        f2 = A2SArray[JIndex]
      

       
        X = f1 + 0.5*f2
        Y = (np.sqrt(3)/2.0)*f2
        Z = KIndex * TempRes + TempMin
        
        I,J,K = faces.T
        VolumeIndex = len(Fig.data)
        Fig.add_trace(pgo.Mesh3d(
            x=X, y=Y, z=Z,
            i=I,j=J,k=K,
            customdata=CustomData,
            color=PhaseColours[Phase], opacity=0.4,
            hoverinfo='skip',
            flatshading=True, name=Phase, legendgroup=Phase,
            showlegend=True, showscale=False
        ))
        VolumeTraceIndex.append(VolumeIndex)

    #------------------Build Volume Buttons--------------------  
    Buttons = []
    # 1) Build the “All Phases” visibility mask
    AllVisibility = [False] * len(Fig.data)
    for VisibleIndex in VolumeTraceIndex:
        AllVisibility[VisibleIndex] = True
    
    # show the first slice’s axes+labels too:
    AllVisibility[AxisTraceIndex[0]]   = True
    AllVisibility[LabelTraceIndex[0]]  = True
    AllVisibility[TickLabelTraceIndex[0]]  = True
    
    
    Buttons = [
        dict(
            label="All Phases",
            method="update",
            args=[{"visible": AllVisibility}, {}]
        )
    ]
    
    # 2) One button per phase
    for PhaseIndex, Phase in enumerate(UniquePhases):
        Visibility = [False] * len(Fig.data)
        # only this mesh
        Visibility[VolumeTraceIndex[PhaseIndex]] = True
        # plus the axes/labels for slice 0
        Visibility[AxisTraceIndex[0]]     = True
        Visibility[LabelTraceIndex[0]]    = True
        Visibility[TickLabelTraceIndex[0]]    = True
    
        Buttons.append(
            dict(
                label=Phase,
                method="update",
                args=[{"visible": Visibility}, {}]
            )
        )
    #------------------Build 2D Slider-----------------------
    # 1) Build the list of slider steps
    Steps = []
    
    # 1) “All” step: only static + Tmin (first layer)  

    # turn on slice 0’s surface, outline & tick labels
    Visibility = [False] * len(Fig.data) 
    for SliceIndex in SliceTraceIndex:
        Visibility[SliceIndex] = False
    for VolumeIndex in VolumeTraceIndex:
        Visibility[VolumeIndex] = True
    Visibility[AxisTraceIndex[0]] = True
    Visibility[LabelTraceIndex[0]] = True
    Visibility[TickLabelTraceIndex[0]] = True
    
    
    Steps.append(dict(
        method="update",
        args=[{"visible": Visibility}],
        label="All",

    ))
                           
    def vis_for(SliceIndex=None):
       n = len(Fig.data)
       return [
           # 1) If this is the “All” step (slice_idx is None) and it’s a volume trace → show it
           True
           if SliceIndex is None and i in VolumeTraceIndex
           # 2) If this is a specific slice step and it’s one of that slice’s traces → show it
           or (SliceIndex is not None and i in {
                   SliceTraceIndex[SliceIndex],
                   AxisTraceIndex[SliceIndex],
                   LabelTraceIndex[SliceIndex],
                   TickLabelTraceIndex[SliceIndex],
               })
           # 3) Otherwise hide the geometry but keep its legend entry
           else "legendonly"
           for i in range(n)
       ]
        
    for i, T in enumerate(SortedUniqueTemperature):
        Steps.append(dict(
            method="update",
            args=[{"visible": vis_for(i)}],
            label=f"{int(T)} K"))
        
    
    Fig.update_layout(
        title=f"({A1})-({A2})-({A3}) - {TempMin:.0f}K-{TempMax:.0f}k Anisothermal Stack",
        updatemenus=[dict(
            buttons=Buttons,
            direction='down',
            showactive=True,
            x=0, y=1.0,
            xanchor='left', yanchor='bottom'
        )],
        sliders=[dict(
            active=0,            # start on "All"
            currentvalue={"prefix": "Viewing: "},
            pad={"t": 80},       # move it down a bit
            steps=Steps
        )],
        annotations=[
            dict(
                text=(
                    "This figure has been generated using the Anisothermal Ternary Plotter, "
                    "produced by S. F. Yeates (The University of Sheffield). "
                    "For questions regarding use of this tool, contact sfyeates1@sheffield.ac.uk."
                ),
                xref="paper", yref="paper",
                x=0,     # left‐align
                y=-0.12, # a little below the plot area
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="left")],
        scene=dict(
            xaxis=dict(title="",showticklabels=False,ticks='', showgrid=False, zeroline=False,showbackground=False),
            yaxis=dict(title="",showticklabels=False,ticks='', showgrid=False, zeroline=False,showbackground=False),
            zaxis=dict(title="Temperature (K)")
        )
        
    )
    
    
        # if you haven’t already:
    Fig.update_layout(
        margin=dict(b=100),  # make extra room at the bottom
        annotations=[
            dict(
                text=(
                    "This figure has been generated using the Anisothermal Ternary Plotter, "
                    "produced by S. F. Yeates (The University of Sheffield). "
                    "For questions regarding use of this tool, contact sfyeates1@sheffield.ac.uk."
                ),
                xref="paper", yref="paper",
                x=0,     # left‐align
                y=-0.12, # a little below the plot area
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="left"
            )
        ]
    )

     
    Fig.write_html(f"{filename}.html")
    Fig.show()
    
# --- CSV loading and call ---

with open(filename, newline='') as f:
    Lines = f.readlines()

MetaData, DataLines = {}, []
for Line in Lines:
    if Line.startswith('#'):
        key, val = Line[1:].split('=', 1)
        MetaData[key.strip().lower()] = val.strip()
    elif Line.strip():
        DataLines.append(Line)

AxisDefs = [grp.split(',') for grp in MetaData['axis_defs'].split(';')]
AxisLabels = MetaData['axis_names'].split(';')


# Flatten list of all elements for initial read-in
AllElements = sorted({el for group in AxisDefs for el in group})

Read = csv.DictReader(DataLines)

ElementList = {el: [] for el in AllElements}
TempList, PhaseList = [], []
TempColumn = Read.fieldnames[0]

# Read data into elemental lists
for Row in Read:
    # Temperature and phases
    TempList.append(float(Row[TempColumn]))
    PhaseList.append(Row['Phases'])
    # Each element's mass percentage
    for el in AllElements:
        Col = f"{el}/Mass %"
        ElementList[el].append(float(Row[Col]))

# Now sum grouped elements into pseudo-axes
AxisLists = []
for Group in AxisDefs:
    summed = []
    for i in range(len(TempList)):
        total = sum(ElementList[el][i] for el in Group)
        summed.append(total)
    AxisLists.append(summed)

Axis1ListRaw, Axis2ListRaw, Axis3ListRaw = AxisLists

A1Array = np.array(Axis1ListRaw)
A2Array = np.array(Axis2ListRaw)
A3Array = np.array(Axis3ListRaw)

A2Max = np.max(A2Array)
A3Max = np.max(A3Array)


tol = 1e-14
Mask = (A2Array <= A2Max) \
       & (A3Array <= A3Max) \
       & ((A2Array/A2Max + A3Array/A3Max) <= 1.0 + tol)

# Keep only the “triangle rows” in each of your four parallel lists:
Axis1List = A1Array[Mask].tolist()
Axis2List = A2Array[Mask].tolist()
Axis3List = A3Array[Mask].tolist()

PhaseList = [phase for keep, phase in zip(Mask, PhaseList) if keep]
TempList     = [T for keep, T in zip(Mask, TempList) if keep]
 



Name = filename[:-4]


PlotterFunction(Axis1List, Axis2List, Axis3List, PhaseList,
                TempList, AxisLabels,Name, TempRes=None, sigma=None)

