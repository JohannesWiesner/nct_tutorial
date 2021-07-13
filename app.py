# -*- coding: utf-8 -*-
"""
Network Control Theory Tutorial

@author: Johannes.Wiesner
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import dash
import dash_cytoscape as cyto
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_table
import itertools
import operator
import plotly.express as px
from network_control.utils import matrix_normalization
from network_control.energies import minimum_energy,optimal_energy
from nct_utils import state_trajectory

###############################################################################
## Set Default Data ###########################################################
###############################################################################

# set seed
np.random.seed(28)

# create a default adjacency matrix
A = np.array([[0, 1, 1, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 1, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 0, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 0]])

# create default (random) x0 and xf (between 0 and 1)
states_df = pd.DataFrame({'x0':np.random.rand(len(A)),'xf':np.random.rand(len(A))})

###############################################################################
## Dash App ###################################################################
###############################################################################

## Topology-Modification ######################################################

# FIXME: transform networkx coordinates into dash/plotly space
# - Positions could however also be irrelevant here, because layout component
# from dash can also decide automatically over node positions
def from_A_to_elements(A):
    '''Create a lists of elements from a numpy adjaceny matrix that can be inter-
    preted by dash_cytoscape.Cytoscape. The following steps are implemented from 
    https://community.plotly.com/t/converting-networkx-graph-object-into-cytoscape-format/23224/2
    '''
    
    # create graph object
    G = nx.Graph(A)
    
    # get node positions
    pos = nx.spring_layout(G)
    
    # convert networkx to cytoscape layout
    cy = nx.readwrite.json_graph.cytoscape_data(G)
    
    # Add the dictionary key 'label' to the node dict (this is a required attribute for dash)
    # Delete the key 'value' from node dict (not needed)
    # Delete the key 'name' from node dict (not needed)
    # Add the dictionary key 'controller' to the node dict and set to True
    for node_dict in cy['elements']['nodes']:
         for _,d in node_dict.items():
             d['label'] = d.pop('value')
             del d['name'] 
             d['controller'] = True
             
    # NOTE: in cytoscape, all ids of the nodes must be strings, that's why
    # we convert the edge ids also to strings (but check if this is really 
    # necessary)
    for edge_dict in cy['elements']['edges']:
        for _,d in edge_dict.items():
            d['source'] = str(d['source'])
            d['target'] = str(d['target'])
    
    # Add the positions you got from as a value for data in the nodes portion of cy
    # NOTE: This might be not necessary, as positions can be automatically
    # determined in the layout attribute from cyto.Cytoscape (see FIXME above)
    for n,p in zip(cy['elements']['nodes'],pos.values()):
        n['pos'] = {'x':p[0],'y':p[1]}
    
    # Take the results and write them to a list
    elements = cy['elements']['nodes'] + cy['elements']['edges']
    
    return elements

# NOTE: What's that utils module? https://dash.plotly.com/cytoscape/reference
def get_edge_dicts(elements):
    '''Extract all edge dictionaries from elements. Edge dicts are
    identfied by their 'weight' key'''
    
    edge_dicts = []
    
    for d in elements:
        if 'weight' in d['data']:
            edge_dicts.append(d)
            
    return edge_dicts

# NOTE: What's that utils module? https://dash.plotly.com/cytoscape/reference
def get_node_dicts(elements):
    '''Extract all node dictionaries from elements. Node dicts are
    identified by not having a 'weight' key'''
    
    node_dicts = []
    
    for d in elements:
        if not 'weight' in d['data']:
            node_dicts.append(d)
            
    return node_dicts

def add_edges(selectedNodeData,elements):
    '''For each combination of selected nodes, check if this combination is connected 
    by an edge. If not, create an edge dict for that combination and modify the elements list'''
    
    edge_dicts = get_edge_dicts(elements)
    edge_ids = [(d['data']['source'],d['data']['target']) for d in edge_dicts]
    
    # get a list of ids of all nodes that user has currently selected and that
    # should be connected by an edge. Sort the list alphanumerically (that ensures 
    # that we get only get combinations of source and target ids where source id 
    # is always the lower integer)
    node_ids = [d['id'] for d in selectedNodeData]
    node_ids.sort()
    
    # create all pair-wise combinations of the selected nodes
    source_and_target_ids = list(itertools.combinations(node_ids,2))
    
    # for each source and target tuple, check if this edge already exists. If not,
    # create a new edge dict and add it to elements
    # NOTE: Currently, new edges always get a default weight of 1. This could/should
    # be fixed (either by adding an entry field where user can select
    # weight strength when creating the new edges or by allowing user to modify
    # edge strength post-hoc after edge is created)
    for (source,target) in source_and_target_ids:
        if not (source,target) in edge_ids:
              new_edge = {'data':{'weight':1.0,'source':source,'target':target}}
              elements.append(new_edge)
    
    return elements

def drop_edges(selectedEdgeData,elements):
    '''Drop an input list of selected edges from cytoscape elements'''

    # get source and target ids for all currently selected edges
    source_and_target_ids = [(d['source'],d['target']) for d in selectedEdgeData]
    
    # iterate over all dictionaries in elements, identify edge dicts by their
    # 'weight' key and check again if this edge dict belongs to the currently selected
    # edges. If yes, add its index to list of to be dropped dictionaires.
    drop_these_dicts = []
    
    for idx,d in enumerate(elements):
        if 'weight' in d['data']:
            if (d['data']['source'],d['data']['target']) in source_and_target_ids:
                drop_these_dicts.append(idx)
    
    # drop selected edge dictionaries from elements
    elements = [i for j,i in enumerate(elements) if j not in drop_these_dicts]
    
    return elements

## Figure Plotting ###########################################################

def from_elements_to_A(elements):
    '''Extract nodes and edges from current elements and convert them to 
    adjacency matrix
    '''

    # FIXME: This is inefficient, we iterate over the same list twice
    edge_dicts = get_edge_dicts(elements)
    node_dicts = get_node_dicts((elements))
    
    edges = [(d['data']['source'],d['data']['target'],d['data']['weight']) for d in edge_dicts]
    nodes = [d['data']['id'] for d in node_dicts]
    
    n_nodes = len(nodes)
    A = np.zeros((n_nodes,n_nodes))

    for edge in edges:
        i = int(edge[0])
        j = int(edge[1])
        weight = edge[2]
        A[i,j] = weight
        A[j,i] = weight
    
    return A

def from_elements_to_B(elements):
    '''Extract nodes from current elements, check which nodes are selected
    as controllers and get a corresponding control matrix B that can be 
    fed to control_package functions.
    '''
    
    # get a list of all nodes from current elements (get their ID and their
    # controller attribute)
    node_dicts = get_node_dicts(elements)
    nodes = [(d['data']['id'],d['data']['controller']) for d in node_dicts]
    
    # sort nodes by their ids and get controller attribute
    nodes.sort(key=operator.itemgetter(0))
    c_attributes = [n[1] for n in nodes]

    # create B matrix
    B = np.zeros(shape=(len(nodes),len(nodes)))

    for idx,c in enumerate(c_attributes):
        if c == True:
            B[idx,idx] = 1
      
    return B

def get_state_trajectory_fig(A,x0,T,c):
    '''Generate a plotly figure that plots a state trajectory using an input 
    matrix, a source state, a time horizon and a normalization constant'''
    
    # simulate state trajectory
    x = state_trajectory(A=A,xi=x0,T=T)
    
    # create figure
    x_df = pd.DataFrame(x).reset_index().rename(columns={'index':'Node'})
    x_df = x_df.melt(id_vars='Node',var_name='t',value_name='Value')
    fig = px.line(x_df,x='t',y='Value',color='Node')
    fig.update_layout(title='x(t+1) = Ax(t)',title_x=0.5)
    
    return fig

# TO-DO: n_err should also be visualized somewhere
def get_minimum_energy_figure(A,T,B,x0,xf,c):
            
    # compute minimum energy
    x,u,n_err = minimum_energy(A,T,B,x0,xf)
    
    # create figure (FIXME: could the following be shorted?)
    x_df = pd.DataFrame(x).reset_index().rename(columns={'index':'t'})
    x_df = x_df.melt(id_vars='t',var_name='Node',value_name='Value')
    x_df['Type'] = 'x'
    
    u_df = pd.DataFrame(u).reset_index().rename(columns={'index':'t'})
    u_df = u_df.melt(id_vars='t',var_name='Node',value_name='Value')
    u_df['Type'] = 'u'
    
    fig_df = pd.concat([x_df,u_df])
    fig = px.line(fig_df,x='t',y='Value',color='Node',facet_row='Type')
    fig.update_layout(title='Minimum Control Energy',title_x=0.5)
    
    return fig

# TO-DO: n_err should also be visualized somewhere
def get_optimal_energy_figure(A,T,B,x0,xf,rho,S,c):
            
    # compute optimal energy
    x,u,n_err = optimal_energy(A,T,B,x0,xf,rho,S)
    
    # create figure (FIXME: could the following be shorted?)
    x_df = pd.DataFrame(x).reset_index().rename(columns={'index':'t'})
    x_df = x_df.melt(id_vars='t',var_name='Node',value_name='Value')
    x_df['Type'] = 'x'
    
    u_df = pd.DataFrame(u).reset_index().rename(columns={'index':'t'})
    u_df = u_df.melt(id_vars='t',var_name='Node',value_name='Value')
    u_df['Type'] = 'u'
    
    fig_df = pd.concat([x_df,u_df])
    fig = px.line(fig_df,x='t',y='Value',color='Node',facet_row='Type')
    fig.update_layout(title='Optimal Control Energy',title_x=0.5)
    
    return fig
    
###############################################################################
## Dash App ###################################################################
###############################################################################

# run dash and take in the list of elements
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        
    # cytoscape graph
    cyto.Cytoscape(
        id='cytoscape-compound',
        layout={'name':'cose'},
        style={'width':'50%', 'height':'500px'},
        elements=from_A_to_elements(A), # initialize elements with a function call
        selectedNodeData=[],
        selectedEdgeData=[],
        stylesheet=[
            # Add show node labels
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)'
                }
            },
            # Style all nodes that are controller nodes
            {
                'selector': '[?controller]',
                'style': {
                    'border-width': 2,
                    'border-color': 'black'
                }
            }
        ]
    ),
    
    # x0/xf data table
    dash_table.DataTable(
        id='states-table',
        columns=[{'id': 'x0','name':'x0','type':'numeric'},{'id':'xf','name':'xf','type':'numeric'}],
        data=states_df.to_dict('records'),
        editable=True
    )]),
    
    # topology control elements
    html.Div([html.Button('(Dis-)Connect Nodes',id='edge-button',n_clicks=0),
              html.Button('(Un-)set controll',id='controll-button',n_clicks=0)]
             ),
    # network control elements
    html.Div([dcc.Input(id='T',type="number",debounce=True,placeholder='Time Horizon (T)',value=20),
              dcc.Input(id='c',type="number",debounce=True,placeholder='Normalization Constant (c)',value=1),
              dcc.Input(id='rho',type="number",debounce=True,placeholder='rho',value=1),
              html.Button('Plot Trajectories',id='plot-button',n_clicks=0)
              ]),
    
    # figures
    dcc.Graph(id='state-trajectory-fig',figure={}),
    dcc.Graph(id='minimum-energy-fig',figure={}),
    dcc.Graph(id='optimal-energy-fig',figure={}),
    
    # debugging fields (can be deleted when not necessary anymore)
    html.Div([
    html.Pre(id='selected-node-data-json-output'),
    html.Pre(id='selected-edge-data-json-output'),
    html.Pre(id='current-elements')
    ])
])

## Just for debugging (can be deleted when not necessary anymore ##############

@app.callback(Output('selected-node-data-json-output','children'),
              Input('cytoscape-compound','selectedNodeData'))
def displaySelectedNodeData(data):
    return json.dumps(data,indent=2)

@app.callback(Output('selected-edge-data-json-output','children'),
              Input('cytoscape-compound','selectedEdgeData'))
def displaySelectedEdgeData(data):
    return json.dumps(data,indent=2)

@app.callback(Output('current-elements','children'),
              Input('cytoscape-compound','elements'))
def displayCurrentElements(elements):
    return json.dumps(elements,indent=2)

## Callback Functions #########################################################

@app.callback(Output('cytoscape-compound','elements'),
              Input('edge-button','n_clicks'),
              Input('controll-button','n_clicks'),
              State('cytoscape-compound','elements'),
              State('cytoscape-compound','selectedNodeData'),
              State('cytoscape-compound','selectedEdgeData'),
              prevent_initial_call=True)

def updateElements(edge_button,controll_button,elements,selectedNodeData,selectedEdgeData):
    
    # check which button was triggered
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # add or delete edges
    if button_id == 'edge-button':
        
        # user must exclusively selecte either nodes OR edges but not both at the same time
        if len(selectedNodeData) > 0 and len(selectedEdgeData) > 0:
            return elements
        
        if len(selectedNodeData) >= 2 and len(selectedEdgeData) == 0:
            return add_edges(selectedNodeData,elements)
        
        if len(selectedNodeData) == 0 and len(selectedEdgeData) >= 1:
            return drop_edges(selectedEdgeData,elements)
        
        else:
            return elements
     
    # set or unset controll nodes
    elif button_id == 'controll-button':

        # get a list of ids of all nodes that user has currently selected and for
        # which controll state should be switched
        node_ids = [d['id'] for d in selectedNodeData]
        
        # identify node dicts in elements and toggle their 'controller' attribute 
        # if they are part of the selected nodes
        node_dicts = get_node_dicts(elements)
        
        for d in node_dicts:
            if d['data']['id'] in node_ids:
                d['data']['controller'] = not d['data']['controller']
                
        return elements
    
@app.callback(Output('state-trajectory-fig','figure'),
              Output('minimum-energy-fig','figure'),
              Output('optimal-energy-fig','figure'),
              Input('plot-button','n_clicks'),
              State('cytoscape-compound','elements'),
              State('T','value'),
              State('c','value'),
              State('rho','value'),
              State('states-table','derived_virtual_data'),
              prevent_initial_call=True)

def updateFigures(n_clicks,elements,T,c,rho,states_data):
    
    # digest data for network_control package
    A = from_elements_to_A(elements)
    B = from_elements_to_B(elements)
    
    # normalize A
    # FIXME: It should also be possible to not normalize the matrix (i.o.
    # to get an intuition for what happens when you not apply normalization)
    A = matrix_normalization(A,c)
    
    print(f"This is A:\n{A}\n")
    print(f"This is B:\n{B}\n")
    
    # FIXME: Currently we use reshape (-1,1), but this is a bug in network
    # control-package. When this is fixed, we don't need reshape anymore
    x0 = np.array([d['x0'] for d in states_data]).reshape(-1,1)
    xf = np.array([d['xf'] for d in states_data]).reshape(-1,1)
    
    print(f"This is x0:\n{x0}\n")
    print(f"This is xf:\n{xf}\n")
        
    # FIXME: S should also be settable by user (controll which nodes to constrain)
    S = B.copy()
    
    fig_1 = get_state_trajectory_fig(A,x0=x0,T=T,c=c)
    fig_2 = get_minimum_energy_figure(A=A,T=T,B=B,x0=x0,xf=xf,c=c)
    fig_3 = get_optimal_energy_figure(A=A,T=T,B=B,x0=x0,xf=xf,rho=rho,S=S,c=c)
    
    return fig_1,fig_2,fig_3
    
if __name__ == '__main__':
    app.run_server(debug=True)
