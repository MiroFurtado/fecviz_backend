'''
Bluebonnet Campaign Finance Analysis Module

AUTHOR: Miro Furtado
'''
from requests import get
import json
from house_list import candidates_list
import pickle

#Plotly imports
import plotly.offline as plt
import plotly.graph_objs as go

#Math/Analysis imports
from sklearn import linear_model
import numpy as np

#Surpress pandas warnings
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class Campaign:
    """Campaign class.
    This is where we will represent information about projected turnout, finance etc. 
    """
    def __init__(self, state, number, party):
        self.state = state
        self.number = number
        self.cash = 0
        self.party = party
        self.expenditures = 0
        self.json_538 = None
        self.raised_indiv = 0
        self.opposition = None
    
    def is_incumbent(self):
        return self.opposition is None
    
    def funding_share(self):
        "Calculates the share of individual contributions raised by Campaign vs opponent"
        if(self.is_incumbent()):
            return 1.
        return self.raised_indiv/(self.raised_indiv+self.opposition.raised_indiv)
    
    def set_coh(self, coh):
        "Updates the cash on hand for campaign"
        self.cash = coh

    def set_raised_indiv(self, amt):
        "Updates the cash on hand for campaign"
        self.raised_indiv = amt

    def set_expenditures(self, expenditures):
        "Updates the expenditures for campaign"
        self.expenditures = expenditures

class State:
    def __init__(self, name, party='DEM'):
        self.name = name
        self.party = party
        if self.party=='DEM':
            self.opp = 'REP'
        else:
            self.opp = 'DEM'
        self.districts = {}
    
    def gen_report(self):
        self.gen_coh_graph()
        self.gen_stack_graph()
        self.gen_pvi_fundraising_scatter()
        self.gen_performance_graph()

    def gen_pvi_fundraising_scatter(self):
        pvi_list = []
        share_list = []
        for i,district in self.districts.items():
            pvi = district.json_538['pvi']
            share = district.funding_share()
            pvi_list.append(pvi)
            share_list.append(share)

        regr = linear_model.LinearRegression()
        shaped_share = np.array(share_list).reshape(-1,1)
        regr.fit(shaped_share,pvi_list)

        trace = go.Scatter(
            x = 100*share_list,
            y = pvi_list,
            mode = 'markers'
        )
        trace2 = go.Scatter(
            x=share_list, 
            y=regr.predict(shaped_share),
            mode='lines',
            line=dict(color='blue', width=3)
        )


        data = [trace,trace2]
        layout = go.Layout(
            title=('Fundraising share by Cook PVI (%s %s)' % (self.name, self.party)),
            showlegend=False,
            annotations=[
                dict(
                    x=0.1,
                    y=43,
                    xref='x',
                    yref='y',
                    text='R squared = %.3f' % regr.score(shaped_share,pvi_list),
                    showarrow=False,
                )],
                xaxis=dict(title='Share of Fundraising'),
                yaxis=dict(title='Cook PVI')
        )

        fig = go.Figure(data=data, layout=layout)

        self.regr = regr
        # Plot and embed in ipython notebook!
        plt.iplot(fig, filename='basic-scatter',show_link=False, config={'displayModeBar': False})
    
    def gen_performance_graph(self,out_file=None):
        if(not out_file):
            out_file = ("%s_%s_performance_viz.html" % (self.name,self.party))

        pvi_list = []
        share_list = []
        nums_district = []
        for i,district in self.districts.items():
            nums_district.append(i)
            pvi = district.json_538['pvi']
            share = district.funding_share()
            pvi_list.append(pvi)
            share_list.append(share)

        shaped_pvi = np.array(pvi_list).reshape(-1,1)
        shaped_share = np.array(share_list).reshape(-1,1)
        regr = linear_model.LinearRegression()
        regr.fit(shaped_pvi,shaped_share)

        performance = list(np.transpose((shaped_share-regr.predict(shaped_pvi)))[0])
        nums_district, performance = list(zip(*sorted(zip(nums_district,performance),key= lambda x: x[1])[::-1]))
        trace1 = go.Bar(
            x=nums_district,
            y=100*performance
        )
        for i in range(len(performance)):
            if performance[i] < 0:
                break
            last = nums_district[i]
        first = nums_district[i]

        data = [trace1]
        layout = go.Layout(
            title='Financial Performance Chart by %s District'%self.name,
            xaxis=dict(type='category', tickangle=45),
            yaxis=dict(title='Actual Minus Expected Share', tickformat="%f%"),
            shapes= [
                {
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': nums_district[0],
                    'y0': -0.15,
                    'x1': last,
                    'y1': -0.02,
                    'line': {
                        'width': 0,
                    },
                    'fillcolor': 'rgba(50, 171, 96, 1.0)',
                },
                {
                    'type': 'rect',
                    'xref': 'x',
                    'yref': 'y',
                    'x0': first,
                    'y0': 0.02,
                    'x1': nums_district[-1],
                    'y1': 0.15,
                    'line': {
                        'width': 0,
                    },
                    'fillcolor': 'rgba(0,0,0, 1.0)',
                }
            ]
        )
        fig = go.Figure(data=data,layout=layout)
        plt.iplot(fig, filename=out_file,show_link=False, config={'displayModeBar': False})
        

    def gen_stack_graph(self,out_file=None):
        if(not out_file):
            out_file = ("%s_%s_coh_viz.html" % (self.name,self.party))
        fundings = []
        for i,district in self.districts.items():
            total = (district.cash+district.expenditures)/district.json_538['vep']
            fundings.append((str(i), district.cash/district.json_538['vep'], district.expenditures/district.json_538['vep'], total))
        cash = list(zip(*sorted(fundings,key= lambda x: x[3])[::-1]))

        trace2 = go.Bar(
            x=cash[0],
            y=cash[1],
            name='Unspent',
        )
        trace1 = go.Bar(
            x=cash[0],
            y=cash[2],
            name='Spent'
        )

        data = [trace1, trace2]
        layout = go.Layout(
            title=('%s %s per Voter Finances' % (self.name, self.party)),
            barmode='stack',
            xaxis=dict(type='category',tickangle=45),
            yaxis=dict(tickformat="$",hoverformat = '$.2f')
        )

        fig = go.Figure(data=data, layout=layout)
        plt.iplot(fig, filename='stacked-bar',show_link=False, config={'displayModeBar': False})


    def gen_coh_graph(self,out_file=None):
        "This code is a mess, but this generates a cash on hand per eligible voter per district barchart"

        if(not out_file):
            out_file = ("%s_%s_coh_viz.html" % (self.name,self.party))
        fundings = []
        for i,district in self.districts.items():
            fundings.append((str(i), district.raised_indiv/district.json_538['vep']))
        cash = list(zip(*sorted(fundings,key= lambda x: x[1])[::-1])) #This line of complicated function magic sorts the two lists by size

        trace1 = go.Bar(
            x=cash[0],
            y=cash[1],
            name='Cash per voter',
        )

        data = [trace1]#[trace1, trace2]
        layout = go.Layout(
            title='%s Dollars RAISED per Voter (%s)' % (self.party, self.name),
            xaxis=dict(type='category',tickangle=45),
            yaxis=dict(tickformat="$",hoverformat = '$.2f')
        )
        fig = go.Figure(data=data,layout=layout)
        plt.iplot(fig, filename=out_file,show_link=False, config={'displayModeBar': False})



    def add_district(self, number, party=None):
        """Adds a district to a state by its number
        """
        if(not party):
            party=self.party
        dist = Campaign(self.name,number, party=party)
        self.districts[number] = dist
        return dist

    def build_from_FEC(self,cash_df):
        """Builds out a State (ie. populates its districts) from an FEC housesenate finance dataframe
        """
        #Pare down the relevant dataframe by party and state
        campaigns = cash_df[(cash_df[4]==self.party) & (cash_df[18]==self.name)]
        
        #Checks if given district is in dictionary
        for idx, candidate in campaigns.iterrows():
            if(candidate[19]==0):
                continue #We ignore senate campaigns for now!
            elif(not str(candidate[19]) in self.districts):
                dist = Campaign(self.name,candidate[19],party=self.party)
                self.districts[str(candidate[19])] = dist
                dist.set_coh(candidate[10])
                dist.set_expenditures(candidate[7])
                dist.set_raised_indiv(candidate[17])
            else:
                dist = self.districts[str(candidate[19])]
                dist.set_coh(max(dist.cash, candidate[10]))
                dist.set_expenditures(max(dist.expenditures,candidate[7]))
                dist.set_raised_indiv(max(dist.raised_indiv, candidate[17]))

        
        campaigns = cash_df[(cash_df[4]==self.opp) & (cash_df[18]==self.name)]
        
        #Checks if given district is in dictionary
        for idx, candidate in campaigns.iterrows():
            if(candidate[19]==0):
                continue #We ignore senate campaigns for now!
            elif(not self.districts[str(candidate[19])].opposition ):
                dist = Campaign(self.name,candidate[19],party=self.opp)
                self.districts[str(candidate[19])].opposition = dist
                dist.set_coh(candidate[10])
                dist.set_expenditures(candidate[7])
                dist.set_raised_indiv(candidate[17])
            else:
                dist = self.districts[str(candidate[19])].opposition
                dist.set_coh(max(dist.cash, candidate[10]))
                dist.set_expenditures(max(dist.expenditures,candidate[7]))
                dist.set_raised_indiv(max(dist.raised_indiv, candidate[17]))

def build_all_from_538(party):
    """Builds out entire country dictionary from 538 data
    """
    state_dict = {}
    for district in candidates_list:
        state = district[0]
        if state not in state_dict:
            state_obj = State(state,party=party)
            state_dict[state] = state_obj 
        else:
            state_obj = state_dict[state]
        district_number = district[1]
        if not district_number in state_obj.districts:
            dist = state_obj.add_district(district_number)
        else:
            dist = state_obj.districts[district_number]

        # this url stores FiveThirtyEight's prediction data for each district
        url = 'https://projects.fivethirtyeight.com/2018-midterm-election-forecast/house/{state}-{district_number}.json'.format(state=state, district_number=district_number)
        response = get(url)
        loaded_json = json.loads(response.text)
        
        dist.json_538 = loaded_json
    return state_dict

def build_all(fec_df,party='DEM'):
    print("Borrowing some data from Nate Silvers...")
    states = build_all_from_538(party)
    print("Building in FEC campaign finance details...")
    for state in states.values():
        state.build_from_FEC(fec_df)
    print("Pickling data as all_states.pickle")
    with open('all_states.pickle', 'wb') as handle:
        pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done!")
    return states

def gen_full_pvi_scatter(states):
    pvi_list = []
    share_list = []
    for state in states.values():
        for i,district in state.districts.items():
            pvi = district.json_538['pvi']
            share = district.funding_share()
            pvi_list.append(pvi)
            share_list.append(share)
    regr = linear_model.LinearRegression()
    shaped_share = np.array(share_list).reshape(-1,1)
    regr.fit(shaped_share,pvi_list)

    trace = go.Scatter(
        x = 100*share_list,
        y = pvi_list,
        mode = 'markers'
    )
    trace2 = go.Scatter(
        x=share_list, 
        y=regr.predict(shaped_share),
        mode='lines',
        line=dict(color='blue', width=3)
    )

    data = [trace,trace2]
    layout = go.Layout(
        title=('Fundraising share by Cook PVI (Nationwide)'),
        showlegend=False,
        annotations=[
            dict(
                x=0.1,
                y=43,
                xref='x',
                yref='y',
                text='R squared = %.3f' % regr.score(shaped_share,pvi_list),
                showarrow=False,
            )],
            xaxis=dict(title='Share of Fundraising'),
            yaxis=dict(title='Cook PVI')
    )

    fig = go.Figure(data=data, layout=layout)

    # Plot and embed in ipython notebook!
    plt.iplot(fig, filename='basic-scatter',show_link=False, config={'displayModeBar': False})



def rebuild_fec(states, fec_df):
    for state in states.values():
        state.build_from_FEC(fec_df)
    return states
    
def load(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)