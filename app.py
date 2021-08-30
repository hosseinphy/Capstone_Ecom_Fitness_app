###############################################
# Changed by Hossein Azizi
# Changed date: 03/21/2019
# Licensce: free to use
#############################################

#import local_data as sample_data
from flask import Flask, render_template, request
from altair import Chart, X, Y, Axis, Data, DataFormat
import pandas as pd
import numpy as np
import altair as alt
import json
import pickle
import dill
# load a simple dataset as a pandas DataFrame
from vega_datasets import data


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import base

#from utils import  GroupbyEstimator, category_factory


#cars = data.cars()
#electricity = data.iowa_electricity()
#barley_yield = data.barley()

class GroupbyEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self, column, estimator_factory):
        # column is the value to group by; estimator_factory can be
        # called to produce estimators
        self.column = column
        self.est_fact = estimator_factory
        self.est_dict = {}
    
    def fit(self, X, y):
        X = X.copy()
        X['label'] = y
        # Create an estimator and fit it with the portion in each group
        for key, df_city in X.groupby(self.column):
            self.est_dict[key] = self.est_fact().fit(df_city, df_city['label'])
        return self

    def predict(self, X):
        X = X.copy()
        X['label'] = 0
        predict_dic = {}
        cities = X[self.column].unique().tolist()        

        for key, df_city in X.groupby(self.column):
            predict_dic[key] = self.est_dict[key].predict(df_city)
                                
        ordered_predict_list = [predict_dic[k] for k in cities]
        return np.concatenate(ordered_predict_list)


def category_factory():
    
    min_tree_splits = range(2,6)
    min_tree_leaves = range(2,8)
    nmax_features = range(0,10)
    max_tree_depth = range(0,20)

    # categorical_columns = ['Quarter','Month', 'Week', 'Dayofyear', 'Day']
    categorical_columns = ['Year','Month', 'Week', 'Day','Quarter']
    numeric_columns = ['price', 'ReleaseNumber']
    trans_columns = ColumnTransformer([
        ('numeric', 'passthrough', numeric_columns),
        ('categorical','passthrough' , categorical_columns)

    ])

    features = Pipeline([
        ('columns', trans_columns),
        ('scaler', MaxAbsScaler()),
    ])
    

    param_grid = {
                  'max_depth' : max_tree_depth,
                  'max_features':nmax_features,
                  'min_samples_leaf':min_tree_leaves 
                 }

    gs = GridSearchCV(
                        DecisionTreeRegressor(min_samples_split=2), 
                        param_grid, cv=40, n_jobs=2
                     )

    
    pipe = Pipeline([('feature', features), ('gs_est', gs)])
    
    return pipe




goog = pd.read_json('google_trends.json')

# Flask app
#####################
app = Flask(__name__)


#def get_dropdown_values():
#
#    """
#    dummy function, replace with e.g. database call. If data not change, this function is not needed but dictionary
#    could be defined globally
#    """
#
#    class_entry_relations = {'Cardio': ['Treadmill', 'Indoor bike'],
#                             'Strength Fitness': ['Dumbells', 'Weight Lift']}
#
#    return class_entry_relations
#
#
#@app.route('/_update_dropdown')
#def update_dropdown():
#
#    # the value of the first dropdown (selected by the user)
#    selected_class = request.args.get('selected_class', type=str)
#
#    # get values for the second dropdown
#    updated_values = get_dropdown_values()[selected_class]
#
#    # create the value sin the dropdown as a html string
#    html_string_selected = ''
#    for entry in updated_values:
#        html_string_selected += '<option value="{}">{}</option>'.format(entry, entry)
#
#    return jsonify(html_string_selected=html_string_selected)
#
#
#@app.route('/_process_data')
#def process_data():
#    selected_class = request.args.get('selected_class', type=str)
#    selected_entry = request.args.get('selected_entry', type=str)
#    # process the two selected values here and return the response; here we just create a dummy string
#    return jsonify(random_text="you selected {} and {}".format(selected_class, selected_entry))
#
#
#
#@app.route('/')
#def index():
#
#    """
#    Initialize the dropdown menues
#    """
#
#    class_entry_relations = get_dropdown_values()
#
#    default_classes = sorted(class_entry_relations.keys())
#    default_values = class_entry_relations[default_classes[0]]
#
#    return render_template('test.html',
#                           all_classes=default_classes,
#                           all_entries=default_values)
#


# render index.html home page
@app.route("/")
def index():
    return render_template('test.html')


# render predict.html page
@app.route("/predict")
def show_predicts():
    return render_template("predict.html")


##########################
# Functions & classes
##########################

def convert_to_datetime(col):
    return pd.to_datetime(col)


# unpickle the predictive model
#with open('cat_model.pkl', 'rb') as file:
#    cat_model = pickle.load(file)


#dump the model
with open('cat_model.dill', 'rb') as f:
    cat_model = dill.load(f)

#app.vars={}
#app.questions={}
#app.questions['Please choose one catagory to plot?']=('Closing price','Adjusted closing price','Opening price')
#app.nquestions=len(app.questions)
#should be 1

##########################
# Flask routes
##########################
# render index.html home page

@app.route("/", methods=['GET', 'POST'])
def test():

	if request.method == 'POST':

		# Load historical and test data
		hist_data = pd.read_json('hist_dataset.json')
		true_data = pd.read_json('true_dataset.json')

		# categories for predictions
		cat_dic = {
			'0' : 'dumbells',
			'1' : 'elliptical',
			'2' : 'weight lift',
			'3' : 'treadmill',
			'4' : 'indoor bike',
		}
			
		# what category to predict
		cat_to_pred = cat_dic[str(request.form['F_type'])]

		# now clean test data, add features and make predictions
		# create a test dataFrame
		true_df = true_data.drop(columns=['title', 'sku', 'NewReleaseFlag'])

		# select corresponding collumns
		true_df = true_df.loc[true_df.product_category == cat_to_pred]

		true_df["Year"] = true_df.date.dt.year
		true_df["Month"] = true_df.date.dt.month
		true_df["Week"] = true_df.date.dt.week
		true_df["Weekday"] = true_df.date.dt.weekday
		true_df["Day"] = true_df.date.dt.day
		true_df["Dayofyear"] = true_df.date.dt.dayofyear
		true_df["Quarter"] = true_df.date.dt.quarter

		true_df = true_df.drop(columns=['date'])


		# create a y_test vector
		y_test = true_df['inventory']


		# predict the label
		y_pred = cat_model.predict(true_df)

		# now plot the values
		dh = hist_data.loc[hist_data.product_category == cat_to_pred]#pd.read_json('treadmill_hist_data.json')
		dh_gr = pd.DataFrame(dh.groupby('date').inventory.sum().reset_index())
		dt = true_data.loc[true_data.product_category == cat_to_pred]
		dt_gr = pd.DataFrame(dt.groupby('date').inventory.sum().reset_index())

		# now create a new df where the columnlabel is replaced by the predicted label
		dp = dt
		dp['inventory'] = y_pred
		dp_gr = pd.DataFrame(dp.groupby('date').inventory.sum().reset_index())


		chart1 = alt.Chart(dh_gr
				).mark_line(  
				   opacity = 0.8,
				    strokeWidth=2
				).encode(
				x=alt.X("date:T", axis=alt.Axis(title='')),
				y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
				color=alt.value("blue"), 
			).properties(title="Historical/validation data for {}".format(cat_to_pred),width=800,height=200
			)

		chart2 = alt.Chart(dt_gr
				).mark_line(  
				   opacity = 0.8,
				    strokeWidth=2
				).encode(
				x=alt.X("date:T", axis=alt.Axis(title='')),
				y=alt.Y("inventory:Q", axis=alt.Axis(title='Inventory')),
				color=alt.value("red"), 
			).properties(title="Historical/validation data for {}".format(cat_to_pred),width=800,height=200
			)

		combined1 = chart1 + chart2 

		dh_gr = dh_gr[dh_gr.date >= pd.to_datetime('2021-05-01')]
		dt_gr = dt_gr[dt_gr.date >= pd.to_datetime('2021-05-01')]
		dp_gr = dp_gr[dp_gr.date >= pd.to_datetime('2021-05-01')]

		chart3 = alt.Chart(dh_gr
				).mark_line(  
				   opacity = 0.8,
				    strokeWidth=2
				).encode(
				x=alt.X("date:T", axis=alt.Axis(title='')),
				y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
				color=alt.value("blue"), 
			).properties(title="",width=800,height=200
			)

		chart4 = alt.Chart(dt_gr
				).mark_line(  
				   opacity = 0.8,
				    strokeWidth=2
				).encode(
				x=alt.X("date:T", axis=alt.Axis(title='')),
				y=alt.Y("inventory:Q", axis=alt.Axis(title='Inventory')),
				color=alt.value("red"), 
			).properties(title="",width=800,height=200
			)

		chart5 = alt.Chart(dp_gr
				).mark_line(  
				   opacity = 0.8,
				    strokeWidth=2
				).encode(
				x=alt.X("date:T", axis=alt.Axis(title='Date')),
				y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
				color=alt.value("darkorange"), 
			).properties(title="",width=800,height=200
			)

		combined2 = chart3 + chart4 + chart5

		combined = combined1 & combined2

		return render_template('pred_res.html', combined=combined.to_json())

		#return render_template('test.html', pred=str(pred))
	return render_template('test.html')


@app.route("/bike_predict")
def chart():

	#plot figure
	dh = pd.read_json('bike_hist_data.json')
	dh_gr = pd.DataFrame(dh.groupby('date').inventory.sum().reset_index())
	dt = pd.read_json('bike_true_data.json')
	dt_gr = pd.DataFrame(dt.groupby('date').inventory.sum().reset_index())
	dp = pd.read_json('bike_pred_data.json')
	dp_gr = pd.DataFrame(dp.groupby('date').inventory.sum().reset_index())


	chart1 = alt.Chart(dh_gr
			).mark_line(  
			   opacity = 0.8,
			    strokeWidth=2
			).encode(
			x=alt.X("date:T", axis=alt.Axis(title='')),
			y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
			color=alt.value("blue"), 
			tooltip = ["date", "inventory"]
		).properties(title="",width=800,height=200
		).interactive()

	chart2 = alt.Chart(dt_gr
			).mark_line(  
			   opacity = 0.8,
			    strokeWidth=2
			).encode(
			x=alt.X("date:T", axis=alt.Axis(title='')),
			y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
			color=alt.value("red"), 
			tooltip = ["date", "inventory"]
		).properties(title="",width=800,height=200
		).interactive()

	combined1 = chart1 + chart2 


	dh_gr = dh_gr[dh_gr.date >= pd.to_datetime('2021-05-01')]
	dt_gr = dt_gr[dt_gr.date >= pd.to_datetime('2021-05-01')]
	dp_gr = dp_gr[dp_gr.date >= pd.to_datetime('2021-05-01')]


	chart3 = alt.Chart(dh_gr
			).mark_line(  
			   opacity = 0.8,
			    strokeWidth=2
			).encode(
			x=alt.X("date:T", axis=alt.Axis(title='')),
			y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
			color=alt.value("blue"),
			tooltip = ["date", "inventory"]
		).properties(title="",width=800,height=200
		).interactive()

	chart4 = alt.Chart(dt_gr
			).mark_line(  
			   opacity = 0.8,
			    strokeWidth=2
			).encode(
			x=alt.X("date:T", axis=alt.Axis(title='')),
			y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
			color=alt.value("red"), 
			tooltip = ["date", "inventory"]
		).properties(title="",width=800,height=200
		).interactive()

	chart5 = alt.Chart(dp_gr
			).mark_line(  
			   opacity = 0.8,
			    strokeWidth=2
			).encode(
			x=alt.X("date:T", axis=alt.Axis(title='')),
			y=alt.Y("inventory:Q", axis=alt.Axis(title='')),
			color=alt.value("darkorange"), 
			tooltip = ["date", "inventory"]
		).properties(title="",width=800,height=200
		).interactive()

	combined2 = chart3 + chart4 + chart5

	combined = combined1 & combined2
	return combined.to_json()


### Altair Data Routes
#########################

WIDTH = 600
HEIGHT = 300

@app.route("/data/googel_trends")
def data_google_trends():

        result = goog
        # reshape the dataframe
        result_melted = result.melt('date', var_name='Product', value_name='Trends')

        # group rsults for each month
        result_grouped = result.groupby(pd.Grouper(key='date',freq='M')).sum()

        df = result_melted
        df_grouped = df.groupby('Product').sum()
        df_grouped = df_grouped.reset_index()

        strength_df = result_grouped.loc[:,'Dumbells':'Weight lifting Plates']
        cardio_df = result_grouped.loc[:,'treadmill':'Step machine']
        result_grouped['Cardio fitness'] = result_grouped.loc[:,'treadmill':'Step machine'].sum(axis=1)
        result_grouped['Strength fitness'] = result_grouped.loc[:,'Dumbells':'Weight lifting Plates'].sum(axis=1)

        df2 = result_grouped[['Cardio fitness','Strength fitness']]
        df2 = df2.reset_index()
        df2 = df2.melt('date', var_name='Fitness_type', value_name='Trends')

        size_hist = (alt.Chart(df_grouped, width=500, height=200).mark_bar()
            .encode(x="Product:N",
                    y="Trends",
                    color=alt.value('blue') )
                    )

        chart2 = alt.Chart(df2
                        ).mark_area(
                           opacity = 0.8,
                            strokeWidth=1
                        ).encode(
                        x="date:T",
                        y="Trends:Q",
                        color="Fitness_type:N",
                        tooltip=['date', 'Fitness_type', 'Trends']
                ).properties(title="",width=800,height=200
                )


        combined = chart2 & size_hist

        return combined.to_json()

@app.route("/data/stock_fitness")
def data_stock():
	alt.data_transformers.disable_max_rows()
	daily_data = pd.read_json('fitness_data.json')

	grouped_features = ["date", "Year", "Quarter","Month", "Week", "Weekday", "Dayofyear", "Day", "sku", "product"]

	src1 = pd.DataFrame(daily_data.groupby(["date", "prod_type", "prod_class"]).inventory.sum())
	src1 = src1.reset_index()
	src1['date'] = src1['date'].apply(convert_to_datetime)

	src2 = pd.DataFrame(daily_data.groupby(["date", "prod_type", "prod_class"]).inventory.sum().rolling(window=30, center=True).mean())
	src2 = src2.reset_index()
	src2['date'] = src2['date'].apply(convert_to_datetime)

	chart = alt.Chart(src1
			).mark_area(
			    point={
			      "color": 'orange',  
			      "filled": False,
			      "fill": "white",
			      "size":2  
			    },
			    strokeWidth=1
			   ).encode(
			alt.X('date:T', axis=alt.Axis(title='Date')),
			alt.Y('inventory:Q', axis=alt.Axis(title='Inventory',format='f')),
			tooltip=['prod_type:N', 'inventory'],
			color = 'prod_class:N',
		).properties(title="",width=800,height=200
		).interactive()

	mean = alt.Chart(src2
			).mark_line(  
			    color='black',
			    strokeWidth=2
			).encode(
			x="date:T",
			y="inventory:Q", 
			color = 'prod_class:N',
			tooltip=['date', 'inventory']
		).properties(title="",width=800,height=200
		).interactive()


	return chart.to_json()



@app.route("/data/bike_df")
def data_bike_df():
	daily_data = pd.read_json('bike_data.json')

	# price
	src_pr = pd.DataFrame(daily_data.groupby("date").price.sum())
	src_pr = src_pr.reset_index()
	src_pr['date'] = src_pr['date'].apply(convert_to_datetime)

	src_mpr = pd.DataFrame(daily_data.groupby("date").price.sum().rolling(window=20, center=True).mean())
	src_mpr = src_mpr.reset_index()
	src_mpr['date'] = src_mpr['date'].apply(convert_to_datetime)


	# inventory
	src1 = pd.DataFrame(daily_data.groupby("date").inventory.sum())
	src1 = src1.reset_index()
	src1['date'] = src1['date'].apply(convert_to_datetime)

	src2 = pd.DataFrame(daily_data.groupby("date").inventory.sum().rolling(window=30, center=True).mean())
	src2 = src2.reset_index()
	src2['date'] = src2['date'].apply(convert_to_datetime)


	chart_pr = alt.Chart(src_pr
			).mark_line(
			    color="blue",
			    point={
			      "color": 'blue',  
			      "filled": False,
			      "fill": "white",
			      "size":2  
			    },
			    strokeWidth=1
			   ).encode(
			    alt.X('date:T', axis=alt.Axis(title='Date')),
			    alt.Y('price:Q', axis=alt.Axis(title='Price',format='$f'))
		
		)

	mean_pr = alt.Chart(src_mpr
			).mark_line(  
			    color='Purple',
			    strokeWidth=2
			).encode(
			x="date:T",
			y="price:Q", 
			tooltip=['date', 'price']
		).properties(title="",width=800,height=200
		).interactive()


	combined_pr = chart_pr + mean_pr



	chart_inv = alt.Chart(src1
			).mark_line(
			    color="orange",
			    point={
			      "color": 'orange',  
			      "filled": False,
			      "fill": "white",
			      "size":2  
			    },
			    strokeWidth=1
			   ).encode(
			alt.X('date:T', axis=alt.Axis(title='Date')),
			alt.Y('inventory:Q', axis=alt.Axis(title='Inventory',format='f'))
	    
		)

	mean_inv = alt.Chart(src2
			).mark_line(  
			    color='red',
			    strokeWidth=1
			).encode(
			x="date:T",
			y="inventory:Q", 
			tooltip=['date', 'inventory']
		).properties(title="",width=800,height=200
		).interactive()


	combined_inv = chart_inv + mean_inv


	weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
	yearmonth = ["Jan-2019", "Feb-2019", "Mar-2019", "Apr-2019", "May-2019",
		     "Jun-2019", "Jul-2019", "Aug-2019", "Sep-2019", "Oct-2019", "Nov-2019", 
		     "Dec-2019", "Jan-2020", "Feb-2020", "Mar-2020", "Apr-2020", "May-2020",
		     "Jun-2020", "Jul-2020", "Aug-2020", "Sep-2020", "Oct-2020", "Nov-2020", 
		     "Dec-2020", "Jan-2021", "Feb-2021", "Mar-2021", "Apr-2021", "May-2021",
		     "Jun-2021", "Jul-2021"            
		    ]

	daily_grouped = daily_data.groupby("Weekday").inventory.sum()

	monthly_grouped = daily_data.groupby(["Year", "Month"]).inventory.sum()

	daily_grouped = pd.DataFrame(daily_grouped)
	daily_grouped['Days'] = weekdays
	daily_grouped = daily_grouped.reset_index().drop(columns='Weekday')

	#monthly_grouped = pd.DataFrame(monthly_grouped)
	#monthly_grouped['Year-month'] = yearmonth
	#monthly_grouped = monthly_grouped.reset_index().drop(columns=['Year', 'Month'])


	monthly_grouped = pd.DataFrame(daily_data.groupby(["Year", "Month"]).inventory.sum()).reset_index()
	monthly_grouped['day'] = len(monthly_grouped) * [1]
	monthly_grouped['Year-month'] = pd.to_datetime(monthly_grouped[['Year', 'Month', 'day']])
	monthly_grouped = monthly_grouped.drop(columns=['Year', 'Month'])


#	bike_monthyear = alt.Chart(monthly_grouped
#			).mark_line(
#			    color="blue",
#			    point=True,
#			    strokeWidth=3
#			   ).encode(
#			alt.X('Year-month:T', axis=alt.Axis(title='')),
#			alt.Y('inventory:Q', axis=alt.Axis(title='',format='f')),
#			#tooltip=['Year-month:T', 'inventory']
#		).properties(title="Total inventory per month ",width=400,height=200).interactive()


	# Create a selection that chooses the nearest point & selects based on x-value
	nearest = alt.selection(type='single', nearest=True, on='mouseover',
				fields=['Year-month'], empty='none')

	line = alt.Chart().mark_line(point=True, color='blue').encode(
	    alt.X('Year-month:T', axis=alt.Axis(title='')),
	    alt.Y('inventory:Q', axis=alt.Axis(title='',format='f')),
	)

	# Transparent selectors across the chart. This is what tells us
	# the x-value of the cursor
	selectors = alt.Chart().mark_point().encode(
	    x='Year-month:T',
	    opacity=alt.value(0),
	).add_selection(
	    nearest
	)

	# Draw points on the line, and highlight based on selection
	points = line.mark_point().encode(
	    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
	)

	# Draw text labels near the points, and highlight based on selection
	text = line.mark_text(align='left', dx=5, dy=-5).encode(
	    text=alt.condition(nearest, 'inventory:Q', alt.value(' '))
	)

	# Draw a rule at the location of the selection
	rules = alt.Chart().mark_rule(color='gray').encode(
	    x='Year-month:T',
	).transform_filter(
	    nearest
	)

	# Put the five layers into a chart and bind the data
	bike_monthyear = alt.layer(line, selectors, points, rules, text,
			       data=monthly_grouped, 
			       width=400, height=200,title='Total inventory per month')

	# bike_monthyear.save('stocks.html')


	bike_weekdays = alt.Chart(daily_grouped
			).mark_circle(
			    color="blue",
#			    point=False,
#			    strokeWidth=3
			   ).encode(
			x=alt.X('Days:N', sort=weekdays),
			y="inventory:Q",
			tooltip=['Days', 'inventory']
		).properties(title="Total inventory per day ",width=400,height=200).interactive()

	combined_inv2 = bike_monthyear | bike_weekdays

	combined = combined_pr & combined_inv & combined_inv2
	
	return combined.to_json()




@app.route("/data/df_matrix")
def data_matrix():
	df_quantile = pd.read_json('data_quantile.json')
	
	chart = alt.Chart(df_quantile).mark_point().encode(
	    alt.X(alt.repeat("column"), type='quantitative'),
	    alt.Y(alt.repeat("row"), type='quantitative'),
	    color=alt.value("steelblue"),
	    opacity=alt.value(0.4)
	).properties(
	    width=150,
	    height=150
	).repeat(
	    row=['price', 'inventory', 'ReleaseNumber'],
	    column=['price', 'inventory', 'ReleaseNumber']
	).interactive()
	
	return chart.to_json()



@app.route("/data/feat_imp")
def data_fimp():
	fimp = pd.read_json('feature_importance.json')
	chart = alt.Chart(fimp
			).mark_bar(  
			   opacity = 0.8,
			    strokeWidth=1
			).encode(
			x=alt.X("importance:Q", axis=alt.Axis(title='')),
			y=alt.Y("feature:N", axis=alt.Axis(title='')),
			color=alt.value("blue"), 
		).properties(title="",width=800,height=200
		)

	return chart.to_json()




if __name__ == "__main__":
    app.run(debug=True)
